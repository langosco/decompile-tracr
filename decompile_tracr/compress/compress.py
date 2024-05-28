from typing import Optional
import functools

import sklearn.decomposition
import jax
import chex
from jaxtyping import ArrayLike
import jax.numpy as jnp

import jax
import chex
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
import einops
import haiku as hk

from tracr.compiler.assemble import AssembledTransformerModel
from tracr.transformer.model import CompiledTransformerModel

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.sample import sample
from decompile_tracr.dataset import compile
from decompile_tracr.compress import metrics
from decompile_tracr.compress import autoencoder
from decompile_tracr.compress.utils import AssembledModelInfo


from metamodels_for_rasp.train import Updater, TrainState


@chex.dataclass
class CompressedModel:
    tracr_model: AssembledTransformerModel
    wenc: ArrayLike
    wdec: ArrayLike

    def __post_init__(self):
        self.original_size = self.tracr_model.params[
            'pos_embed']['embeddings'].shape[-1]
        self.hidden_size = self.wenc.shape[-1]
        self._cm: CompiledTransformerModel = get_compiled_model(
            self.tracr_model)
        self.params = update_params(
            self.tracr_model.params, self.wenc, self.wdec, None,
        )
    
    def __call__(self, x):
        """Forward pass through the compressed model."""
        fwd = hk.without_apply_rng(hk.transform(
            lambda x: self._hk_forward(x)))
        return fwd.apply(self.params, x)

    def _hk_forward(self, x: ArrayLike):
        token_embed = hk.Embed(
            embedding_matrix=self.params['token_embed']['embeddings'], 
            name="token_embed",
        )
        position_embed = hk.Embed(
            embedding_matrix=self.params['pos_embed']['embeddings'], 
            name="pos_embed",
        )

        def unembed(x, use_unembed_argmax):
            x = x @ self.wdec  # decode back to original size
            return self._cm.unembed(x, use_unembed_argmax)
        
        unembed = hk.to_module(unembed)()
        
        output = CompiledTransformerModel(
            transformer=self._cm.transformer,
            token_embed=token_embed,
            position_embed=position_embed,
            unembed=unembed,
            use_unembed_argmax=self._cm.use_unembed_argmax,
            pad_token=self._cm.pad_token,
        )(x, use_dropout=False)
        return output

    def encode_activations(self, x: ArrayLike):
        return x @ self.wenc
    
    def decode_activations(self, x: ArrayLike):
        return x @ self.wdec


@chex.dataclass
class Residuals:
    inputs: ArrayLike  # integer array (batch_size, seq_len)
    residuals: ArrayLike  # float array (batch_size, layers, seq_len, d_model)


class ResidualsSampler:
    """Helper class to sample transformer activations."""
    def __init__(self, model: AssembledTransformerModel):
        info = AssembledModelInfo(model=model)
        self.model = model
        self.seq_len = info.seq_len
        self.bos = info.bos

    def sample_tokens(
        self, key: jax.random.PRNGKey, batch_size: int = 512):
        """Utility function to sample a sequence of input tokens.
        """
        bos = self.bos * jnp.ones((batch_size, 1), dtype=int)
        inputs = jax.random.randint(
            key, (batch_size, self.seq_len-1), 0, 5)
        inputs = jnp.concatenate([bos, inputs], axis=1)
        return inputs

#    @functools.partial(jax.jit, static_argnames=('self','flatten_leading_axes'))
    def sample_residuals(
        self, 
        key: jax.random.PRNGKey,
        batch_size: int,
        flatten_leading_axes: bool = True,
    ) -> Residuals:
        inputs = self.sample_tokens(key, batch_size=batch_size)
        out = self.model.forward(self.model.params, inputs)
        res = out.transformer_output.residuals
        embeddings = out.transformer_output.input_embeddings
        res = einops.rearrange([embeddings, *res], 'l b s d -> b l s d')
        if flatten_leading_axes:
            res = einops.rearrange(res, 'b l s d -> (b l s) d')
        return Residuals(inputs=inputs, residuals=res)


def train_svd(
    key: jax.random.PRNGKey,
    assembled_model: AssembledTransformerModel,
    hidden_size: int,
) -> dict:
    """A cheaper method that is ~equivalent to a linear autoencoder."""
    N = 2**7
    d_model = AssembledModelInfo(model=assembled_model).d_model
    residuals_sampler = ResidualsSampler(model=assembled_model)
    svd = sklearn.decomposition.TruncatedSVD(n_components=hidden_size)
    data = residuals_sampler.sample_residuals(key, batch_size=N).residuals
    svd.fit(data)
    wenc = svd.transform(np.identity(d_model))
    wdec = svd.inverse_transform(np.identity(hidden_size))
    aux = dict(svd=svd)
    return (CompressedModel(
        tracr_model=assembled_model, wenc=wenc, wdec=wdec,
    ), aux)


def init_autoencoder(
    key: jax.random.PRNGKey, 
    model: AssembledTransformerModel,
    hidden_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.bfloat16,
    lr: Optional[float] = 1e-3,
) -> tuple[Updater, TrainState]:
    d_model = AssembledModelInfo(model=model).d_model
    if hidden_size is None:
        hidden_size = int(d_model // 1.2)

    ae = autoencoder.Autoencoder(
        hidden_size=hidden_size, 
        output_size=d_model,
        use_bias=False,
        dtype=dtype,
        tie_embeddings=False,
    )
    loss_fn = autoencoder.AutoencoderLoss(ae.apply)
    optimizer = optax.adam(lr)
    updater = Updater(
        opt=optimizer,
        model=ae,
        loss_fn=loss_fn,
    )
    residuals_sampler = ResidualsSampler(model=model)
    subkey1, subkey2 = jax.random.split(key)
    train_state = updater.init_train_state(
        rng=subkey1, 
        inputs=residuals_sampler.sample_residuals(
            subkey2, batch_size=1).residuals,
    )
    return updater, train_state


def train_autoencoder(
    key: jax.random.PRNGKey,
    assembled_model: AssembledTransformerModel,
    nsteps: int = 100,
    **init_args,
) -> tuple[CompressedModel, dict]:
    BATCH_SIZE = 2**8
    updater, state = init_autoencoder(
        key, assembled_model, **init_args)
    residuals_sampler = ResidualsSampler(model=assembled_model)
    log = []

    @jax.jit
    def step(state):
        state.rng, subkey = jax.random.split(state.rng)
        batch = residuals_sampler.sample_residuals(
            subkey, batch_size=BATCH_SIZE).residuals
        state, aux = updater.update(state, batch)
        return state, aux

    for _ in range(nsteps):
        state, aux = step(state)
        log.append(aux)

    wenc, wdec = autoencoder.get_wenc_and_wdec(state.params)
    aux = dict(model=updater.model, state=state, log=log, 
               hidden_size=updater.model.hidden_size, original_size=wenc.shape[0])
    return (CompressedModel(
        tracr_model=assembled_model, wenc=wenc, wdec=wdec), aux)


def get_metrics(assembled_model: AssembledTransformerModel, 
                apply_fn: callable):
    """apply_fn computes decode(encode(x))."""
    try:
        metric_fn = metrics.Accuracy(assembled_model=assembled_model)
        name = "accuracy"
    except AssertionError:
        metric_fn = metrics.MSE(assembled_model=assembled_model)
        name = "mse"

    residuals_sampler = ResidualsSampler(model=assembled_model)

    test_data = residuals_sampler.sample_residuals(
        jax.random.key(0), batch_size=2**15, flatten_leading_axes=False
    ).residuals[:, -1]
    return {name: metric_fn(test_data, apply_fn(test_data))}



@jax.jit
def update_params(
    model_params: dict, 
    wenc: ArrayLike, 
    wdec: ArrayLike, 
    w_orth: ArrayLike,
):
    """Return new set of transformer params that operates on the 
    compressed residual stream."""
    if w_orth is not None:
        wenc = wenc @ w_orth
        wdec = w_orth.T @ wdec

    new_params = {k: {kk: None for kk in v.keys()} 
                  for k, v in model_params.items()}

    for layer in model_params.keys():
        name = layer.split("/")[-1]
        if "embed" in layer:
            assert name in ['pos_embed', 'token_embed'], name
            new_params[layer]['embeddings'] = (
                model_params[layer]['embeddings'] @ wenc
            )
        elif "attn" in layer and name == "linear":
            new_params[layer]['w'] = model_params[layer]['w'] @ wenc
            new_params[layer]['b'] = model_params[layer]['b'] @ wenc
        elif "attn" in layer:
            assert name in ['query', 'key', 'value']
            new_params[layer]['w'] = wdec @ model_params[layer]['w']
            new_params[layer]['b'] = model_params[layer]['b']
        elif layer.endswith("mlp/linear_1"):
            new_params[layer]['w'] = wdec @ model_params[layer]['w']
            new_params[layer]['b'] = model_params[layer]['b']
        elif layer.endswith("mlp/linear_2"):
            new_params[layer]['w'] = model_params[layer]['w'] @ wenc
            new_params[layer]['b'] = model_params[layer]['b'] @ wenc
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    return new_params
        
    
def get_compiled_model(model: AssembledTransformerModel
                       ) -> CompiledTransformerModel:
    """Extract a compiled model object from an assembled model."""
    hk_transformed = hk.without_apply_rng(hk.transform(
        model.get_compiled_model))
    return hk_transformed.apply(model.params)


if __name__ == "__main__":
    # Train an autoencoder to compress the residual stream
    # of a Tracr-complied model:
    rng = np.random.default_rng(None)
    key = jax.random.key(0)

    # sample a program and compile it
    toks = tokenizer.tokenize(sample.sample(rng, 5, only_categorical=True))
    m = compile.compile_tokens_to_model(toks)
    accuracy = metrics.Accuracy(assembled_model=m)
    d_model = m.params['token_embed']['embeddings'].shape[-1]
    hidden_size = int(d_model // 1.1)
    print()
    print("Compiled model dim:", d_model)
    print("Hidden size:", hidden_size)

    # train autoencoder
    key, subkey = jax.random.split(key)
    train_out = train_autoencoder(
        subkey, m, nsteps=50_000, lr=2e-3, hidden_size=hidden_size)
    state, log, model = (
        train_out['state'], train_out['log'], train_out['model'])

    plt.plot([l["train/loss"] for l in log])
    plt.yscale('log')
    plt.xscale('log')

    print('Loss:', log[-1]["train/loss"])


    # metrics
    embed = metrics.Embed(assembled_model=m)
    unembed = metrics.Unembed(assembled_model=m)
    decode = metrics.Decode(assembled_model=m)

    residuals_sampler = ResidualsSampler(model=m)
    key, subkey = jax.random.split(key)
    test_data = residuals_sampler.sample_residuals(
        subkey, batch_size=2**12, flatten_leading_axes=False)
    decoded = model.apply({'params': state.params}, test_data.residuals)
    decoded = np.array(decoded, dtype=np.float32)

    print("Acc: ", accuracy(test_data.residuals[:, -1], decoded[:, -1]))
