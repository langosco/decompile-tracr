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
from decompile_tracr.dataset.reconstruct import ModelFromParams


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


class Compresser:
    def __init__(self, wenc, wdec):
        self.wenc = wenc
        self.wdec = wdec
        self.original_size, self.hidden_size = wenc.shape

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
    def __init__(self, model: ModelFromParams):
        self.model = model
        self.seq_len = model.seq_len

    def sample_tokens(self, key: jax.random.PRNGKey, batch_size: int = 512):
        return jax.random.randint(key, (batch_size, self.seq_len), 0, 5)

    def sample_residuals(
        self, 
        key: jax.random.PRNGKey,
        batch_size: int,
        flatten_leading_axes: bool = True,
    ) -> Residuals:
        inputs = self.sample_tokens(key, batch_size=batch_size)
        out = self.model.from_tokens.apply(self.model.params, inputs)
        res = out.residuals
        embeddings = out.input_embeddings
        res = einops.rearrange([embeddings, *res], 'l b s d -> b l s d')
        if flatten_leading_axes:
            res = einops.rearrange(res, 'b l s d -> (b l s) d')
        return Residuals(inputs=inputs, residuals=res)


def train_svd(model: ModelFromParams, hidden_size: int):
    """A cheaper method that is ~equivalent to a linear autoencoder."""
    N = 2**6
    residuals_sampler = ResidualsSampler(model)
    svd = sklearn.decomposition.TruncatedSVD(n_components=hidden_size)
    key = jax.random.key(0)
    data = residuals_sampler.sample_residuals(key, batch_size=N).residuals
    svd.fit(data)
    wenc = svd.transform(np.identity(model.d_model))
    wdec = svd.inverse_transform(np.identity(hidden_size))
    aux = dict(svd=svd)
    return wenc, wdec, aux


def init_autoencoder(
    key: jax.random.PRNGKey, 
    model: ModelFromParams,
    hidden_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.bfloat16,
    lr: Optional[float] = 2e-3,
) -> tuple[Updater, TrainState]:
    if hidden_size is None:
        hidden_size = int(model.d_model // 1.2)

    ae = autoencoder.Autoencoder(
        hidden_size=hidden_size, 
        output_size=model.d_model,
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
    residuals_sampler = ResidualsSampler(model)
    subkey1, subkey2 = jax.random.split(key)
    train_state = updater.init_train_state(
        rng=subkey1, 
        inputs=residuals_sampler.sample_residuals(
            subkey2, batch_size=1).residuals,
    )
    return updater, train_state


def train_autoencoder(
    key: jax.random.PRNGKey,
    model: ModelFromParams,
    nsteps: int = 50_000,
    **init_args,
):
    BATCH_SIZE = 2**8
    updater, state = init_autoencoder(
        key, model=model, **init_args)
    residuals_sampler = ResidualsSampler(model)
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
    return wenc, wdec, aux


def get_metrics(assembled_model: AssembledTransformerModel, 
                apply_fn: callable):
    """apply_fn computes decode(encode(x))."""
    try:
        metric_fn = metrics.Accuracy(assembled_model=assembled_model)
        name = "accuracy"
    except AssertionError:
        metric_fn = metrics.MSE(assembled_model=assembled_model)
        name = "mse"

    h = AssembledModelInfo(assembled_model).num_heads
    residuals_sampler = ResidualsSampler(
        model=ModelFromParams(assembled_model.params, num_heads=h))

    test_data = residuals_sampler.sample_residuals(
        jax.random.key(0), batch_size=2**15, flatten_leading_axes=False
    ).residuals[:, -1]
    return {name: metric_fn(test_data, apply_fn(test_data))}



@jax.jit
def update_params(
    params: dict, 
    wenc: ArrayLike, 
    wdec: ArrayLike, 
    w_orth: ArrayLike = None,
):
    """Return new set of transformer params that operates on the 
    compressed residual stream."""
    if w_orth is not None:
        wenc = wenc @ w_orth
        wdec = w_orth.T @ wdec

    new_params = {k: {kk: None for kk in v.keys()} 
                  for k, v in params.items()}

    for layer in params.keys():
        name = layer.split("/")[-1]
        if "embed" in layer:
            assert name in ['pos_embed', 'token_embed'], name
            new_params[layer]['embeddings'] = (
                params[layer]['embeddings'] @ wenc)
        elif "attn" in layer and name == "linear":
            new_params[layer]['w'] = params[layer]['w'] @ wenc
            new_params[layer]['b'] = params[layer]['b'] @ wenc
        elif "attn" in layer:
            assert name in ['query', 'key', 'value']
            new_params[layer]['w'] = wdec @ params[layer]['w']
            new_params[layer]['b'] = params[layer]['b']
        elif layer.endswith("mlp/linear_1"):
            new_params[layer]['w'] = wdec @ params[layer]['w']
            new_params[layer]['b'] = params[layer]['b']
        elif layer.endswith("mlp/linear_2"):
            new_params[layer]['w'] = params[layer]['w'] @ wenc
            new_params[layer]['b'] = params[layer]['b'] @ wenc
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
    info = AssembledModelInfo(model=m)
    model = ModelFromParams(m.params, num_heads=info.num_heads)
    wenc, wdec, aux = train_autoencoder(
        subkey, model, nsteps=50_000, lr=2e-3, hidden_size=hidden_size)
    state, log, aenc = (
        aux['state'], aux['log'], aux['model'])

    plt.plot([l["train/loss"] for l in log])
    plt.yscale('log')
    plt.xscale('log')

    print('Loss:', log[-1]["train/loss"])


    # metrics
    embed = metrics.Embed(assembled_model=m)
    unembed = metrics.Unembed(assembled_model=m)
    decode = metrics.Decode(assembled_model=m)

    residuals_sampler = ResidualsSampler(model=model)
    key, subkey = jax.random.split(key)
    test_data = residuals_sampler.sample_residuals(
        subkey, batch_size=2**12, flatten_leading_axes=False)
    decoded = aenc.apply({'params': state.params}, test_data.residuals)
    decoded = np.array(decoded, dtype=np.float32)

    print("Acc: ", accuracy(test_data.residuals[:, -1], decoded[:, -1]))
