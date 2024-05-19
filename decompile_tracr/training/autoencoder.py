from typing import Optional
import functools

import sklearn.decomposition
import jax
import chex
from jaxtyping import ArrayLike
import jax.numpy as jnp
import flax.linen as nn

import jax
import chex
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
import einops

from tracr.compiler.assemble import AssembledTransformerModel

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling
from decompile_tracr.dataset import compile
from decompile_tracr.training import metrics

from metamodels_for_rasp.train import Updater, TrainState


def train_svd(
    key: jax.random.PRNGKey,
    assembled_model: AssembledTransformerModel,
    hidden_size: int,
):
    """A cheaper method that is ~equivalent to a linear autoencoder."""
    N = 2**15
    d_model = assembled_model.params['token_embed']['embeddings'].shape[-1]
    seq_len = assembled_model.input_encoder._max_seq_len
    residuals_sampler = ResidualsSampler(
        model=assembled_model, seq_len=seq_len, batch_size=N)
    svd = sklearn.decomposition.TruncatedSVD(n_components=hidden_size)
    data = residuals_sampler.sample_residuals(key).residuals
    svd.fit(data)
    wenc = svd.transform(np.identity(d_model))
    wdec = svd.inverse_transform(np.identity(hidden_size))

    def apply_fn(x):
        return x @ wenc @ wdec

    out = dict(wenc=wenc, wdec=wdec, mse=None, accuracy=None)
    out.update(get_metrics(assembled_model, apply_fn))
    return out


# autoencoder to compress the residual stream

class Autoencoder(nn.Module):
    hidden_size: int
    output_size: int
    dtype: Optional[jnp.dtype] = jnp.float32
    use_bias: Optional[bool] = False
    tie_embeddings: Optional[bool] = False

    def setup(self):
        self.encoder = nn.Dense(
            self.hidden_size, dtype=self.dtype, use_bias=self.use_bias)
        self.decoder = nn.Dense(
            self.output_size, dtype=self.dtype, use_bias=self.use_bias)

    @nn.compact
    def __call__(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.output_size))
        return self.decoder(self.encoder(x))

    def encode(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.output_size))
        return self.encoder(x)

    def decode(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.hidden_size))
        if self.tie_embeddings:
            w = self.encoder.variables['Dense_0']['kernel']
            return self.decoder.apply({'params': {'kernel': w}}, x)
        else:
            return self.decoder(x)


def get_loss_fn(apply_fn):
    """Return MSE loss function for the autoencoder.
    Usage:
    model = Autoencoder(hidden_size=64, output_size=128)
    loss_fn = get_loss_fn(model.apply)
    """
    def loss_fn(
        params: dict, key: jax.random.PRNGKey, x: ArrayLike,
    ) -> float:
        lam = 0.8
        y = apply_fn({"params": params}, x)
        l2_loss = jnp.mean((x - y)**2) / 2
        l1_loss = jnp.mean(jnp.abs(x - y))
        loss = lam * l2_loss + (1 - lam) * l1_loss
        aux = {"loss": loss}
        return loss, aux
    return loss_fn


@chex.dataclass
class Residuals:
    inputs: ArrayLike  # integer array (batch_size, seq_len)
    residuals: ArrayLike  # float array (batch_size, layers, seq_len, d_model)


@chex.dataclass(frozen=True)
class ResidualsSampler:
    """When flatten_leading_axes is True, the first three axes (batch,
    layer, seq_len) are flattened into one axis."""
    model: AssembledTransformerModel
    seq_len: int
    batch_size: int
    flatten_leading_axes: bool = True


    def sample_tokens(self, key: jax.random.PRNGKey):
        """Utility function to sample a sequence of input tokens.
        """
        bos: int = self.model.input_encoder.bos_encoding
        inputs = jax.random.randint(
            key, (self.batch_size, self.seq_len-1), 0, 5)
        inputs = jnp.concatenate(
            [bos * jnp.ones((self.batch_size, 1), dtype=int), inputs], axis=1)
        return inputs

#    @functools.partial(jax.jit, static_argnames=('self',))
    def sample_residuals(self, key: jax.random.PRNGKey) -> Residuals:
        inputs = self.sample_tokens(key)
        out = self.model.forward(self.model.params, inputs)
        res = out.transformer_output.residuals
        embeddings = out.transformer_output.input_embeddings
        res = einops.rearrange([embeddings, *res], 'l b s d -> b l s d')
        if self.flatten_leading_axes:
            res = einops.rearrange(res, 'b l s d -> (b l s) d')
        return Residuals(inputs=inputs, residuals=res)


def init_autoencoder(
    key: jax.random.PRNGKey, 
    model: AssembledTransformerModel,
    hidden_size: Optional[int] = None,
    dtype: Optional[jnp.dtype] = jnp.bfloat16,
    lr: Optional[float] = 1e-3,
) -> tuple[Updater, TrainState]:
    d_model = model.params['token_embed']['embeddings'].shape[-1]
    if hidden_size is None:
        hidden_size = int(d_model // 1.2)
#        hidden_size = 25
    ae = Autoencoder(
        hidden_size=hidden_size, 
        output_size=d_model,
        dtype=dtype,
        tie_embeddings=False,
    )
    loss_fn = get_loss_fn(ae.apply)
    optimizer = optax.adam(lr)
    updater = Updater(
        opt=optimizer,
        model=ae,
        loss_fn=loss_fn,
    )
    residuals_sampler = ResidualsSampler(model=model, seq_len=5, batch_size=1)
    subkey1, subkey2 = jax.random.split(key)
    train_state = updater.init_train_state(
        rng=subkey1, 
        inputs=residuals_sampler.sample_residuals(subkey2).residuals,
    )
    return updater, train_state


def train_autoencoder(
    key: jax.random.PRNGKey,
    assembled_model: AssembledTransformerModel,
    nsteps: int = 100,
    lr: float = 1e-3,
    hidden_size: Optional[int] = None,
) -> dict:
    BATCH_SIZE = 2**10
    seq_len = assembled_model.input_encoder._max_seq_len
    key, subkey = jax.random.split(key)
    updater, state = init_autoencoder(
        subkey, assembled_model, lr=lr, hidden_size=hidden_size)
    residuals_sampler = ResidualsSampler(
        model=assembled_model, seq_len=seq_len, batch_size=BATCH_SIZE)
    log = []

    @jax.jit
    def step(state):
        state.rng, subkey = jax.random.split(state.rng)
        batch = residuals_sampler.sample_residuals(subkey).residuals
        state, aux = updater.update(state, batch)
        return state, aux

    for _ in range(nsteps):
        state, aux = step(state)
        log.append(aux)

    def apply_fn(x):
        return updater.model.apply({'params': state.params}, x)
    
    out = dict(state=state, log=log, model=updater.model, 
               mse=None, accuracy=None)
    out.update(get_metrics(assembled_model, apply_fn))
    return out


def get_metrics(assembled_model: AssembledTransformerModel, 
                apply_fn: callable):
    """apply_fn computes decode(encode(x))."""
    seq_len = assembled_model.input_encoder._max_seq_len
    try:
        metric_fn = metrics.Accuracy(assembled_model=assembled_model)
        name = "accuracy"
    except AssertionError:
        metric_fn = metrics.MSE(assembled_model=assembled_model)
        name = "mse"

    residuals_sampler = ResidualsSampler(
        model=assembled_model, seq_len=seq_len, batch_size=2**15, 
        flatten_leading_axes=False)

    key = jax.random.key(0)
    test_data = residuals_sampler.sample_residuals(key).residuals[:, -1]
    return {name: metric_fn(test_data, apply_fn(test_data))}



def get_wenc_and_wdec(autoencoder_params: dict
                      ) -> tuple[ArrayLike, ArrayLike]:
    """Extract encoder and decoder weights from autoencoder params.
    """
    wenc = autoencoder_params['encoder']['kernel']
    if 'decoder' in autoencoder_params:
        wdec = autoencoder_params['decoder']['kernel']
    else:
        wdec = wenc.T
    return wenc, wdec


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
        
    






if __name__ == "__main__":
    # Train an autoencoder to compress the residual stream
    # of a Tracr-complied model:
    rng = np.random.default_rng(None)
    key = jax.random.key(0)

    # sample a program and compile it
    toks = tokenizer.tokenize(sampling.sample(rng, 5, only_categorical=True))
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

    residuals_sampler = ResidualsSampler(
        model=m,
        seq_len=6,
        batch_size=2**12,
        flatten_leading_axes=False,
    )
    key, subkey = jax.random.split(key)
    test_data = residuals_sampler.sample_residuals(subkey)
    decoded = model.apply({'params': state.params}, test_data.residuals)
    decoded = np.array(decoded, dtype=np.float32)


    print("Acc: ", accuracy(test_data.residuals[:, -1], decoded[:, -1]))