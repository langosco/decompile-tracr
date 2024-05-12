from typing import Optional

import jax
import chex
from jaxtyping import ArrayLike
import jax.numpy as jnp
import flax.linen as nn

import jax
import chex
from jaxtyping import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import haiku as hk
import optax
import einops

from tracr.compiler.assemble import AssembledTransformerModel

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling
from decompile_tracr.dataset import compile

from metamodels_for_rasp.train import Updater, TrainState


# autoencoder to compress the residual stream

class Encoder(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        return x


class Decoder(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.output_size)(x)
        return x


class Autoencoder(nn.Module):
    hidden_size: int
    output_size: int

    def setup(self):
        self.encoder = Encoder(self.hidden_size)
        self.decoder = Decoder(self.output_size)

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
        y = apply_fn({"params": params}, x)
        loss = jnp.mean((x - y)**2)
        aux = {"loss": loss}
        return loss, aux
    return loss_fn


@chex.dataclass
class Residuals:
    inputs: ArrayLike  # integer array (batch_size, seq_len)
    residuals: ArrayLike  # float array (batch_size, layers, seq_len, d_model)


def get_residuals_sampler(
    model: AssembledTransformerModel, 
    seq_len: int, 
    batch_size: int,
    flatten_leading_axes: bool = True,
) -> callable:
    @hk.without_apply_rng
    @hk.transform
    def embed(tokens):
        compiled_model = model.get_compiled_model()
        return compiled_model.embed(tokens)

    @hk.without_apply_rng
    @hk.transform
    def transformer(embeddings: ArrayLike):
        """embeddings must be float arrays 
        of shape (batch_size, seq_len, d_model)
        """
        compiled_model = model.get_compiled_model()
        return compiled_model.transformer(
            embeddings, jnp.ones(embeddings.shape[:-1]), use_dropout=False)

    def sample_embeddings(key: jax.random.PRNGKey):
        """Utility function to sample embeddings for 
        a random sequence of input tokens.
        """
        bos: int = model.input_encoder.bos_encoding
        inputs = jax.random.randint(key, (batch_size, seq_len-1), 0, 5)
        inputs = jnp.concatenate(
            [bos * jnp.ones((batch_size, 1), dtype=int), inputs], axis=1)
        return embed.apply(model.params, inputs), inputs

    @jax.jit
    def get_residuals(key: jax.random.PRNGKey) -> Residuals:
        embeddings, inputs = sample_embeddings(key)
        out = transformer.apply(model.params, embeddings)
        res = out.residuals
        res = einops.rearrange([embeddings, *res], 'l b s d -> b l s d')
        if flatten_leading_axes:
            res = einops.rearrange(res, 'b l s d -> (b l s) d')
        return Residuals(inputs=inputs, residuals=res)

    return get_residuals


def init_autoencoder(
    key: jax.random.PRNGKey, 
    model: AssembledTransformerModel,
    hidden_size: Optional[int] = None,
) -> tuple[Updater, TrainState]:
    d_model = model.params['token_embed']['embeddings'].shape[-1]
    if hidden_size is None:
        hidden_size = int(d_model // 1.5)
    ae = Autoencoder(hidden_size=hidden_size, output_size=d_model)
    loss_fn = get_loss_fn(ae.apply)
    optimizer = optax.adam(3e-3)
    updater = Updater(
        opt=optimizer,
        model=ae,
        loss_fn=loss_fn,
    )
    get_residuals = get_residuals_sampler(model, seq_len=5, batch_size=10)
    subkey1, subkey2 = jax.random.split(key)
    train_state = updater.init_train_state(
        rng=subkey1, 
        inputs=get_residuals(subkey2).residuals,
    )
    return updater, train_state


def train_autoencoder(
    key: jax.random.PRNGKey,
    assembled_model: AssembledTransformerModel,
    nsteps: int = 100
):
    BATCH_SIZE = 128
    seq_len = assembled_model.input_encoder._max_seq_len
    key, subkey = jax.random.split(key)
    updater, state = init_autoencoder(subkey, assembled_model)
    get_residuals = get_residuals_sampler(
        assembled_model, seq_len=seq_len, batch_size=BATCH_SIZE)
    log = []

    for _ in range(nsteps):
        key, subkey = jax.random.split(key)
        batch = get_residuals(subkey).residuals
        state, aux = updater.update(state, batch)
        log.append(aux)
    
    return state, log, updater.model


if __name__ == "__main__":
    # Train an autoencoder to compress the residual stream
    # of a Tracr-complied model:
    rng = np.random.default_rng(0)
    key = jax.random.key(0)

    # sample a program and compile it
    toks = tokenizer.tokenize(sampling.sample(rng, 5))
    m = compile.compile_tokens_to_model(toks)

    # train autoencoder
    state, log, model = train_autoencoder(key, m, nsteps=1000)

    plt.plot([l["train/loss"] for l in log])
    plt.yscale('log')
    plt.xscale('log')

    print('final loss:', log[-1]["train/loss"])