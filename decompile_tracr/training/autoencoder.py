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

from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling

from metamodels_for_rasp.train import Updater


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

    def encode(self, x):
        chex.assert_shape(x, (..., self.output_size))
        self.encoder(x)

    def decode(self, x):
        chex.assert_shape(x, (..., self.hidden_size))
        self.decoder(x)


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


def get_residuals_sampler(
    model: AssembledTransformerModel, 
    seq_len: int, 
    batch_size: int,
    flatten_leading_axes: bool = True,
) -> callable:
    d_model = model.params['token_embed']['embeddings'].shape[-1]

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
        inputs = jax.random.randint(key, (batch_size, seq_len), 0, 5)
        inputs = jnp.concatenate(
            [bos * jnp.ones((batch_size, 1), dtype=int), inputs], axis=1)
        return embed.apply(model.params, inputs)

    @jax.jit
    def get_residuals(key: jax.random.PRNGKey) -> ArrayLike:
        embeddings = sample_embeddings(key)
        out = transformer.apply(model.params, embeddings)
        res = jnp.concatenate(out.residuals)
        chex.assert_shape(res, (None, seq_len+1, d_model))
        if flatten_leading_axes:
            res = einops.rearrange(res, 'b s d -> (b s) d')
            embeddings = einops.rearrange(embeddings, 'b s d -> (b s) d')
        return res, embeddings

    return get_residuals


def init_autoencoder(key: jax.random.PRNGKey, 
                     model: AssembledTransformerModel):
    d_model = model.params['token_embed']['embeddings'].shape[-1]
    ae = Autoencoder(hidden_size=d_model//2, output_size=d_model)
    loss_fn = get_loss_fn(ae.apply)
    optimizer = optax.adam(1e-4)
    updater = Updater(
        opt=optimizer,
        model=ae,
        loss_fn=loss_fn,
    )
    get_residuals = get_residuals_sampler(model, seq_len=5, batch_size=10)
    subkey1, subkey2 = jax.random.split(key)
    train_state = updater.init_train_state(subkey1, get_residuals(subkey2)[0])
    return updater, train_state

