from typing import Optional

import jax
import chex
from jaxtyping import ArrayLike
import jax.numpy as jnp
import flax.linen as nn

import jax
import chex
import jax.numpy as jnp


class Autoencoder(nn.Module):
    hidden_size: int
    output_size: int
    dtype: Optional[jnp.dtype] = jnp.float32
    use_bias: Optional[bool] = False
    tie_embeddings: Optional[bool] = False

    def setup(self):
        self._encoder = nn.Dense(
            self.hidden_size, dtype=self.dtype, use_bias=self.use_bias)
        self._decoder = nn.Dense(
            self.output_size, dtype=self.dtype, use_bias=self.use_bias)

    @nn.compact
    def __call__(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.output_size))
        return self.decode(self.encode(x))

    def encode(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.output_size))
        return self._encoder(x)

    def decode(self, x, is_training=False):
        del is_training
        chex.assert_shape(x, (..., self.hidden_size))
        if self.tie_embeddings:
            w = self._encoder.variables['params']['kernel'].T
            return self._decoder.apply({'params': {'kernel': w}}, x)
        else:
            return self._decoder(x)


class AutoencoderLoss:
    """MSE loss function for the autoencoder."""
    def __init__(self, apply: callable):
        self.apply = apply

    def __call__(
        self,
        params: dict, 
        key: jax.random.PRNGKey, 
        x: ArrayLike,
    ) -> float:
        lam = 0.8
        y = self.apply({"params": params}, x)
        l2_loss = jnp.mean((x - y)**2) / 2
        l1_loss = jnp.mean(jnp.abs(x - y))
        loss = lam * l2_loss + (1 - lam) * l1_loss
        aux = {"loss": loss}
        return loss, aux


def get_wenc_and_wdec(autoencoder_params: dict
                      ) -> tuple[ArrayLike, ArrayLike]:
    """Extract encoder and decoder weights from autoencoder params.
    """
    wenc = autoencoder_params['_encoder']['kernel']
    if '_decoder' in autoencoder_params:
        wdec = autoencoder_params['_decoder']['kernel']
    else:
        wdec = wenc.T
    return wenc, wdec