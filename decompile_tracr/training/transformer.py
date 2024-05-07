from typing import Callable, Any, Optional
from collections import defaultdict

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
import numpy as np
import chex
import optax
import haiku as hk

from tracr.transformer.model import CompiledTransformerModel
from tracr.compiler.assemble import AssembledTransformerModel

from metamodels_for_rasp.train import Updater


@struct.dataclass
class TransformerConfig:
    vocab_size: int
#    output_vocab_size: int
    dtype: Any = jnp.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 2048
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.1
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None
    decode: bool = False


def sinusoidal_init(max_len=2048):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
          inputs: input data.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3, but it is: %d" % inputs.ndim
        )
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=config.max_len)(
                None, pos_emb_shape, None
            )
        else:
            pos_embedding = self.param(
                "pos_embedding", config.posemb_init, pos_emb_shape
            )
        pos_embedding = pos_embedding[:, :length, :]
        return inputs + pos_embedding


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        inputs = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.elu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=deterministic
        )
        return output


class AttentionBlock(nn.Module):
    """Transformer encoder layer."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic, causal_mask):
        config = self.config

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
            decode=config.decode,
        )(x, mask=causal_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        return x


class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x,
        *,
        is_training: bool = True,
    ):
        activations = defaultdict(list)
        config = self.config
        chex.assert_shape(x, (None, None))  # batch, seq
#        causal_mask = np.tril(np.ones((1, 1, config.max_len, config.max_len)))

        x = nn.Embed(
            num_embeddings=config.vocab_size, 
            features=config.emb_dim, 
            name="embed",
        )(x)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not is_training)
        x = AddPositionEmbs(config)(x)

        for _ in range(config.num_layers):
            x += AttentionBlock(config)(
                x, deterministic=not is_training, causal_mask=None)
            activations['residuals'].append(x)
            x += MLPBlock(config)(x, deterministic=not is_training)
            activations['residuals'].append(x)

        x = nn.LayerNorm(dtype=config.dtype)(x)
#        logits = nn.Dense(
#            config.output_vocab_size,
#            kernel_init=config.kernel_init,
#            bias_init=config.bias_init,
#        )(x)
#        return logits
        activations['residuals'] = jnp.stack(activations['residuals'], axis=1)
        chex.assert_shape(activations['residuals'], 
                          (None, config.num_layers*2, None, config.emb_dim))
        return x, activations


def get_loss_fn(apply_fn):
    """Construct loss fn for layer-wise training."""
    def loss_fn(params, rngkey, data: dict):
        inputs, target_residuals = data['inputs'], data['target_residuals']
        _, activations = apply_fn({"params": params}, inputs)
        aux = {}
        return jnp.mean((activations['residuals'] - target_residuals)**2), aux
    return loss_fn


def get_compiled_model(model: AssembledTransformerModel
                       ) -> CompiledTransformerModel:
    hk_transformed = hk.without_apply_rng(hk.transform(
        model.get_compiled_model))
    return hk_transformed.apply(model.params)


def init_transformer(key: jax.random.PRNGKey,
                     model: AssembledTransformerModel):
    cm = get_compiled_model(model)
    cfg = model.model_config
    # TODO: should we depend on cm.use_unembed_argmax?
    # Ie should we differentiate between 

    if cm.use_unembed_argmax:
        pass
    else:
        pass

    config = TransformerConfig(
        vocab_size=model.input_encoder.vocab_size,
#        output_vocab_size=model.output_encoder.vocab_size,
        emb_dim=cm.position_embed.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        qkv_dim=cfg.key_size,
        mlp_dim=cfg.mlp_hidden_size,
        max_len=model.input_encoder._max_seq_len,
        dropout_rate=0.0,  # no dropout on the residual stream
        attention_dropout_rate=0.0,
        decode=False,
    )

    transformer = Transformer(config=config)
    optimizer = optax.adam(1e-4)
    loss_fn = get_loss_fn(transformer.apply)
    updater = Updater(
        opt=optimizer,
        model=transformer,
        loss_fn=loss_fn,
    )

    key, subkey = jax.random.split(key)
    data = {
        'inputs': np.random.choice(10, size=(10, 5)),
        'target_residuals': np.random.rand(
            10, config.num_layers*2, 5, config.emb_dim),
    }
    state = updater.init_train_state(subkey, data['inputs'])
    return transformer, updater, state