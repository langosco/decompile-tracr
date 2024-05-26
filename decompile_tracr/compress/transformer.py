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
import einops

from decompile_tracr.compress.autoencoder import Residuals
from decompile_tracr.compress import autoencoder

from tracr.transformer.model import CompiledTransformerModel
from tracr.compiler.assemble import AssembledTransformerModel

from metamodels_for_rasp.train import TrainState


@struct.dataclass
class TransformerConfig:
    vocab_size: int
#    output_vocab_size: int
    dtype: Any = jnp.bfloat16
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
    posemb_init: Optional[Callable] = nn.initializers.normal(stddev=0.02)
    decode: bool = False


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert x.ndim == 3, (
            "Number of dimensions should be 3, but it is: %d" % x.ndim
        )
        length = x.shape[1]
        pos_emb_shape = (1, config.max_len, x.shape[-1])
        pos_embedding = self.param(
            "pos_embedding", config.posemb_init, pos_emb_shape,
        )
        pos_embedding = pos_embedding[:, :length, :]
        return x + pos_embedding


class MLPBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        config = self.config
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        x = nn.elu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=deterministic
        )
        return x


class AttentionBlock(nn.Module):
    """Transformer encoder layer."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, deterministic, causal_mask):
        config = self.config

        # Attention block.
        assert x.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(x)
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
        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=deterministic)
        return x


class Embed(nn.Module):
    """Embedding layer."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, is_training: bool = True):
        config = self.config

        assert inputs.ndim == 2
        x = nn.Embed(
            num_embeddings=config.vocab_size, 
            features=config.emb_dim, 
            name="embed",
        )(inputs)

        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=not is_training)
        x = AddPositionEmbs(config)(x)
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
        activations['inputs'] = x
        config = self.config
        chex.assert_shape(x, (None, None))  # batch, seq
        x = Embed(config)(x, is_training=is_training)
        activations['Embed_0'] = x

        for layer in range(config.num_layers):
            x += AttentionBlock(config)(
                x, deterministic=not is_training, causal_mask=None)
            activations[f'AttentionBlock_{layer}'] = x
            x += MLPBlock(config)(x, deterministic=not is_training)
            activations[f'MLPBlock_{layer}'] = x

        return x, dict(activations)


def get_loss_fn(apply_fn):
    """Compute loss in a way that allows us to take 
    gradients wrt a given layer individually. That is,
    jax.grad(loss_fn)(layer_params, params, batch)
    will treat the `params` argument as constant and only compute
    gradients wrt `layer_params`.
    """
    def loss_fn(layer_params: dict, params: dict, batch: Residuals,
                ) -> tuple[float, dict]:
        assert len(layer_params.keys()) == 1
        layer_name = list(layer_params.keys())[0]
        params = params.copy()
        params.update(layer_params)  # inject layer params
        _, activations = apply_fn({"params": params}, batch.inputs)
        chex.assert_equal_shape([activations[layer_name], batch.residuals])
        loss = jnp.mean(jnp.square(
            activations[layer_name] - batch.residuals
        ))
        aux = {f'loss/{layer_name}': loss}
        return loss, aux
    return loss_fn


def get_compiled_model(model: AssembledTransformerModel
                       ) -> CompiledTransformerModel:
    """Extract a compiled model object from an assembled model."""
    hk_transformed = hk.without_apply_rng(hk.transform(
        model.get_compiled_model))
    return hk_transformed.apply(model.params)


def get_transformer_config(
    model: AssembledTransformerModel,
    d_model: int,
):
    """Initialize transformer model for training.
    Return updater and initial training state."""
    cm = get_compiled_model(model)
    cfg = model.model_config
    # TODO: should we depend on cm.use_unembed_argmax?
    # Ie should we differentiate between categorical and numerical
    # output?

    if cm.use_unembed_argmax:
        pass
    else:
        pass

    return TransformerConfig(
        vocab_size=model.input_encoder.vocab_size,
#        output_vocab_size=model.output_encoder.vocab_size,
        emb_dim=d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        qkv_dim=cfg.key_size*cfg.num_heads,
        mlp_dim=d_model*2,
        max_len=model.input_encoder._max_seq_len,
        dropout_rate=0.0,  # no dropout on the residual stream
        attention_dropout_rate=0.0,
        decode=False,
    )


def init_transformer(
    key: jax.random.PRNGKey,
    config: TransformerConfig,
    lr: float,
) -> tuple[Transformer, TrainState]:
    transformer = Transformer(config=config)
    optimizer = optax.adam(lr)
    key, subkey = jax.random.split(key)
    variables = transformer.init(subkey, np.ones((1, 1), dtype=np.int32))
    key, subkey = jax.random.split(key)
    state = TrainState(
        step=0,
        rng=subkey,
        opt_state=optimizer.init(variables['params']),
        params=variables['params'],
    )
    return transformer, optimizer, state


@chex.dataclass
class DataGenerator:
    # TODO: maybe make this an iterator instead?
    assembled_model: AssembledTransformerModel
    encode: callable
    batch_size: int
    seq_len: int

    def __post_init__(self):
        assert self.seq_len <= self.assembled_model.input_encoder._max_seq_len
        self._residuals_sampler = autoencoder.ResidualsSampler(
            model=self.assembled_model, 
            seq_len=self.seq_len, 
            batch_size=self.batch_size,
            flatten_leading_axes=False,
        )

    def __call__(self, key: jax.random.PRNGKey):
        """Return dict of residuals for each layer. Keys are layer names."""
        res = self._residuals_sampler.sample_residuals(key)
        res.residuals = self.encode(res.residuals)
        x = einops.rearrange(res.residuals, 'b l s d -> l b s d')
        return {k: Residuals(inputs=res.inputs, residuals=v) 
            for k, v in zip(layer_names(), x)}



@chex.dataclass
class TransformerTrainingArgs:
    nsteps: int
    learning_rate: float = 1e-4


def train_transformer(
    key: jax.random.PRNGKey,
    get_batch: DataGenerator,
    args: TransformerTrainingArgs,
) -> tuple[dict, list, Transformer]:
    """Train a transformer to layer-wise match the compressed residuals
    of a Tracr-compiled model.
    """
    dummy_batch = get_batch(key)
    d_model = dummy_batch['Embed_0'].residuals.shape[-1]
    config = get_transformer_config(get_batch.assembled_model, d_model)
    key, subkey = jax.random.split(key)
    model, opt, state = init_transformer(
        subkey, config, lr=args.learning_rate)

    loss_fn = get_loss_fn(model.apply)

    def compute_layerwise_grads(params: dict, batch: dict[str, Residuals]):
        assert set(batch.keys()) == set(params.keys())
        grads = {}
        aux = {}
        for k, v in params.items():
            g, a = jax.grad(loss_fn, has_aux=True)({k: v}, params, batch[k])
            grads.update(g)
            aux.update(a)
        return grads, aux

    @jax.jit
    def step(state: TrainState):
        state.rng, subkey = jax.random.split(state.rng)
        batch = get_batch(subkey)
        grads, aux = compute_layerwise_grads(state.params, batch)
        updates, state.opt_state = opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        aux = {f'train/{k}': v for k, v in aux.items()}
        return state, aux

    log = []
    for _ in range(args.nsteps):
        state, aux = step(state)
        log.append(aux)
    
    return model, state, log


def layer_names():
    yield 'Embed_0'
    for i in range(30):
        yield f'AttentionBlock_{i}'
        yield f'MLPBlock_{i}'
    raise NotImplementedError(
        f"Maximum number of layers ({i}) exceeded.")