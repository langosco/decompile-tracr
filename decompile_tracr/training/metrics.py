import jax.numpy as jnp
import numpy as np
import chex
import haiku as hk

from tracr.compiler.assemble import AssembledTransformerModel
from tracr.transformer.encoder import CategoricalEncoder


@chex.dataclass
class Embed:
    assembled_model: AssembledTransformerModel

    def __post_init__(self):
        @hk.without_apply_rng
        @hk.transform
        def _embed(tokens):
            compiled_model = self.assembled_model.get_compiled_model()
            return compiled_model.embed(tokens)

        self.embed = lambda tokens: _embed.apply(
            self.assembled_model.params, tokens)
    
    def __call__(self, tokens):
        return self.embed(tokens)


@chex.dataclass
class Unembed:
    assembled_model: AssembledTransformerModel

    def __post_init__(self):
        @hk.without_apply_rng
        @hk.transform
        def _unembed(x):
            cm = self.assembled_model.get_compiled_model()
            return cm.unembed(x, use_unembed_argmax=cm.use_unembed_argmax)

        self.unembed = lambda x: _unembed.apply(
            self.assembled_model.params, x)

        @hk.without_apply_rng
        @hk.transform
        def _use_unembed_argmax():
            cm = self.assembled_model.get_compiled_model()
            return cm.use_unembed_argmax

        self.use_unembed_argmax = _use_unembed_argmax.apply(
            self.assembled_model.params)

    def __call__(self, tokens):
        return self.unembed(tokens)


@chex.dataclass
class Accuracy:
    assembled_model: AssembledTransformerModel

    def __post_init__(self):
        self.unembed = Unembed(assembled_model=self.assembled_model)
        assert isinstance(
            self.assembled_model.output_encoder, CategoricalEncoder)
        assert self.unembed.use_unembed_argmax

    def __call__(self, x, y):
        x, y = self.unembed(x), self.unembed(y)
        x, y = x[1:], y[1:]  # ignore compiler_bos
        return jnp.mean(x == y)


@chex.dataclass
class MSE:
    assembled_model: AssembledTransformerModel

    def __post_init__(self):
        self.unembed = Unembed(assembled_model=self.assembled_model)
        assert not isinstance(
            self.assembled_model.output_encoder, CategoricalEncoder)
        assert not self.unembed.use_unembed_argmax

    def __call__(self, x, y):
        x, y = self.unembed(x), self.unembed(y)
        x, y = x[1:], y[1:]  # ignore compiler_bos
        return jnp.mean((x - y)**2)


@chex.dataclass
class Decode:
    assembled_model: AssembledTransformerModel

    def __post_init__(self):
        self.unembed = Unembed(assembled_model=self.assembled_model)

    def __call__(self, x):
        unembedded = np.squeeze(self.unembed(x))
        unembedded = unembedded.tolist()
        tokens = self.assembled_model.output_encoder.decode(unembedded)
        return ['compiler_bos'] + tokens[1:]
