import jax
import chex
import haiku as hk
from tracr.compiler.assemble import AssembledTransformerModel


class AssembledModelInfo:
    def __init__(self, model: AssembledTransformerModel):
        self.d_model: int = model.params['token_embed']['embeddings'].shape[-1]
        self.key_size: int = model.model_config.key_size
        self.mlp_dim: int = model.model_config.mlp_hidden_size
        self.num_heads: int = model.model_config.num_heads
        self.num_layers: int = model.model_config.num_layers
        self.seq_len: int = model.input_encoder._max_seq_len
        self.vocab_size: int = model.input_encoder.vocab_size
        self.bos: int = model.input_encoder.bos_encoding
        self.use_unembed_argmax = hk.transform(
            model.get_compiled_model).apply(
                {}, jax.random.key(0)).use_unembed_argmax

