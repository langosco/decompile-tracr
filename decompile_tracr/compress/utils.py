import chex
from tracr.compiler.assemble import AssembledTransformerModel


@chex.dataclass
class AssembledModelInfo:
    model: AssembledTransformerModel

    def __post_init__(self):
        self.d_model: int = self.model.params['token_embed']['embeddings'].shape[-1]
        self.key_size: int = self.model.model_config.key_size
        self.mlp_dim: int = self.model.model_config.mlp_hidden_size
        self.num_heads: int = self.model.model_config.num_heads
        self.num_layers: int = self.model.model_config.num_layers
        self.seq_len: int = self.model.input_encoder._max_seq_len
        self.bos: int = self.model.input_encoder.bos_encoding

