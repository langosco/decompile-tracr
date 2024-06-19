import os
from pathlib import Path
import argparse
import shutil
import psutil
import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp
from jaxtyping import ArrayLike

from tracr.compiler import compile_rasp_to_model
from tracr.rasp import rasp
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.transformer.model import Transformer, TransformerConfig

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger


def reconstruct_model_from_datapoint(x: dict):
    params = data_utils.unflatten_params(
        flat=x['weights'], sizes=x['data_idx'], d_model=x['d_model'])
    return ModelFromParams(params, x['n_heads'])


class ModelFromParams:
    def __init__(self, params: dict, num_heads: int):
        self.params = params
        num_layers = (len(params) - 2) / 6; assert num_layers.is_integer()
        num_layers = int(num_layers)
        key_size = (params['transformer/layer_0/attn/key']['b'].shape[0] 
                    // num_heads)
        mlp_hidden_size = \
            params['transformer/layer_0/mlp/linear_1']['b'].shape[0]

        self.model_config = TransformerConfig(
            num_heads=num_heads,
            num_layers=num_layers,
            key_size=key_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout_rate=0.0,
            activation_function=jax.nn.relu,
            layer_norm=False,
            causal=False,
        )

        def embed(tokens: jax.Array) -> jax.Array:
            token_embed = hk.Embed(
                embedding_matrix=params['token_embed']['embeddings'], 
                name="token_embed",
            )
            position_embed = hk.Embed(
                embedding_matrix=params['pos_embed']['embeddings'], 
                name="pos_embed",
            )
            token_embeddings = token_embed(tokens)
            positional_embeddings = position_embed(
                jnp.indices(tokens.shape)[-1])
            return token_embeddings + positional_embeddings  # [B, T, D]

        def from_embeddings(x: ArrayLike):
            mask = np.ones(x.shape[:2])
            return Transformer(config=self.model_config)(
                x, mask=mask, use_dropout=False)

        def from_tokens(tokens: ArrayLike):
            x = embed(tokens)
            mask = np.ones(x.shape[:2])
            return Transformer(config=self.model_config)(
                x, mask=mask, use_dropout=False)

        self.from_embeddings = hk.without_apply_rng(
            hk.transform(from_embeddings))
        self.from_tokens = hk.without_apply_rng(
            hk.transform(from_tokens))
        self.embed = hk.without_apply_rng(hk.transform(embed))

        self.seq_len = self.params['pos_embed']['embeddings'].shape[0] - 1
        self.d_model = self.params['pos_embed']['embeddings'].shape[1]

