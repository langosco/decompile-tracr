import os
from pathlib import Path
import argparse
import shutil
import psutil
import jax
import haiku as hk
import numpy as np

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
    return get_model(params, x['n_heads'], x['n_layers']), params


def get_model(params: dict, num_heads: int, num_layers: int):
    key_size = (params['transformer/layer_0/attn/key']['b'].shape[0] 
                // num_heads)
    mlp_hidden_size = params['transformer/layer_0/mlp/linear_1']['b'].shape[0]

    transformer_config = TransformerConfig(
        num_heads=num_heads,
        num_layers=num_layers,
        key_size=key_size,
        mlp_hidden_size=mlp_hidden_size,
        dropout_rate=0.0,
        activation_function=jax.nn.relu,
        layer_norm=False,
        causal=False,
    )

    @hk.without_apply_rng
    @hk.transform
    def model(x):  # x: (batch, seq_len, d_model)
        mask = np.ones(x.shape[:2])
        return Transformer(config=transformer_config)(
            x, mask=mask, use_dropout=False)

    return model
