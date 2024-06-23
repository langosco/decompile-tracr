import os
import pytest
import numpy as np
import shutil
import chex

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.dataloading import load_dataset
from decompile_tracr.dataset.reconstruct import ModelFromParams
from decompile_tracr.dataset.make_dataset import make_dataset, merge
from decompile_tracr.dataset.config import DatasetConfig
from decompile_tracr.dataset.config import default_base_data_dir
from decompile_tracr.dataset.compile import compile_
from decompile_tracr.compress.metrics import Embed, Unembed
from decompile_tracr.compress.utils import AssembledModelInfo


rng = np.random.default_rng()


@pytest.fixture(scope="module")
def dataset_config():
    default = default_base_data_dir()
    config = DatasetConfig(
        base_data_dir=default / ".tests_cache",
        ndata=50,
        program_length=5,
#        compress="svd",
#        n_augs=0,
    )
    return config


@pytest.fixture(scope="module")
def make_test_data(dataset_config):
    base_dir = dataset_config.paths.data_dir
    shutil.rmtree(base_dir, ignore_errors=True)
    make_dataset(rng, config=dataset_config)
    merge(dataset_config)
    data_utils.add_ids(dataset_config.paths.dataset)


def test_sampling(dataset_config, make_test_data):
    #data = data_utils.load_batches(base_dir / "unprocessed")
    pass


def test_deduplication(make_test_data):
    pass


def test_tokenization(dataset_config, make_test_data):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""
    data = data_utils.load_batches(dataset_config.paths.programs)
    for x in data:
        program = tokenizer.detokenize(x['tokens'])
        retokenized = tokenizer.tokenize(program)
        assert x['tokens'] == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
        )


def test_unflattening(dataset_config, make_test_data):
    data = load_dataset(dataset_config.paths.dataset)
    for i, w in enumerate(data['weights']):
        x = {k: v[i] for k, v in data.items()}
        params = data_utils.unflatten_params(
            w, sizes=x['layer_idx'], d_model=x['d_model'])

        # recompile and compare
        m = compile_(tokenizer.detokenize(data['tokens'][i]))
        info = AssembledModelInfo(model=m)
        chex.assert_trees_all_equal(params, m.params)
        assert info.num_heads == x['n_heads']
        assert info.num_layers == x['n_layers']
        assert info.d_model == x['d_model']

        # run forward pass through reconstructed and recompiled models
        # recompiled model:
        input_ = ["compiler_bos", 4, 2, 3]
        out_compiled = m.apply(input_).decoded

        # reconstructed model:
        embed, unembed = Embed(assembled_model=m), Unembed(assembled_model=m)
        model = ModelFromParams(params, num_heads=x['n_heads'])
        input_tokens = np.array([m.input_encoder.encode(input_)])
        out_reconstr = model.from_tokens.apply(params, input_tokens)
        unembedded = unembed(out_reconstr.output)
        decoded = m.output_encoder.decode(np.squeeze(unembedded).tolist())

        assert out_compiled[1:] == decoded[1:], (
            f"Reconstructed model {i} output does not match original output. "
            f"Expected: {out_compiled[1:]}, got: {decoded[1:]}"
        )


def test_datapoint_attributes(dataset_config):
    KEYS = set([
        'categorical_output', 'd_model', 'layer_idx', 'n_heads', 
        'n_layers', 'n_sops', 'tokens', 'weights', 'ids',
    ])
    data = load_dataset(dataset_config.paths.dataset, end=0)
    actual_keys = set(data.keys())
    assert actual_keys >= KEYS, (
        f"Missing keys {KEYS - actual_keys} in dataset.")    