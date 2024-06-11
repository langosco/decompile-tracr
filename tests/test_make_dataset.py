import os
import pytest
import numpy as np
import shutil

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.dataloading import load_dataset
from decompile_tracr.dataset.reconstruct import get_model
from decompile_tracr.dataset.make_dataset import make_dataset, merge
from decompile_tracr.dataset.config import DatasetConfig
from decompile_tracr.dataset.config import default_data_dir


rng = np.random.default_rng()


@pytest.fixture(scope="module")
def dataset_config():
    default = default_data_dir()
    config = DatasetConfig(
        ndata=50,
        program_length=5,
        data_dir=default.parent / ".tests_cache",
        name='testing_make_dataset',
#        compress="svd",
#        n_augs=0,
    )
    return config


@pytest.fixture(scope="module")
def make_test_data(dataset_config):
    base_dir = dataset_config.paths.data_dir
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir)
    make_dataset(rng, config=dataset_config)
    merge(dataset_config, make_test_splits=False)


def test_sampling(dataset_config, make_test_data):
    #data = data_utils.load_batches(base_dir / "unprocessed")
    pass


def test_deduplication(make_test_data):
    pass


def test_tokenization(dataset_config, make_test_data):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""
    data = data_utils.load_batches_from_subdirs(dataset_config.paths.programs)
    for x in data:
        program = tokenizer.detokenize(x['tokens'])
        retokenized = tokenizer.tokenize(program)
        assert x['tokens'] == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
        )


def test_unflattening(dataset_config, make_test_data):
    data = load_dataset(dataset_config.paths.dataset)
    params = [
        data_utils.unflatten_params(w, s, d) 
        for w, s, d in zip(data['weights'], data['layer_idx'], data['d_model'])
    ]

    models = [get_model(p, h, l) 
              for p, h, l in zip(params, data['n_heads'], data['n_layers'])]
    
    for model, p in zip(models, params):
        test_input = rng.random((1, 5, 10))
        out = model.apply(p, test_input)
        assert True # TODO

    
