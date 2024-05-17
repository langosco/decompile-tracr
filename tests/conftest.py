import os
import pytest
import numpy as np
import shutil

from decompile_tracr.dataset.make_dataset import make_dataset, to_h5
from decompile_tracr.dataset.config import DatasetConfig
from decompile_tracr.dataset.config import default_data_dir


rng = np.random.default_rng(42)


@pytest.fixture(scope="session")
def dataset_config():
    default = default_data_dir()
    config = DatasetConfig(
        ndata=50,
        program_length=5,
        data_dir=default.parent / ".tests_cache",
        name='testing_make_dataset',
    )
    return config


@pytest.fixture(scope="session")
def make_test_data(dataset_config):
    base_dir = dataset_config.paths.data_dir
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir)

    make_dataset(rng, config=dataset_config)
    to_h5(dataset_config, make_test_splits=False)

    return None

