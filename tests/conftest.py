import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pytest
import shutil
import numpy as np

from rasp_gen.dataset import data_utils
from rasp_gen.dataset.make_dataset import make_dataset
from rasp_gen.dataset.config import load_config

rng = np.random.default_rng()


def make_test_data():
    config = load_config("test")
    base_dir = config.paths.data_dir
    shutil.rmtree(base_dir, ignore_errors=True)
    make_dataset(rng, config=config, ndata=50)
    data_utils.merge_h5(config)
    data_utils.add_ids(config.paths.dataset)


make_test_data()