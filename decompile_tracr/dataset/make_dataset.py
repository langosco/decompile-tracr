# Desc: Script to generate a dataset for training a decompiler
# Usually you would want to instead run generate.py and compile.py
# in parallel (eg by running many instances of the scripts), but
# this script is useful for generating a sample dataset using a 
# single thread.

import os
os.environ["JAX_PLATFORMS"]  = "cpu"
import argparse
import numpy as np

from decompile_tracr.dataset import generate
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset.compile import compile_batches
from decompile_tracr.dataset.compress import compress_batches
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--ndata', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--make_test_splits', action='store_true')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--only_merge', action='store_true')
    return parser.parse_args()


def make_dataset(rng: np.random.Generator, config: DatasetConfig):
    if config.compress is None:
        generate.generate_batches(rng, config=config)
        dedupe.dedupe(config)
        compile_batches(config)
    else:
        compress_batches(config=config)
    return 


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    config = load_config(args.config)
    if args.ndata is not None:
        config.ndata = args.ndata
    
    if not args.only_merge:
        make_dataset(rng, config)

    data_utils.merge_h5(config)
    data_utils.add_ids(dataset=config.paths.dataset)

    if args.make_test_splits:
        data_utils.make_test_splits(dataset=config.paths.dataset)
