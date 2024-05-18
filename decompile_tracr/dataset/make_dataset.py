# Desc: Script to generate a dataset for training a decompiler
# Usually you would want to instead run generate.py and compile.py
# in parallel (eg by running many instances of the scripts), but
# this script is useful for generating a sample dataset using a 
# single thread.

import argparse
import numpy as np
import jax

from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile, compile_and_compress
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--name', type=str, default="train")
    parser.add_argument('--program_length', type=int, default=7,
                        help='program length (nr of sops)')
    parser.add_argument('--ndata', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--make_test_splits', action='store_true')
    parser.add_argument('--only_to_h5', action='store_true')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    return parser.parse_args()


def make_dataset(rng: np.random.Generator, config: DatasetConfig):
    generate.generate(rng, config=config)
#    tokenize_lib.tokenize_lib(config)
    dedupe.dedupe(config)
    if config.compress:
        key = jax.random.key(rng.integers(0, 2**32))
        compile_and_compress.compile_all(key, config=config)
    else:
        compile.compile_all(config)


def to_h5(config: DatasetConfig, make_test_splits: bool = False):
    data_utils.load_json_and_save_to_hdf5(config)
    if make_test_splits:
        data_utils.make_test_splits(dataset=config.paths.dataset),


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    config = load_config(args.config)
    config.ndata = args.ndata
    config.program_length = args.program_length
    config.name = args.name
    config.data_dir = args.data_dir

    if not args.only_to_h5:
        make_dataset(rng, config)
    to_h5(config, make_test_splits=args.make_test_splits)
