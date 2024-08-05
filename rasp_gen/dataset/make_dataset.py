# Desc: Script to generate a dataset for training a decompiler
# Usually you would want to instead run generate.py and compile.py
# in parallel (eg by running many instances of the scripts), but
# this script is useful for generating a sample dataset using a 
# single thread.

import os
os.environ["JAX_PLATFORMS"]  = "cpu"
import argparse
import numpy as np

from rasp_gen.dataset import generate
from rasp_gen.dataset import dedupe
from rasp_gen.dataset.compile import compile_batches
from rasp_gen.dataset.compress import compress_batches
from rasp_gen.dataset import data_utils
from rasp_gen.dataset.config import DatasetConfig, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--ndata', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--make', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--add_ids', action='store_true')
    parser.add_argument('--make_test_splits', action='store_true')
    return parser.parse_args()


def make_dataset(rng: np.random.Generator, config: DatasetConfig, ndata: int):
    if config.compress is None:
        generate.generate_batches(rng, config=config, ndata=ndata)
        dedupe.dedupe(config)
        compile_batches(config)
    else:
        compress_batches(config=config)
    return 


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)

    if args.make:
        make_dataset(rng, config=config, ndata=args.ndata)
    
    if args.merge:
        data_utils.merge_h5(config)
    
    if args.add_ids:
        data_utils.add_ids(dataset=config.paths.dataset)

    if args.make_test_splits:
        data_utils.make_test_splits(dataset=config.paths.dataset)
