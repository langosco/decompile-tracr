# Desc: Script to generate a dataset for training a decompiler
# Usually you would want to instead run generate.py and compile.py
# in parallel (eg by running many instances of the scripts), but
# this script is useful for generating a sample dataset using a 
# single thread.

import numpy as np
import argparse

from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--name', type=str, default="train")
    parser.add_argument('--program_length', type=int, default=7,
                        help='program length (nr of sops)')
    parser.add_argument('--ndata', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    return parser.parse_args()


def make_dataset(rng: np.random.Generator, config: DatasetConfig):
    generate.generate(rng, config=config)
    tokenize_lib.tokenize_lib(config)
    dedupe.dedupe(config)
    compile.compile_all(config)
    data_utils.load_json_and_save_to_hdf5(config)
    data_utils.make_test_splits(dataset=config.paths.dataset),


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = DatasetConfig(
        ndata=args.ndata,
        program_length=args.program_length,
        data_dir=args.data_dir,
        name=args.name,
    )
    make_dataset(rng, config)
