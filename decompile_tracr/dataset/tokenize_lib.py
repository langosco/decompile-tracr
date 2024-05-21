from pathlib import Path
import argparse
import h5py

import numpy as np
import jax

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset import lib
from decompile_tracr.dataset.generate import to_filter
from decompile_tracr.dataset.data_utils import save_batch
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import compile, compile_and_compress


logger = setup_logger(__name__)


def tokenize_loop(config: DatasetConfig):
    data = []
    for program_id, program in enumerate(lib.examples):
        tokens = tokenizer.tokenize(program)
        if to_filter(tokens, config=config):
            logger.warning(f"Program {program_id} is too long. (Not skipping).")

        data.append({
            "name": "lib",
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })
    return data


def tokenize_lib(config: DatasetConfig, save=True):
    logger.info("Begin tokenizing example programs.")
    data = tokenize_loop(config)
    logger.info(f"Done tokenizing {len(data)} example programs.")
    if save:
        save_batch(
            data=data, 
            savedir=config.paths.programs_cache,
            overwrite=True,
            filename="lib",
        )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--compile', action='store_true',
                        help="Also compile the examples.")
    parser.add_argument('--dont_save_tokenized', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    examples = tokenize_lib(config.paths.programs_cache, save=not args.dont_save_tokenized)
    rng = np.random.default_rng()
    key = jax.random.key(rng.integers(0, 2**32))

    if args.compile:
        for x in examples:
            if not config.compress:
                x['weights'] = compile.get_weights(
                    x['tokens'], config.max_weights_length)
            else:
                key, subkey = jax.random.split(key)
                compressed = compile_and_compress.process_tokens(
                    subkey, x['tokens'], config)
                to_save = compile_and_compress.to_datapoints(
                    compressed, x)

        data = data_utils.dataset_to_arrays(data=examples, config=config)
        with h5py.File(config.paths.dataset, 'a') as f:
            f.create_group("lib")
            data_utils.init_h5(f["lib"], data, maxn=100)


        if config.compress is not None:
            key = jax.random.key(rng.integers(0, 2**32))
            compile_and_compress.compile_all(key, config=config)
        else:
            compile.compile_all(config)