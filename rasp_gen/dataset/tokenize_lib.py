from pathlib import Path
import argparse
import h5py

import numpy as np
import jax

from rasp_gen.tokenize import tokenizer
from rasp_gen.dataset.logger_config import setup_logger
from rasp_gen.dataset.config import DatasetConfig, load_config
from rasp_gen.dataset import lib
from rasp_gen.dataset.generate import to_filter
from rasp_gen.dataset.data_utils import save_json
from rasp_gen.dataset import data_utils
from rasp_gen.dataset.compile import compile_batch


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
    assert config.compress is None
    logger.info("Begin tokenizing example programs.")
    data = tokenize_loop(config)
    logger.info(f"Done tokenizing {len(data)} example programs.")
    if save:
        save_json(
            rng=None,
            data=data, 
            savedir=config.paths.programs_cache,
            overwrite=True,
            filename="lib",
        )
    return data


def compile_and_save_to_h5(examples: list[dict], config: DatasetConfig):
    logger.info("Compiling examples.")
    data = compile_batch(examples, config=config)
    logger.info("Saving to h5.")
    with h5py.File(config.paths.dataset, 'a', libver="latest") as f:
        f.create_group("lib")
        data_utils.init_h5(f["lib"], data, maxn=100)
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--compile', action='store_true',
                        help="Also compile the examples.")
    parser.add_argument('--dont_save_tokenized', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    with jax.default_device(jax.devices('cpu')[0]):
        examples = tokenize_lib(config, save=not args.dont_save_tokenized)
        rng = np.random.default_rng()
        key = jax.random.key(rng.integers(0, 2**32))

        if args.compile:
            compile_and_save_to_h5(examples, config)