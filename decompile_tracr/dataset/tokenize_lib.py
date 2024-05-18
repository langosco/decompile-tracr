from pathlib import Path
import argparse

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset import lib
from decompile_tracr.dataset.generate import to_filter
from decompile_tracr.dataset.data_utils import save_batch


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


def tokenize_lib(config: DatasetConfig):
    logger.info("Begin tokenizing example programs.")
    data = tokenize_loop(config)
    logger.info(f"Done tokenizing {len(data)} example programs.")
    save_batch(
        data=data, 
        savedir=config.paths.programs_cache,
        overwrite=True,
        filename="lib",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenize_lib(config.paths.programs_cache)