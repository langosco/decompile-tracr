import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset.config import DatasetConfig, load_config

logger = logger_config.setup_logger(__name__)


# Deduplicate based on tokenized rasp code.

def dedupe(config: DatasetConfig) -> list[dict]:
    """Load, dedupe, and save data."""
    savedir = config.paths.programs
    logger.info("Begin loading data and deduping.")
    data = data_utils.load_batches(config.paths.programs_cache)
    if savedir.exists():
        previously_deduped = data_utils.load_batches_from_subdirs(savedir)
        prev_len = len(previously_deduped)
        if prev_len > 0:
            logger.info(f"(dedupe.py) Found existing data in {savedir}. "
                        f"Loaded {prev_len} existing programs.")
    else:
        previously_deduped = None
    deduped = data_utils.dedupe(data, reference=previously_deduped)
    save_deduped(deduped, config)
    return deduped


def save_deduped(
    deduped: list[dict],
    config: DatasetConfig,
) -> None:
    """Split data by name and save to data/programs/{name}."""
    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)
    
    logger.info(f"Splitting data by name: {list(deduped_by_name.keys())}")
    for name, data in deduped_by_name.items():
        for batch in data_utils.batched(data, config.compiling_batchsize):
            data_utils.save_batch(
                batch,
                savedir=config.paths.programs / name
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--batchsize', type=int, default=180,
                        help="batch size for saving deduped data.")
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.compiling_batchsize = args.batchsize
    dedupe(config)