import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict
from typing import Optional

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset.tokenize_lib import tokenize_lib
from decompile_tracr.dataset.config import DatasetConfig, load_config

logger = logger_config.setup_logger(__name__)


# Deduplicate based on tokenized rasp code.

def dedupe(config: DatasetConfig) -> list[dict]:
    """Load, dedupe, and save data."""
    savedir = config.paths.programs
    logger.info("Begin loading data and deduping.")
    data = data_utils.load_batches(config.paths.programs_cache)
    reference = tokenize_lib(config, save=False)
    if savedir.exists():
        previously_deduped = data_utils.load_batches_from_subdirs(savedir)
        reference.extend(previously_deduped)
        prev_len = len(previously_deduped)
        if prev_len > 0:
            logger.info(f"(dedupe.py) Found existing data in {savedir}. "
                        f"Loaded {prev_len} existing programs.")
    deduped = _dedupe(data, reference=reference)
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
            data_utils.save_json(
                batch,
                savedir=config.paths.programs / name
            )


def _dedupe(data: list[dict], reference: Optional[list[dict]] = None,
            ) -> list[dict]:
    """Deduplicate programs by RASP string.
    Assume data is a list of dicts that include the 
    key "tokens", as returned by load_batches().

    Args:
    - data: list of dicts with keys "tokens".
    - reference: list of dicts with keys "tokens". If provided,
    treat examples in data that match elements of reference as duplicates.
    """
    if reference is None:
        reference: set[list[int]] = set()
    else:
        reference = set([tuple(x['tokens']) for x in reference])
    deduped: list[dict] = []

    logger.info(f"Deduplicating {len(data)} programs.")
    logger.info(f"Reference set size: {len(reference)}")
    for x in data:
        tokens = tuple(x['tokens'])
        if tokens not in reference:
            reference.add(tokens)
            deduped.append(x)

    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({100*(len(data) - len(deduped)) / len(data)}%)")
    logger.info(f"Remaining new datapoints: {len(deduped)}")

    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--batchsize', type=int, default=None,
                        help="batch size for saving deduped data.")
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batchsize is not None:
        config.compiling_batchsize = args.batchsize
    dedupe(config)