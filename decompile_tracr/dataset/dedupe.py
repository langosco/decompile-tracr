"""Deduplicate programs based on tokenized RASP code and save 
to disk as HDF5 file.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict
from typing import Optional
import h5py
import numpy as np

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.dataloading import load_dataset
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset.tokenize_lib import tokenize_lib
from decompile_tracr.dataset.config import DatasetConfig, load_config

logger = logger_config.setup_logger(__name__)


# Deduplicate based on tokenized rasp code.

def dedupe(config: DatasetConfig) -> list[dict]:
    """Load, dedupe, and save data."""
    programs = config.paths.programs
    logger.info("Begin loading data and deduping.")
    data = data_utils.load_batches(config.paths.programs_cache)
    reference: list = tokenize_lib(config, save=False)
    reference = [x['tokens'] for x in reference]
    if programs.exists():
        previously_deduped = load_dataset(
            programs, group=None)['tokens'].tolist()
        reference.extend(previously_deduped)
        prev_len = len(previously_deduped)
        if prev_len > 0:
            logger.info(f"(dedupe.py) Found existing data in {programs}. "
                        f"Loaded {prev_len} existing programs.")
    deduped = _dedupe(data, reference=reference)
    deduped = {k: [x[k] for x in data] for k in deduped[0].keys()}

    with h5py.File(programs, "a", libver="latest") as f:
        if "tokens" not in f:
            data_utils.init_h5(f, deduped)
        else:
            data_utils.append_h5(f, deduped)
    return deduped


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
        reference = set([tuple(x) for x in reference])
    deduped: list[dict] = []

    logger.info(f"Deduplicating {len(data):.2f} programs.")
    logger.info(f"Reference set size: {len(reference):.2f}")
    for x in data:
        tokens = tuple(x['tokens'])
        if tokens not in reference:
            reference.add(tokens)
            deduped.append(x)

    logger.info(f"Removed: {len(data) - len(deduped):.2f} programs. "
                f"({100*(len(data) - len(deduped)) / len(data):.2f}%)")
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