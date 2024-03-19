import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

import jax

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset import config

logger = logger_config.setup_logger(__name__)
DEFAULT_BATCHSIZE = 1024


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches/...
# Save deduped data back to data/deduped/{name}/data.json
# TODO: allow for deduping new generated data without
# forcing a re-compile of all old data.

def save_deduped(
    deduped: list[dict],
    savedir: str | Path = config.deduped_dir,
    batchsize: int = DEFAULT_BATCHSIZE,
) -> None:
    """Split data by name and save to data/deduped/{name}."""
    savedir = Path(savedir)
    if savedir.exists():
        logger.info(f"Deleting existing data at {savedir}.")
        shutil.rmtree(savedir, ignore_errors=True)
        os.makedirs(savedir)

    logger.info("Begin loading data and deduping.")
    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)
    
    logger.info(f"Splitting data by name: {list(deduped_by_name.keys())}")

    for name, data in deduped_by_name.items():
        for batch in data_utils.batched(data, batchsize):
            data_utils.save_batch(
                batch,
                savedir=savedir / name
            )


def dedupe(loaddir: str | Path, savedir: str | Path,
           batchsize: int) -> list[dict]:
    """Load, dedupe, and save data."""
    data = data_utils.load_batches(loaddir)
    deduped = data_utils.dedupe(data)
    save_deduped(deduped, savedir, batchsize)
    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=config.unprocessed_dir, 
                        help="override default load path (data/unprocessed/...)")
    parser.add_argument('--savepath', type=str, default=config.deduped_dir,
                        help="override default save path (data/deduped/...)")
    parser.add_argument('--batchsize', type=int, default=DEFAULT_BATCHSIZE,
                        help="batch size for saving deduped data.")
    args = parser.parse_args()

    dedupe(args.loadpath, args.savepath, batchsize=args.batchsize)