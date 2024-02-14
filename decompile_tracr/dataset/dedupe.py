import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import shutil
import argparse
from collections import defaultdict
from pathlib import Path

import jax

from decompile_tracr.dataset import utils
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset import config

logger = logger_config.setup_logger(__name__)
BATCHSIZE = 1024


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches/...
# Save deduped data back to data/deduped/{name}/data.json


def save_deduped(deduped: list[dict], savedir: str = config.deduped_dir):
    """Split data by name and save to data/deduped/{name}."""
    # delete data/deduped if it exists
    savedir = Path(savedir)
    shutil.rmtree(savedir, ignore_errors=True)
    os.makedirs(savedir)

    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)
    
    logger.info(f"Splitting data by name: {list(deduped_by_name.keys())}")

    for name, data in deduped_by_name.items():
        for batch in utils.batched(data, BATCHSIZE):
            utils.save_batch(
                batch,
                savedir=savedir / name
            )


def dedupe(loaddir: str, savedir: str):
    """Load, dedupe, and save data."""
    data = utils.load_batches(loaddir)
    deduped = utils.dedupe(data)
    save_deduped(deduped, savedir)
    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/unprocessed/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    args = parser.parse_args()

    if args.loadpath is None:
        args.loadpath = config.unprocessed_dir
    
    if args.savepath is None:
        args.savepath = config.deduped_dir

    dedupe(args.loadpath, args.savepath)