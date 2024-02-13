import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict

import jax

from decompile_tracr.dataset import utils
from decompile_tracr.dataset import logger_config

logger = logger_config.setup_logger(__name__)

# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches/...
# Save deduped data back to data/deduped/{name}/data.json

def to_tuples(data: list[list]):
    return [tuple(tuple(x) for x in d) for d in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/unprocessed/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    args = parser.parse_args()


    logger.info(f"Loading data for deduplication")
    with jax.default_device(jax.devices("cpu")[0]):  # keep data on cpu
        data = utils.load_batches(
            loadpath=args.loadpath,
        )


    logger.info(f"Deduplicating {len(data)} programs.")
    deduped = utils.dedupe(data)
    logger.info(f"Found {len(deduped)} unique programs.")
    logger.info(f"Percent duplicates: {100 * (1 - len(deduped) / len(data)):.2f}%")


    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)


    for name, data in deduped_by_name.items():
        logger.info(f"{name}: saving {len(data)} programs")
        utils.save_deduped(
            data, 
            name=name, 
            savepath=args.savepath,
        )


