import os
import argparse

from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config
from rasp_tokenizer import paths


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches/{dir} for all {dir}.
# For each {dir}, save deduped data back to a single file 
# at data/deduped/{dir}/data.pkl.

# If args.dirname is specified, load from data/batches/{args.datadir}.
# Then save deduped data to data/deduped/{args.dirname}/data.pkl.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--dirname', type=str, default=None)
    args = parser.parse_args()


    logger = logger_config.setup_logger(__name__)

    if args.dirname is None:
        dirnames = os.scandir(paths.data_dir / "batches")
        dirnames = [entry.name for entry in dirnames if entry.is_dir()]
    else:
        dirnames = [args.dirname]


    for d in dirnames:
        data = data_utils.load_batches(loaddir=d)
        logger.info(f"Deduplicating {len(data)} programs.")

        deduped = data_utils.dedupe(data)
        data_utils.save_deduped(deduped, savedir=d)


