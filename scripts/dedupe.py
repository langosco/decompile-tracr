import argparse
from collections import defaultdict

from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches
# Save deduped data back to data/deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loaddir', type=str, default=None)
    args = parser.parse_args()


    logger = logger_config.setup_logger(__name__)


    data = data_utils.load_batches(loaddir=args.loaddir)

    logger.info(f"Deduplicating {len(data)} programs.")
    deduped = data_utils.dedupe(data)

    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)

    for name, programs in deduped_by_name.items():
        data_utils.save_deduped(deduped, savedir=name)


