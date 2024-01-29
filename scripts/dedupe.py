import argparse

from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config


# load data (list of dicts)
# deduplicate based on tokenized rasp code
# structure as list of dicts, each datapoint is a layer
# save back to a single file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loaddir', type=str, default=None)
    parser.add_argument('--savedir', type=str, default="train")
    args = parser.parse_args()


    logger = logger_config.setup_logger(__name__)


    data = data_utils.load_batches(loaddir=args.loaddir)
    logger.info(f"Deduplicating {len(data)} programs.")

    deduped = data_utils.dedupe(data)
    data_utils.save_deduped(deduped, savedir=args.savedir)


