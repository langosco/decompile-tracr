import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict

from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches
# Save deduped data back to data/deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/batches/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    parser.add_argument('--include_model', action='store_true',
                        help="whether the data includes the compiled "
                        "model. If so, we need to use dill instead "
                        "of pickle. Used for testing.")
    args = parser.parse_args()


    logger = logger_config.setup_logger(__name__)


    logger.info(f"Loading data for deduplication")
    data = data_utils.load_batches(loadpath=args.loadpath,
                                   use_dill=args.include_model)
    logger.info(f"Loaded data with keys {data[0].keys()}")

    if "model" in data[0].keys() and not args.include_model:
        logger.warning("Data includes compiled model, but "
                       "using pickle to load files. Did you "
                       "mean to use --include_model?")


    logger.info(f"Deduplicating {len(data)} programs.")
    deduped = data_utils.dedupe(data)

    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)

    for name, programs in deduped_by_name.items():
        data_utils.save_deduped(
            deduped, 
            name=name, 
            savepath=args.savepath,
            save_model=args.include_model,
        )


