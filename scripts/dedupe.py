import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict

import jax

from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config


# Deduplicate based on tokenized rasp code.

# Load data (list of dicts) from data/batches/...
# Save deduped data back to data/deduped/{name}/data.json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/batches/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    parser.add_argument('--keep_aux', action='store_true',
                        help="whether the data includes the compiled "
                        "model and rasp code. Used for testing.")
    args = parser.parse_args()


    logger = logger_config.setup_logger(__name__)


    logger.info(f"Loading data for deduplication")
    with jax.default_device(jax.devices("cpu")[0]):  # keep data on cpu
        data = data_utils.load_batches(
            loadpath=args.loadpath,
            keep_aux=args.keep_aux,
        )

    logger.info(f"Loaded data with keys {list(data[0].keys())}")

    if not args.keep_aux:
        for x in data:
            aux_present = False
            if 'model' in x:
                model_present = True
                del x['model']
            if 'rasp' in x:
                aux_present = True
                del x['rasp']

        if aux_present:
            logger.warning(
                "Data includes keys 'model' and/or 'rasp'. "
                "Did you mean to use --keep_aux?")


    logger.info(f"Deduplicating {len(data)} programs.")
    deduped = data_utils.dedupe(data)

    deduped_by_name = defaultdict(list)
    for x in deduped:
        deduped_by_name[x['name']].append(x)

    for name, data in deduped_by_name.items():
        print(f"{name}: {len(data)} programs")
        data_utils.save_deduped(
            data, 
            name=name, 
            savepath=args.savepath,
            save_aux=args.keep_aux,
        )


