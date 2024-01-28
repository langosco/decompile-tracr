import argparse

from rasp_tokenizer.data_utils import load_data, process_data, split_dict_data
from rasp_tokenizer import data_utils
from rasp_tokenizer import logger_config

# load data (list of list of dicts)
# deduplicate based on tokenized rasp code
# structure as list of dicts, each datapoint is a layer
# save back to a single file


def deduplicate(data: list[list[dict]]) -> list[list[dict]]:
    all_rasp = set()
    deduped = []
    for program in data:
        r = tuple([x['rasp'] for x in program])

        if r in all_rasp:
            continue
        else:
            all_rasp.add(r)
            deduped.append(program)
    logger.info(f"Deduplicated: {len(deduped)} programs.")
    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({(len(data) - len(deduped)) / len(data)}%)")
    return deduped


def flatten(data: list[list[dict]]) -> list[dict]:
    """Flatten a list of lists of dicts into a single list of dicts.
    Keep track of program_id."""
    out = []
    for program_id, program in enumerate(data):
        for layer in program:
            layer['program_id'] = program_id
            out.append(layer)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loaddir', type=str, default=None)
    parser.add_argument('--savedir', type=str, default="train")
    args = parser.parse_args()

    logger = logger_config.setup_logger(__name__)

    data = data_utils.load_batches(loaddir=args.loaddir)
    logger.info(f"Deduplicating {len(data)} programs.")

    deduped = deduplicate(data)
    flat_and_deduped = flatten(deduped)
    data_utils.save_deduped(flat_and_deduped, savedir=args.savedir)


