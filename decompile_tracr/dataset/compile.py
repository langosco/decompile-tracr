import os
from pathlib import Path
from functools import partial
import fcntl

import jax

from tracr.compiler import compile_rasp_to_model

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset import utils
from decompile_tracr.dataset.logger_config import setup_logger


logger = setup_logger(__name__)


def compile(loaddir: str, savedir: str):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    while True:
        data = load_next_batch(loaddir)
        if data is None:
            return None

        for x in data:
            x['weights'] = get_weights(x['tokens'])

        utils.save_batch(data, savedir)


def get_next_filename(file_list: list[str], loaddir: Path):
    """given a list of filenames, return the next filename to load"""
    total = 0
    for entry in os.scandir(loaddir):
        if entry.is_dir():
            out = get_next_filename(file_list, entry.path)
            if out != "":
                return out

        if not entry.path.endswith(".json"):
            continue

        if entry.path not in file_list:
            total += 1
            return entry.path
    
    return ""


def load_next_batch(loaddir: str):
    """check lockfile for next batch to load"""
    LOCKFILE = loaddir / "lockfile.txt"
    with open(LOCKFILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        f.seek(0)
        file_list = [x.rstrip("\n") for x in f.readlines()]
        path = get_next_filename(file_list, loaddir)

        f.write(path + "\n")
        f.flush()

        fcntl.flock(f, fcntl.LOCK_UN)
    
    if path == "":
        return None
    else:
        return utils.load_json(path)


def get_weights(tokens: list[int]):
    """Get flattened weights for every layer."""
    program = tokenizer.detokenize(tokens)
    model = compile_rasp_to_model(
        program=program,
        vocab={1,2,3,4},
        max_seq_len=5,
    )

    n_layers = len(tokens)
    flat_weights = [utils.get_params(model.params, layername)
            for layername in utils.layer_names(n_layers)]
    flat_weights = [x.tolist() for x in flat_weights]
    return flat_weights
