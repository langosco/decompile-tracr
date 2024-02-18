import os
from pathlib import Path
import fcntl
import argparse

from tracr.compiler import compile_rasp_to_model
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import config
from decompile_tracr.dataset.logger_config import setup_logger


logger = setup_logger(__name__)


def compile(loaddir: str, savedir: str):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    logger.info(f"Compiling data from {loaddir}.")
    while True:
        data = load_next_batch(loaddir)
        if data is None:
            return None

        for x in data:
            try:
                x['weights'] = get_weights(x['tokens'])
            except (InvalidValueSetError, NoTokensError) as e:
                logger.warning(f"Skipping program ({e}).")
                continue

        data_utils.save_batch(data, savedir)


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
    LOCKFILE = loaddir / "files_already_loaded.lock"
    with open(LOCKFILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        f.seek(0)
        file_list = [x.rstrip("\n") for x in f.readlines()]
        path = get_next_filename(file_list, loaddir)

        f.write(path + "\n")
        f.flush()

        fcntl.flock(f, fcntl.LOCK_UN)
    
    if path == "":
        logger.info(f"No more files to load. "
                    "All filenames present in {LOCKFILE}")
        return None
    else:
        return data_utils.load_json(path)


def compile_tokens_to_model(tokens: list[int]):
    """Compile a list of tokens into a model."""
    program = tokenizer.detokenize(tokens)
    model = compile_rasp_to_model(
        program=program,
        vocab=set(range(5)),
        max_seq_len=5,
    )
    return model


def get_weights(tokens: list[int]):
    """Get flattened weights for every layer."""
    model = compile_tokens_to_model(tokens)
    n_layers = len(tokens)
    flat_weights = [
        data_utils.get_params(model.params, layername) 
        for layername in data_utils.layer_names(n_layers)
    ]
    return [x.tolist() for x in flat_weights]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/unprocessed/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    args = parser.parse_args()

    if args.loadpath is None:
        args.loadpath = config.deduped_dir
    
    if args.savepath is None:
        args.savepath = config.full_dataset_dir

    compile(
        loaddir=args.loadpath,
        savedir=args.savepath,
    )
