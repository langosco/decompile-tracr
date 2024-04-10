import os
from pathlib import Path
import fcntl
import argparse
from tqdm import tqdm
import shutil
import psutil
import jax

from tracr.compiler import compile_rasp_to_model
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import config
from decompile_tracr.dataset.logger_config import setup_logger


logger = setup_logger(__name__)
process = psutil.Process()


def compile_all(loaddir: str, savedir: str, max_batches=None):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    logger.info(f"Compiling all RASP programs found in {loaddir}.")
    for _ in range(max_batches or 10**8):
        if compile_single_batch(loaddir, savedir) is None:
            break
        jax.clear_caches()


def compile_single_batch(
    loaddir: str, 
    savedir: str,
    max_weights_len: int = config.MAX_WEIGHTS_LENGTH
) -> list[dict]:
    """Load and compile the next batch of rasp programs."""
    data = load_next_batch(loaddir)
    if data is None:
        return None

    i = 0
    for x in tqdm(data, disable=config.global_disable_tqdm, 
                    desc="Compiling"):
        try:
            x['weights'] = get_weights(x['tokens'], max_weights_len)
        except (InvalidValueSetError, NoTokensError, DataError) as e:
            logger.warning(f"Skipping program ({e}).")

        if i % 25 == 0:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({i}/{len(data)} programs compiled).")
        
        i += 1
    
    data = [x for x in data if 'weights' in x]
    data_utils.save_batch(data, savedir)
    return data


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
        logger.info("No more files to load. "
                    f"All filenames present in {LOCKFILE}")
        return None
    else:
        return data_utils.load_json(path)


def get_weights(tokens: list[int], max_weights_len: int
) -> list[list[float]]:
    """Get flattened weights for every layer."""
    model = compile_tokens_to_model(tokens)
    n_layers = len(tokens)
    flat_weights = [
        data_utils.get_params(model.params, layername) 
        for layername in data_utils.layer_names(n_layers)
    ]
    if any(len(x) > max_weights_len for x in flat_weights):
        raise DataError(f"Too many params (> {max_weights_len})")
    elif any(x.mean() > config.MAX_WEIGHTS_LAYER_MEAN for x in flat_weights):
        raise DataError("Mean of weights too large")
    return [x.tolist() for x in flat_weights]


class DataError(Exception):
    pass


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


def compile_tokens_to_model(tokens: list[int]):
    """Compile a list of tokens into a model."""
    program = tokenizer.detokenize(tokens)
    model = compile_rasp_to_model(
        program=program,
        vocab=set(range(5)),
        max_seq_len=5,
    )
    return model


def delete_existing(savedir: str):
    logger.info(f"Deleting existing data at {savedir}.")
    shutil.rmtree(savedir, ignore_errors=True)
    os.makedirs(savedir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--loadpath', type=str, default=None, 
                        help="override default load path (data/unprocessed/...)")
    parser.add_argument('--savepath', type=str, default=None,
                        help="override default save path (data/deduped/...)")
    parser.add_argument('--single_batch_only', action='store_true',
                        help="compile a single batch (file) and then stop.")
    parser.add_argument('--delete_existing', action='store_true',
                        help="delete current data on startup.")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="maximum number of batches to compile. default: None")
    args = parser.parse_args()

    if args.loadpath is None:
        args.loadpath = config.deduped_dir
    
    if args.savepath is None:
        args.savepath = config.full_dataset_dir
    
    if args.delete_existing:
        delete_existing(args.savepath)
    
    if args.single_batch_only:
        logger.info("Compile one batch of programs.")
        compile_single_batch(
            loaddir=args.loadpath,
            savedir=args.savepath,
        )
    else:
        compile_all(
            loaddir=args.loadpath,
            savedir=args.savepath,
            max_batches=args.max_batches,
        )
