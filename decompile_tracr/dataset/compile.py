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
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.globals import disable_tqdm


logger = setup_logger(__name__)
process = psutil.Process()


def compile_all(config: DatasetConfig, max_batches=None):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    assert config.compress is None
    logger.info(f"Compiling RASP programs found in {config.paths.programs}.")
    for _ in range(max_batches or 10**8):
        if compile_single_batch(config) is None:
            break
        jax.clear_caches()


def compile_single_batch(config: DatasetConfig) -> list[dict]:
    """Load and compile the next batch of rasp programs."""
    assert config.compress is None
    data = load_next_batch(config.paths.programs)
    if data is None:
        return None

    i = 0
    for x in tqdm(data, disable=disable_tqdm, desc="Compiling"):
        try:
            x['weights'] = get_weights(
                x['tokens'], config.max_weights_length)
        except (InvalidValueSetError, NoTokensError, DataError) as e:
            logger.warning(f"Skipping program ({e}).")

        if i % 25 == 0:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({i}/{len(data)} programs compiled).")
        i += 1
    
    data = [x for x in data if 'weights' in x]
    data_utils.save_batch(data, config.paths.compiled_cache)
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
        logger.info(f"Loading next batch: {path}.")
        return data_utils.load_json(path)


def get_weights(tokens: list[int], max_weights_len: int
                ) -> list[list[float]]:
    """Get flattened weights for every layer."""
    model = compile_tokens_to_model(tokens)
    n_layers = 2 * max(int(k[18]) + 1 for k in model.params.keys() 
                       if "layer" in k)
    flat_weights = [
        data_utils.get_params(model.params, layername) 
        for layername in data_utils.layer_names(n_layers)
    ]
    if sum(len(x) for x in flat_weights) > max_weights_len:
        raise DataError(f"Too many params (> {max_weights_len})")
    return [x.tolist() for x in flat_weights]


class DataError(Exception):
    pass


def get_next_filename(file_list: list[str], loaddir: Path):
    """Return the next filename of a json file in loaddir. 
    Recursively search subdirectories.
    Args:
        file_list: list of filenames already loaded (to avoid).
        loaddir: directory to search.
    """
    for entry in os.scandir(loaddir):
        if entry.path.endswith(".json") and entry.path not in file_list:
            return entry.path

        if entry.is_dir():
            out = get_next_filename(file_list, entry.path)
            if out != "":
                return out
    
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
    parser.add_argument('--delete_existing', action='store_true',
                        help="delete current data on startup.")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="maximum number of batches to compile.")
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()
    
    if args.delete_existing:
        delete_existing(args.savepath)
    
    config = load_config(args.config)
    compile_all(
        config=config,
        max_batches=args.max_batches,
    )
