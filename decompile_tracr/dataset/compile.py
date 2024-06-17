import os
from pathlib import Path
import argparse
import shutil
import psutil
import gc
import jax

from tracr.compiler import compile_rasp_to_model
from tracr.rasp import rasp
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.compress.utils import AssembledModelInfo


logger = setup_logger(__name__)
process = psutil.Process()


def compile_batches(
    config: DatasetConfig, 
    max_batches: int = 10**8,
    filename: str = None,
) -> None:
    logger.info(f"Compiling RASP programs found in "
                f"{config.paths.programs} and saving to {filename}.")

    for _ in range(max_batches):
        data = load_next_batch(config.paths.programs)
        if data is None:
            break

        data = compile_batch(data, config=config)
        data_utils.save_h5(data, config.paths.compiled_cache)
        del data
        jax.clear_caches()
        gc.collect()


def compile_batch(data: list[dict], config: DatasetConfig):
    assert config.compress is None
    data = [
        compile_datapoint(d, config.max_weights_length) for d in data]
    data = [d for d in data if d is not None]
    data = [data_utils.datapoint_to_arrays(d, config) for d in data]
    return data


def unsafe_compile_datapoint(x: dict, max_len: int):
    prog = tokenizer.detokenize(x['tokens'])
    model = compile_(prog)
    info = AssembledModelInfo(model=model)
    flat, idx = data_utils.flatten_params(model.params, max_len)
    x['weights'], x['layer_idx'] = flat, idx
    x['d_model'] = info.d_model
    x['n_heads'] = info.num_heads
    x['categorical_output'] = info.use_unembed_argmax
    x['n_layers'] = info.num_layers
    return x


def compile_datapoint(x: dict, max_len: int):
    try:
        return unsafe_compile_datapoint(x, max_len)
    except (NoTokensError, InvalidValueSetError, 
            data_utils.DataError) as e:
        logger.info(f"Failed to compile datapoint. {e}")
        return None


def load_next_batch(loaddir: str):
    """Keep  track of files already loaded.
    Return next batch of data.
    """
    history = loaddir / "files_already_loaded.txt"
    with data_utils.Lock(loaddir / "history.lock"):
        with open(history, "a+") as f:
            f.seek(0)
            file_list = [x.rstrip("\n") for x in f.readlines()]
            path = get_next_filename(file_list, loaddir)

            f.write(path + "\n")
            f.flush()

    if path == "":
        logger.info("No more files to load. "
                    f"All filenames present in {history}")
        return None
    else:
        logger.info(f"Loading next batch: {path}.")
        return data_utils.load_json(path)


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


def compile_(program: rasp.SOp):
    return compile_rasp_to_model(
        program,
        vocab=set(range(5)),
        max_seq_len=5,
    )


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
        logger.info(f"Deleting existing data at {args.savepath}.")
        shutil.rmtree(args.savepath, ignore_errors=True)
        os.makedirs(args.savepath)
    
    config = load_config(args.config)
    compile_batches(
        config=config,
        max_batches=args.max_batches,
    )
