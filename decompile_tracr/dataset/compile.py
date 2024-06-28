import os
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
import argparse
import shutil
import psutil
import gc
import jax
import numpy as np
import h5py

from tracr.compiler import compile_rasp_to_model
from tracr.rasp import rasp
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.tokenize import vocab
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.compress.utils import AssembledModelInfo
from decompile_tracr.dataset import Signals


logger = setup_logger(__name__)
process = psutil.Process()


def compile_batches(
    config: DatasetConfig, 
    max_batches: int = 10**8,
) -> None:
    logger.info(f"Compiling RASP programs found in "
                f"{config.paths.programs} and saving "
                f"to {config.paths.compiled_cache}.")

    for _ in range(max_batches):
        data = load_batch(config=config)
        if data is None:
            logger.info(f"No more data to load in {config.paths.programs}")
            return None

        data = compile_batch(data, config=config)

        if not Signals.n_sigterms >= 2:
            data_utils.save_h5(data, config.paths.compiled_cache)

        if Signals.sigterm:
            break

        del data
        jax.clear_caches()
        gc.collect()


def load_batch(config: DatasetConfig):
    assert config.paths.programs.exists()
    n = data_utils.ndata(config.paths.programs)
    start, end = data_utils.track_idx_between_processes(
        config, name="compile_idx", maximum=n)
    if start == end:
        return None
    with h5py.File(config.paths.programs, 'r') as f:
        data_dict = {k: v[start:end] for k, v in f.items()}
        n = len(data_dict['tokens'])
    data = [{k: v[i] for k, v in data_dict.items()} for i in range(n)]
    return data


def compile_batch(data: list[dict], config: DatasetConfig):
    assert config.compress is None
    data = [compile_datapoint(d, config=config) for d in data]
    data = [d for d in data if d is not None]
    return data


def unsafe_compile_datapoint(x: dict, config: DatasetConfig):
    prog = tokenizer.detokenize(x['tokens'])
    model = compile_(prog)
    flat, idx = data_utils.flatten_params(model.params, config)
    info = AssembledModelInfo(model=model)
    x['weights'], x['layer_idx'] = flat, idx
    x['d_model'] = info.d_model
    x['n_heads'] = info.num_heads
    x['categorical_output'] = info.use_unembed_argmax
    x['n_layers'] = info.num_layers
    return x


def compile_datapoint(x: dict, config: DatasetConfig):
    try:
        return unsafe_compile_datapoint(x, config)
    except (NoTokensError, InvalidValueSetError, 
            data_utils.DataError) as e:
        logger.warning(f"Failed to compile datapoint. {e}")
        return None


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
    parser.add_argument('--max_batches', type=int, default=10**8,
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
