import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = 0.1
import argparse
import gc
import psutil

import jax
import numpy as np
import h5py

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.dataloading import DataLoader
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.compress import compress
from decompile_tracr.dataset.reconstruct import ModelFromParams


logger = setup_logger(__name__)
process = psutil.Process()


# load from default dataset
# reconstruct params
# if no compression: just multiply by a random orthogonal matrix
# if compression: reconstruct model too, and train encoder/decoder,
# then compress params


def compress_batches(config: DatasetConfig) -> None:
    # TODO: enable parallelism
    logger.info(f"Compressing RASP programs found in "
                f"{config.source_paths.dataset} and saving "
                f"to {config.paths.compiled_cache}.")
    
    with h5py.File(config.source_paths.dataset, "r") as f:
        groups = list(f.keys())

    for group in groups:
        dataloader = DataLoader(
            loadfile=config.source_paths.dataset,
            group=group,
            batch_size=config.compiling_batchsize,
        )

        for batch in dataloader:
            batch = compress_batch(batch, config=config)
            data_utils.save_h5(batch, config.paths.compiled_cache)
            del batch
            jax.clear_caches()
            gc.collect()


def compress_batch(batch: dict, config: DatasetConfig) -> dict:
    assert config.compress is not None
    if 'batch_id' in batch:
        del batch['batch_id']

    compressed = [
        compress_datapoint({k: v[i] for k, v in batch.items()}, config)
        for i in range(len(batch['weights']))
    ]

    return [d for d in compressed if d is not None]


def compress_datapoint(x: dict, config: DatasetConfig):
    try:
        return unsafe_compress_datapoint(x, config)
    except data_utils.DataError as e:
        logger.info(f"Failed to compress datapoint. {e}")
        return None


def unsafe_compress_datapoint(x: dict, config: DatasetConfig) -> dict:
    params = data_utils.unflatten_params(
        x['weights'], sizes=x['layer_idx'], d_model=x['d_model'])
    model = ModelFromParams(params, num_heads=x['n_heads'])
    h = int(model.d_model // 1.1)
    wenc, wdec, aux = compress.train_svd(model=model, hidden_size=h)
    params_compressed = compress.update_params(params, wenc, wdec)
    params_compressed, idx = data_utils.flatten_params(
        params_compressed, config=config)
    
    out = x.copy()
    out['weights'] = params_compressed
    out['layer_idx'] = idx
    out['d_model'] = h
    return out

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help="maximum number of batches to compile.")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.default_rng(None).integers(0, 2**32)
    
    if args.device == "cpu":
        device = jax.devices("cpu")[0]
    elif args.device is None:
        device = None
    else:
        device = jax.devices()[int(args.device)]
    
    with jax.default_device(device):
        config = load_config(args.config)
        compress_batches(config=config)
