import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import gc
import psutil
from pathlib import Path

import jax
from jax.random import PRNGKey
import numpy as np
import h5py

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import Signals
from decompile_tracr.dataset.dataloading import load_dataset
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
    logger.info(f"Compressing RASP programs found in "
                f"{config.source_paths.dataset} and saving "
                f"to {config.paths.compiled_cache}.")
    end = 0
    while end <= ndata(config.source_paths.dataset):
        start, end = data_utils.track_idx_between_processes(
            config, name="compress_idx")
        done = load_and_compress_batch(config, start, end)
        if done or Signals.sigterm:
            break


def load_and_compress_batch(config: DatasetConfig, start: int, end: int
                            ) -> None:
    done = False
    with h5py.File(config.source_paths.dataset, "r", libver="latest") as f:
        groups = set.intersection(set(f.keys()), {"train", "val", "test"})
        groups = [g for g in groups if len(f[g]) >= start]
        if len(groups) == 0:
            done = True
            return done

    key = jax.random.key(0)
    for group in groups:
        batch = load_dataset(
            loadfile=config.source_paths.dataset,
            group=group,
            start=start,
            end=end,
        )
        key, subkey = jax.random.split(key)
        batch = compress_batch(subkey, batch, config=config)
        if not Signals.n_sigterms >= 2:
            data_utils.save_h5(batch, config.paths.compiled_cache)
        del batch
        jax.clear_caches()
        gc.collect()
    
    return done


def compress_batch(key: PRNGKey, batch: dict, config: DatasetConfig) -> dict:
    assert config.compress is not None
    if 'batch_id' in batch:
        del batch['batch_id']

    keys = jax.random.split(key, len(batch['weights']))
    compressed = [
        compress_datapoint(k, {k: v[i] for k, v in batch.items()}, config)
        for k, i in zip(keys, range(len(batch['weights'])))
    ]

    return [d for d in compressed if d is not None]


def compress_datapoint(key: PRNGKey, x: dict, config: DatasetConfig):
    try:
        return unsafe_compress_datapoint(key=key, x=x, config=config)
    except data_utils.DataError as e:
        logger.info(f"Failed to compress datapoint. {e}")
        return None


def unsafe_compress_datapoint(key: PRNGKey, x: dict, config: DatasetConfig
                              ) -> dict:
    params = data_utils.unflatten_params(
        x['weights'], sizes=x['layer_idx'], d_model=x['d_model'])
    model = ModelFromParams(params, num_heads=x['n_heads'])
    if config.compress == "svd":
        h = int(model.d_model // 1.1)
        wenc, wdec, aux = compress.train_svd(model=model, hidden_size=h)
    elif config.compress == "autoencoder":
        raise NotImplementedError
    elif config.compress == "orthogonal":
        h = model.d_model
        orth = jax.random.orthogonal(key, n=h)
        wenc, wdec = orth, orth.T
    params_compressed = compress.update_params(params, wenc, wdec)
    params_compressed, idx = data_utils.flatten_params(
        params_compressed, config=config)
    
    out = x.copy()
    out['weights'] = params_compressed
    out['layer_idx'] = idx
    out['d_model'] = h
    return out


def ndata(dataset: Path | str):
    with h5py.File(dataset, "r", libver="latest") as f:
        assert "train/tokens" in f
        return f["train/tokens"].shape[0]
    
    
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
