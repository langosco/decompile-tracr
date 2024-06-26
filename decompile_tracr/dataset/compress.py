"""Load tracr-compiled models and compress them."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import gc
import psutil

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


def compress_batches(
        config: DatasetConfig,
        splits: set[str] = {"train", "val", "test"},
    ) -> None:
    logger.info(f"Compressing RASP programs found in "
                f"{config.source_paths.dataset} and saving "
                f"to {config.paths.compiled_cache}.")

    with h5py.File(config.source_paths.dataset, "r", libver="latest") as f:
        groups = set.intersection(set(f.keys()), splits)

    key = jax.random.key(0)
    for group in groups:
        for batch in data_utils.async_iter_h5(
            dataset=config.source_paths.dataset,
            name=f"compress_idx_{group}",
            batch_size=config.compiling_batchsize,
            group=group,
        ):
            key, subkey = jax.random.split(key)
            batch = compress_batch(subkey, batch, config=config)
            if not Signals.n_sigterms >= 2:
                data_utils.save_h5(
                    batch, config.paths.compiled_cache, group=group)

            if Signals.sigterm:
                break

            del batch
            jax.clear_caches()
            gc.collect()


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
        logger.warning(f"Failed to compress datapoint: {e}")
        return None


def unsafe_compress_datapoint(key: PRNGKey, x: dict, config: DatasetConfig
                              ) -> dict:
    params = data_utils.unflatten_params(
        x['weights'], sizes=x['layer_idx'], d_model=x['d_model'])
    model = ModelFromParams(params, num_heads=x['n_heads'])
    if config.compress == "svd":
        h = int(model.d_model * 0.9)
        wenc, wdec, aux = compress.train_svd(model=model, hidden_size=h)
    elif config.compress == "autoencoder":
        raise NotImplementedError
    elif config.compress == "orthogonal":
        h = model.d_model
        orth = jax.random.orthogonal(key, n=h)
        wenc, wdec = orth, orth.T
    elif config.compress == "svd_orthogonal":
        h = int(model.d_model * 0.9)
        orth = jax.random.orthogonal(key, n=h)
        wenc, wdec, aux = compress.train_svd(model=model, hidden_size=h)
        wenc, wdec = wenc @ orth, wdec @ orth.T
    else:
        raise ValueError(f"Invalid compression method: {config.compress}")
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
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    parser.add_argument('--split', type=str, default=None,
                        help="Name of dataset split to compress. Default all.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.default_rng(None).integers(0, 2**32)
    
    config = load_config(args.config)
    compress_batches(
        config=config, 
        splits={args.split} if args.split else {"train", "val", "test"},
    )
