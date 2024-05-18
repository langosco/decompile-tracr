import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = 0.1
from pathlib import Path
import fcntl
import argparse
from tqdm import tqdm
import shutil
import psutil
import jax
import numpy as np

from tracr.compiler import compile_rasp_to_model
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.globals import disable_tqdm
from decompile_tracr.dataset.compile import compile_tokens_to_model, delete_existing, DataError, get_next_filename, load_next_batch
from decompile_tracr.training import autoencoder, metrics


logger = setup_logger(__name__)
process = psutil.Process()


def compile_all(
        key: jax.random.PRNGKey, config: DatasetConfig, max_batches=None):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    assert config.compress
    logger.info(f"Compiling RASP programs found in {config.paths.programs}.")
    for _ in range(max_batches or 10**8):
        key, subkey = jax.random.split(key)
        if compile_single_batch(subkey, config) is None:
            break
        jax.clear_caches()


def compile_single_batch(
        key: jax.random.PRNGKey, config: DatasetConfig) -> list[dict]:
    """Load and compile the next batch of rasp programs."""
    assert config.compress
    data = load_next_batch(config.paths.programs)
    if data is None:
        return None

    i = 0
    for x in tqdm(data, disable=disable_tqdm, desc="Compiling"):
        key, subkey = jax.random.split(key)
        try:
            model_params, aenc_params = get_params(x['tokens'])
            compressed = get_augmented_compressed_params(
                key=subkey, 
                model_params=model_params, 
                autoencoder_params=aenc_params, 
                max_weights_len=config.max_weights_length
            )
            to_save = to_datapoints(compressed, x)
            data_utils.save_batch(to_save, config.paths.compiled_cache)
        except (InvalidValueSetError, NoTokensError, DataError) as e:
            logger.warning(f"Skipping program ({e}).")


        if i % 25 == 0:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({i}/{len(data)} programs compiled).")
        i += 1
    return data


def get_params(tokens: list[int]):
    """Get params of compiled model and autoencoder."""
    model = compile_tokens_to_model(tokens)
    params = model.params
    d_model = params['token_embed']['embeddings'].shape[-1]
    hidden_size = int(d_model // 1.1)

    # Train autoencoder
    const_key = jax.random.key(123)
    state, log, aenc = autoencoder.train_autoencoder(
        const_key, model, nsteps=50_000, lr=2e-3, hidden_size=hidden_size)
    
    return model.params, state.params


def get_augmented_compressed_params(
    key: jax.random.PRNGKey, 
    model_params: dict, 
    autoencoder_params: dict, 
    max_weights_len: int,
    n_augs: int = 50,
) -> list[dict]:
    hidden_size = autoencoder_params['encoder']['kernel'].shape[-1]
    params_batch = []
    for i in range(n_augs):
        key, subkey = jax.random.split(key)
        w_orth = jax.random.orthogonal(subkey, n=hidden_size)
        p = autoencoder.update_params(
            model_params, autoencoder_params, w_orth)
        p = flatten_weights(p, max_weights_len)
        params_batch.append(p)

        if i == 0 and sum(len(x) for x in p) > max_weights_len:
            raise DataError(f"Too many params (> {max_weights_len})")

    return params_batch


def flatten_weights(params: dict, max_weights_len: int):
    n_layers = 2 * max(int(k[18]) + 1 for k in params.keys() 
                       if "layer" in k)
    flat_weights = [
        data_utils.get_params(params, layername) 
        for layername in data_utils.layer_names(n_layers)
    ]
    return [x.tolist() for x in flat_weights]


def to_datapoints(compressed: list, x: dict):
    datapoints = []
    for i, params in enumerate(compressed):
        example = x.copy()
        example['weights'] = params
        datapoints.append(example)
    return datapoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help="maximum number of batches to compile.")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.default_rng(None).integers(0, 2**32)
    
    config = load_config(args.config)
    key = jax.random.key(args.seed)
    compile_all(
        key=key,
        config=config,
        max_batches=args.max_batches,
    )
