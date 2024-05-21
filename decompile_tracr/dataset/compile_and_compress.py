import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = 0.1
import argparse
from tqdm import tqdm
import psutil
import jax
import numpy as np
from jaxtyping import ArrayLike

from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.globals import disable_tqdm
from decompile_tracr.dataset.compile import compile_tokens_to_model, DataError, load_next_batch
from decompile_tracr.training import autoencoder


logger = setup_logger(__name__)
process = psutil.Process()


def compile_all(
        key: jax.random.PRNGKey, config: DatasetConfig, max_batches=None):
    """
    Load and compile rasp programs in batches.
    Save compiled programs to savedir.
    """
    assert config.compress is not None
    logger.info(f"Compiling RASP programs found in {config.paths.programs}.")
    for _ in range(max_batches or 10**8):
        key, subkey = jax.random.split(key)
        if compile_single_batch(subkey, config) is None:
            break
        jax.clear_caches()


def compile_single_batch(
        key: jax.random.PRNGKey, config: DatasetConfig) -> list[dict]:
    """Load and compile the next batch of rasp programs."""
    assert config.compress is not None
    data = load_next_batch(config.paths.programs)
    if data is None:
        return None

    i = 0
    batch = []
    for x in tqdm(data, disable=disable_tqdm, desc="Compiling"):
        key, subkey = jax.random.split(key)
        try:
            compressed = process_tokens(subkey, x['tokens'], config)
            to_save = to_datapoints(compressed, x)
            if config.n_augs > 0:
                data_utils.save_batch(to_save, config.paths.compiled_cache)
            else:
                batch.extend(to_save)
        except (InvalidValueSetError, NoTokensError, DataError) as e:
            logger.warning(f"Skipping program ({e}).")

        if i % 25 == 0:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({i}/{len(data)} programs compiled).")
        i += 1
    if config.n_augs == 0:
        data_utils.save_batch(batch, config.paths.compiled_cache)
    return data


def process_tokens(
        key: jax.random.PRNGKey, tokens: list[int], config: DatasetConfig):
    """
    1. Compile tokens to model.
    2. Apply SVD or train autencoder to compress residual stream.
    3. Augment encoding with random orthogonal matrices.
    4. For each encoding, construct a new set of transformer weights.
    """
    model_params, wenc, wdec = compile_and_train_encoder(tokens, config)
    key, subkey = jax.random.split(key)
    compressed = get_compressed_params(
        key=subkey, 
        model_params=model_params, 
        wenc=wenc,
        wdec=wdec,
        config=config,
    )
    return compressed


def compile_and_train_encoder(tokens: list[int], config: DatasetConfig):
    """Compile model and fit weights for encoder/decoder.
    """
    model = compile_tokens_to_model(tokens)
    params = model.params
    d_model = params['token_embed']['embeddings'].shape[-1]
    hidden_size = int(d_model // 1.1)

    flat = flatten_weights(params)
    if sum(len(x) for x in flat) > config.max_weights_length:
        raise DataError(f"Too many params (> {config.max_weights_length})")

    const_key = jax.random.key(123)
    if config.compress == "svd":
        train_out = autoencoder.train_svd(const_key, model, hidden_size)
        wenc, wdec = train_out['wenc'], train_out['wdec']
    elif config.compress == "autoencoder":
        train_out = autoencoder.train_autoencoder(
            const_key, model, nsteps=50_000, lr=2e-3, hidden_size=hidden_size)
        wenc, wdec = autoencoder.get_wenc_and_wdec(train_out['state'].params)
    else:
        raise ValueError(f"Unknown compression method: {config.compress}")
    
    logger.info(f'Accuracy: {train_out["accuracy"]}')
    logger.info(f'MSE:      {train_out["mse"]}')
    return model.params, wenc, wdec


def get_compressed_params(
    key: jax.random.PRNGKey, 
    model_params: dict, 
    wenc: ArrayLike,
    wdec: ArrayLike,
    config: DatasetConfig,
) -> list[dict]:
    """Given the original model parameters and a pair of encoder/decoder
    matrices, generate a compressed set of parameters by applying the
    encoder/decoder to the model parameters.
    If n_augs > 0, generate augmentations by applying random orthogonal
    transformations to the encoder and decoder matrices.
    Return flattened weights.
    """
    hidden_size = wenc.shape[-1]
    params_batch = [
        flatten_weights(autoencoder.update_params(
            model_params, wenc, wdec, w_orth=None))
    ]

    for i in range(config.n_augs):
        key, subkey = jax.random.split(key)
        w_orth = jax.random.orthogonal(subkey, n=hidden_size)
        p = autoencoder.update_params(
            model_params, wenc, wdec, w_orth)
        p = flatten_weights(p)
        params_batch.append(p)

        if i == 0 and sum(len(x) for x in p) > config.max_weights_length:
            raise DataError(f"Too many params (> {config.max_weights_length})")

    return params_batch


def flatten_weights(params: dict):
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
        key = jax.random.key(args.seed)
        compile_all(
            key=key,
            config=config,
            max_batches=args.max_batches,
        )
