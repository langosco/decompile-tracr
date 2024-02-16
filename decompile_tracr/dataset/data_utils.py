from itertools import islice
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional
try:
    import ujson as json
except ImportError:
    import json

import chex
import fcntl
import jax.flatten_util

import numpy as np
import chex
import jax
import jax.numpy as jnp

from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import config
from decompile_tracr.dataset.logger_config import setup_logger


logger = setup_logger(__name__)


def load_and_process_data(
        rng=None, 
        loaddir=config.full_dataset_dir,
        max_programs=1000,
        shuffle=False, 
        name="train",
        d_model=64,
        max_rasp_len=32,
        max_weights_len=8192,
    ) -> dict[str, np.ndarray]:
    """Load dataset and process it for model input."""
    logger.info(f"Loading data from {loaddir}.")
    data = load_batches(loaddir=loaddir, max_datapoints=max_programs)
    data = flatten_data(data)

    if shuffle:
        rng = np.random.default_rng(rng)
        rng.shuffle(data)

    data = process_data(
        data=data, 
        d_model=d_model, 
        max_rasp_len=max_rasp_len,
        max_weights_len=max_weights_len,
        name=name,
        filter_large_weights=False,
    )

    return data


def flatten_data(data: list[dict]) -> list[dict]:
    """
    Convert the list of models to a list of layers. Throw away
    model-level information like the rasp program and the compiled
    model.
    """
    out = []
    for program_id, program in enumerate(data):
        n_layers = len(program['weights'])
        assert len(program['tokens']) == n_layers
        for w, t in zip(program["weights"], program["tokens"]):
            out.append({
                "weights": w,
                "tokens": t,
                "program_id": program_id,
                "n_sops": program['n_sops'],
                "n_layers": n_layers,
            })
    return out


def process_data(
    data: list[dict],
    d_model: int,
    max_rasp_len: int = 32,
    max_weights_len: int = 8192,
    name=None,
    filter_large_weights: bool = False,
    ) -> dict[str, np.ndarray]:
    """Process a list of dicts into a dict of arrays for model input."""
    n = len(data)
    out = defaultdict(list)
    for x in data:
        x_proc = process_single_datapoint(
            x, d_model, max_rasp_len, max_weights_len, 
            filter_large_weights)

        if x_proc is None:
            continue

        for k, v in x_proc.items():
            out[k].append(v)

    out = {k: np.stack(v) for k, v in out.items()}
    out = {k: to_int(v) if k in ("tokens", "program_id", "n_sops") else v 
           for k, v in out.items()}
    
    if filter_large_weights:
        if name.startswith("test"):
            logger.warning(f"Received filter_large_weights=True for "
                            f"test data. This is probably a mistake.\n") 
        logger.info(f"Filtered out {n - len(out['tokens'])} datapoints "
                    f"({round(100 * (n - len(out['tokens'])) / n, 2)}%). "
                    f"Total remaining: {len(out['tokens'])}. ({name})]\n")
    # clip weights
    out["weights"] = np.clip(out["weights"], -100, 100)

    assert max_weights_len % d_model == 0
    chex.assert_shape(out["tokens"], (None, max_rasp_len))
    chex.assert_shape(out["weights"], (None, max_weights_len//d_model, d_model))
    assert len(out["tokens"]) == len(out["weights"])
    return out


def process_single_datapoint(
        x: dict[str, tuple],
        d_model: int,
        max_rasp_len: int = 32,
        max_weights_len: int = 8192,
        filter_weights: bool = False,
    ) -> dict[str, np.ndarray] | None:
    """Process a single datapoint (ie single layer) for model 
    input. Assume x is a dict with keys "tokens", "weights", 
    and "program_id".
    1) Rasp tokens: pad to max rasp length.
    2) Weights: pad to max_weights_len, then chunk.
    """
    if len(x['tokens']) > max_rasp_len:
        raise ValueError(f"Program length ({len(x['tokens'])}) exceeds "
                         f"max program length ({max_rasp_len}).")
    elif len(x['weights']) > max_weights_len:
        raise ValueError(f"Weights length ({len(x['weights'])}) exceeds "
                         f"max weights length ({max_weights_len}).")
    
    w_mean = np.mean(x['weights'])
    if filter_weights and np.abs(w_mean) > config.MAX_WEIGHTS_LAYER_MEAN:
        return None
    
    weights = pad_to(x['weights'], max_weights_len, pad_value=0.05)
    weights = pad_and_chunk(weights, d_model)  # (n_chunks, d_model)
    return {
        "tokens": pad_to(x['tokens'], max_rasp_len, pad_value=vocab.pad_id),
        "weights": weights,
        **{k: v for k, v in x.items() if k not in ("tokens", "weights")}
    }


def to_int(array: np.ndarray) -> np.ndarray:
    int_arr = array.astype(np.int32)
    assert np.allclose(array, int_arr), f"Array is not integer: {array}"
    return int_arr


def pad_and_chunk(arr: chex.Array, chunk_size: int) -> chex.Array:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    chunks = padded.reshape(-1, chunk_size)
    return chunks


def pad_to(x: np.ndarray, max_len: int, pad_value: int = 0):
    """Pad a 1D array to a given length. Not jittable."""
    x = np.array(x)
    assert len(x) <= max_len
    chex.assert_rank(x, 1)
    return np.pad(x, (0, max_len - len(x)), constant_values=pad_value)


def load_json(filename: str) -> list[dict]:
    with open(filename, "r") as f:
        return json.load(f)


def load_batches(
    loaddir: Path,
    max_datapoints: Optional[int] = None,
) -> list[dict]:
    """Load all json files in loaddir and merge into a single list. 
    """
    data = []
    for entry in os.scandir(loaddir):
        if entry.name.endswith(".json"):
            batch = load_json(entry.path)
            data.extend(batch)
        
        if max_datapoints is not None and len(data) >= max_datapoints:
            logger.info(f"load_batches: Loaded "
                        f"{len(data)} >= {max_datapoints} datapoints, stopping.")
            data = data[:max_datapoints]
            break

    if len(data) == 0:
        raise ValueError(f"load_batches: No files found in {loaddir}.")
    elif max_datapoints is not None and len(data) < max_datapoints:
        logger.warning(f"load_batches: Loaded {len(data)} < {max_datapoints} datapoints.")

    return data


def dedupe(data: list[dict]) -> list[dict]:
    """Deduplicate programs by RASP string.
    Assume data is a list of dicts that include the 
    key "tokens", as returned by load_batches().

    Deduplicate by adding the RASP string to a set.
    """
    reference = set()
    deduped = []

    logger.info(f"Deduplicating {len(data)} programs.")
    for x in data:
        tuple_tokens = tuple(tuple(t) for t in x['tokens'])
        if tuple_tokens in reference:
            continue
        reference.add(tuple_tokens)
        deduped.append(x)

    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({100*(len(data) - len(deduped)) / len(data)}%)")
    return deduped


def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 2) --> AB CD EF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def split_dict_data(data: dict, val_ratio: float = 0.1):
    """Split a dictionary of data arrays into train and validation sets.
    """
    train, val = {}, {}
    for k, v in data.items():
        split_index = int(len(v) * (1 - val_ratio))
        train[k], val[k] = v[:split_index], v[split_index:]
    return train, val


def save_batch(
        data: list,
        savedir: Path = config.unprocessed_dir,
        overwrite = False,
        filename = None,
    ):
    """Save to a json file."""
    os.makedirs(savedir, exist_ok=True)
    logger.info(f"Saving {len(data)} "
                f"datapoints to {savedir}.")

    if filename is None:
        idx = sequential_count_via_lockfile(savedir / "count.txt")
        filename = f"data_{idx}"
    else:
        filename = Path(filename)
    
    open_mode = "w" if overwrite else "x"
    with open(savedir / f"{filename}.json", open_mode) as f:
        json.dump(data, f)


def get_params(params: dict, layer_name: str) -> jax.Array:
    """
    params: hk parameters as returned by tracr compiler in model.params.
    layer_name: name of the layer to extract parameters for.
    
    Assume layer_name is in format `layer_n/attn` or `layer_n/mlp`.
    Return parameters for that layer as a 1-d array.
    """
    prefix = f'transformer/{layer_name}'

    if layer_name.endswith('attn'):
        layer_params = [
            params[f"{prefix}/{k}"] for k in ['key', 'query', 'value', 'linear']
        ]
    elif layer_name.endswith('mlp'):
        layer_params = [
            params[f'{prefix}/{k}'] for k in ['linear_1', 'linear_2']
        ]
    else:
        raise ValueError(f'Unknown layer name {layer_name}.')
    
    return jax.flatten_util.ravel_pytree(layer_params)[0]


def rw_lockfile(
        lockfile="/tmp/counter.txt", 
        get_new: callable = lambda x: ""
    ):
    """read content of lockfile, then append to it.
    Use a lockfile to ensure atomicity. If the file doesn't exist, 
    create it and start the counter at 1.
    """
    with open(lockfile, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)

        # read from start of file
        f.seek(0)
        content = f.read().strip()

        # write new content
        write_content = get_new(content)
        f.write(write_content)
        f.flush()

        fcntl.flock(f, fcntl.LOCK_UN)
    return content, write_content


def sequential_count_via_lockfile(countfile="/tmp/counter.txt") -> int:
    """Increment a counter in a file. 
    Use a lockfile to ensure atomicity. If the file doesn't exist, 
    create it and start the counter at 1."""
#    return rw_lockfile(countfile, lambda x: str(int(x) + 1))

    try:
        with open(countfile, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            f.seek(0)
            counter_str = f.read().strip()
            counter = 1 if not counter_str else int(counter_str) + 1

            f.seek(0)
            f.truncate()  # Clear the file content
            f.write(str(counter))
            f.flush()

            fcntl.flock(f, fcntl.LOCK_UN)

        return counter
    except ValueError as e:
        raise ValueError(f"Invalid counter value: {counter_str}. {e}")


def layer_names(n_layers: int) -> Generator[str, None, None]:
    assert n_layers % 2 == 0
    for i in range(n_layers // 2):
        yield f"layer_{i}/attn"
        yield f"layer_{i}/mlp"
