from itertools import islice
import os
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional
import time
try:
    import ujson as json
except ImportError:
    import json
import fcntl
import h5py
import numpy as np

import jax
import jax.flatten_util
from jax import numpy as jnp
import chex

from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig


logger = setup_logger(__name__)
NUMPY_DTYPE = np.float32
default_config = DatasetConfig()
default_datadir = default_config.paths.data_dir


# HDF5 utils

def load_json_and_save_to_hdf5(config: DatasetConfig) -> None:
    """Convert dataset in data_dir/full to a single HDF5 file."""
    batch_size = 50
    done = False
    while not done:
        done = _batch_to_hdf5(config=config, max_files=batch_size)


def make_test_splits(dataset: Path) -> None:
    def _save_split_to_new_group(f: h5py.File, n_split: int, group_name: str):
        split = {}
        for k, v in f["train"].items():
            split[k] = v[-n_split:]
            v.resize((v.shape[0] - n_split), axis=0)
        f.create_group(group_name)
        init_h5(f[group_name], split)

    split_frac = 0.03
    with h5py.File(dataset, "r+") as f:
        n_split = int(len(f['train/tokens']) * split_frac)
        _save_split_to_new_group(f, n_split, "val")
        _save_split_to_new_group(f, n_split, "test")


def _batch_to_hdf5(config: DatasetConfig, max_files: int = 500):
    """Convert dataset in data_dir/full to a single HDF5 file.
    Run multiple times if the dataset is too large to fit in memory.
    - Load up to max_data datapoints. 
    - Save as HDF5 (append to existing file if it exists)
    - Delete original files.
    """
    loaddir = config.paths.compiled_cache

    data, files_loaded = load_batches(
        loaddir, max_files=max_files, return_filenames=True)
    data = dataset_to_arrays(data=data, config=config)

    with h5py.File(config.paths.dataset, "a", libver="latest") as f:
        if len(f.keys()) == 0:
            f.create_group("train")
            init_h5(f["train"], data)
        else:
            append_h5(f["train"], data)

    for filename in files_loaded:
        os.remove(loaddir / filename)
    
    remaining = os.scandir(loaddir)
    done = not any(entry.name.endswith(".json") for entry in remaining)
    return done


def init_h5(f: h5py.File, data: dict):
    """Write dict to HDF5 datasets."""
    for k, v in data.items():
        f.create_dataset(k, data=v, maxshape=(10**7, *v.shape[1:]))


def append_h5(f: h5py.File, data: dict):
    """Write dict to HDF5 datasets. Assume datasets corresponding
    to the dict keys already exist and append to them."""
    assert set(f.keys()) == set(data.keys()), (
        f"Keys in existing HDF5 file {set(f.keys())} do not match "
        f"keys in data {set(data.keys())}.")
    for k, v in data.items():
        f[k].resize((f[k].shape[0] + v.shape[0]), axis=0)
        f[k][-v.shape[0]:] = v


def dataset_to_arrays(
    data: list[dict],
    config: DatasetConfig,
) -> dict[str, np.ndarray]:
    """Process a list of dicts into a dict of arrays for model input.
    Deletes original datapoints (i.e. elements of data) to free memory.
    """
    out = defaultdict(list)

    while data:
        for k, v in datapoint_to_array(data.pop(0), config).items():
            out[k].append(v)

    # stack to np.arrays
    out = {k: np.stack(v) for k, v in out.items()}
    out = {k: to_int(v) if k in ("tokens", "n_sops", "n_layers") else v 
           for k, v in out.items()}
    
    # asserts
    chex.assert_shape(out["tokens"], (None, config.max_rasp_length))
    chex.assert_shape(out["weights"], (None, config.max_weights_length))
    chex.assert_shape(out["layer_idx"], (None, config.max_layers))
    assert len(out["tokens"]) == len(out["weights"])
    return out


def datapoint_to_array(x: dict, config: DatasetConfig) -> dict[str, np.ndarray]:
    tokens = pad_to(
        np.array(x['tokens']), 
        config.max_rasp_length, 
        pad_value=vocab.pad_id
    )
    layer_idx = np.cumsum([len(w) for w in x['weights']])
    layer_idx = pad_to(layer_idx, config.max_layers, pad_value=0)
    weights = np.concatenate(x['weights']).astype(NUMPY_DTYPE)
    weights = pad_to(weights, config.max_weights_length, pad_value=0.05)
    return {
        'tokens': tokens,
        'weights': weights,
        'layer_idx': layer_idx,
        'n_sops': x['n_sops'],
        'n_layers': len(x["weights"])-1,
    }


# Misc utils

def get_tokens_by_layer(tokens: list[int]):
    """Split a list of tokens into a list of layers.
    """
    # TODO: optionally we could add info about layer numbering - 
    # probably in form of extra tokens at the start of each layer.
    layers = []
    current = []
    for t in tokens[1:-1]:  # skip eos and bos
        current.append(t)
        if t == vocab.eol_id:
            layers.append(current)
            current = []
    for l in layers:
        l.insert(0, vocab.bos_id)
        assert l[-1] == vocab.eol_id
        l[-1] = vocab.eos_id
    return layers


def to_int(array: np.ndarray) -> np.ndarray:
    int_arr = array.astype(np.int64)
    assert np.allclose(array, int_arr), f"Array is not integer: {array}"
    return int_arr


def pad_and_chunk(arr: chex.Array, chunk_size: int) -> chex.Array:
    pad_size = -len(arr) % chunk_size
    padded = np.pad(arr, (0, pad_size))
    chunks = padded.reshape(-1, chunk_size)
    return chunks


def pad_to(x: np.ndarray, max_len: int, pad_value: int = 0):
    """Pad a 1D array to a given length. Not jittable."""
    assert len(x) <= max_len, f"Expected len(x) <= {max_len}, got {len(x)}."
    assert isinstance(x, np.ndarray), f"Expected np.ndarray, got {type(x)}."
    chex.assert_rank(x, 1)
    return np.pad(x, (0, max_len - len(x)), constant_values=pad_value)


# Data generation and json utils

def load_batches(
    loaddir: Path,
    max_data: Optional[int] = None,
    max_files: Optional[int] = None,
    return_filenames: bool = False,
) -> list[dict]:
    """Load all json files in loaddir and merge into a single list. 
    """
    data = []
    files_loaded = []
    t = time.time()
    for entry in os.scandir(loaddir):
        if entry.name.endswith(".json"):
            batch = load_json(entry.path)
            data.extend(batch)
            files_loaded.append(entry.name)
        
        if max_data is not None and len(data) >= max_data:
            logger.info(f"load_batches: Loaded "
                        f"{len(data)} >= {max_data} datapoints. Stopping "
                        f"and truncating to {max_data} datapoints. "
                        f"({time.time() - t:.2f}s)")
            data = data[:max_data]
            break
        
        if max_files is not None and len(files_loaded) >= max_files:
            logger.info(f"load_batches: Loaded {len(files_loaded)} files. "
                        f"Stopping. ({time.time() - t:.2f}s).")
            break

    if len(data) == 0:
        raise FileNotFoundError(f"load_batches: No files found in {loaddir}.")
    elif max_data is not None and len(data) < max_data:
        logger.warning(f"load_batches: Loaded {len(data)} < {max_data} "
                       f"datapoints. {time.time() - t:.2f}s")

    if return_filenames:
        return data, files_loaded
    else:
        return data


def load_json(filename: str) -> list[dict]:
    with open(filename, "r") as f:
        return json.load(f)


def load_batches_from_subdirs(
    loaddir: Path,
    max_data_per_subdir: Optional[int] = None,
) -> list[dict]:
    """Load all json files in loaddir/subdirs and merge into a single list."""
    subdirs = list(loaddir.iterdir())
    subdirs = [x for x in subdirs if x.is_dir()]
    data = [load_batches(d, max_data=max_data_per_subdir) for d in subdirs]
    data = [x for ds in data for x in ds]
    return data


def dedupe(data: list[dict], reference: Optional[list[dict]] = None,
) -> list[dict]:
    """Deduplicate programs by RASP string.
    Assume data is a list of dicts that include the 
    key "tokens", as returned by load_batches().

    Args:
    - data: list of dicts with keys "tokens".
    - reference: list of dicts with keys "tokens". If provided,
    treat examples in data that match elements of reference as duplicates.
    """
    if reference is None:
        reference: set[list[int]] = set()
    else:
        reference = set([tuple(x['tokens']) for x in reference])
    deduped: list[dict] = []

    logger.info(f"Deduplicating {len(data)} programs.")
    logger.info(f"Reference set size: {len(reference)}")
    for x in data:
        tokens = tuple(x['tokens'])
        if tokens in reference:
            continue
        reference.add(tokens)
        deduped.append(x)

    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({100*(len(data) - len(deduped)) / len(data)}%)")
    logger.info(f"Remaining new datapoints: {len(deduped)}")

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
    savedir: Path | str,
    overwrite = False,
    filename = None,
) -> None:
    """Save data to a json file.
    If filename is not set, use a counter to generate a new 
    filename.
    """
    savedir = Path(savedir)
    os.makedirs(savedir, exist_ok=True)

    if filename is None:
        idx = sequential_count_via_lockfile(savedir / "count.txt")
        filename = f"data_{idx}"
    else:
        filename = Path(filename)
    
    savepath = savedir / f"{filename}.json"
    logger.info(f"Saving {len(data)} "
                f"datapoints to {savepath}")
    open_mode = "w" if overwrite else "x"
    with open(savepath, open_mode) as f:
        json.dump(data, f)


def get_params(
    params: dict, layer_name: str, include_unflatten_fn: bool = False,
) -> jax.Array:
    """
    params: hk parameters as returned by tracr compiler in model.params.
    layer_name: name of the layer to extract parameters for.
    
    Assume layer_name is in format `layer_n/attn` or `layer_n/mlp`.
    Return parameters for that layer as a 1-d array.
    """
    prefix = f'transformer/{layer_name}'

    if layer_name.endswith('attn'):
        layer_params = [
            params[f"{prefix}/{k}"] for k in ['key', 'query', 
                                              'value', 'linear']
        ]
    elif layer_name.endswith('mlp'):
        layer_params = [
            params[f'{prefix}/{k}'] for k in ['linear_1', 'linear_2']
        ]
    elif layer_name == 'embed':
        layer_params = [params['token_embed'], params['pos_embed']]
    else:
        raise ValueError(f'Unknown layer name {layer_name}.')
    
    flat, unflatten = jax.flatten_util.ravel_pytree(layer_params)
    return flat if not include_unflatten_fn else (flat, unflatten)


def rw_lockfile(
    lockfile="/tmp/counter.txt", 
    get_new: callable = lambda x: ""
) -> tuple[str, str]:
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
    assert n_layers % 2 == 0, "n_layers must be even."
    yield "embed"
    for i in range(n_layers // 2):
        yield f"layer_{i}/attn"
        yield f"layer_{i}/mlp"