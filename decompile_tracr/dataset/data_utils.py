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
import h5py
import numpy as np

import chex
from jaxtyping import ArrayLike

from decompile_tracr.tokenize import vocab
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig


logger = setup_logger(__name__)
NUMPY_DTYPE = np.float32
default_config = DatasetConfig()
default_datadir = default_config.paths.data_dir


# HDF5 utils

def merge_h5(
    config: DatasetConfig,
    name: str = "train",
) -> None:
    """Merge all h5 files in source_dir into a single file.
    """
    source_dir = config.paths.compiled_cache
    target_file = config.paths.dataset
    source_files = list(source_dir.glob("*.h5"))
    if len(source_files) == 0:
        raise FileNotFoundError(f"No h5 files found in {source_dir}.")
    
    with h5py.File(target_file, "w") as target:
        for file in source_files:
            with h5py.File(file, "r") as source:
                if name not in target:
                    target.create_group(name)
                    init_h5(target[name], source)
                else:
                    append_h5(target[name], source)
            os.remove(file)
    logger.info(f"Merged {len(source_files)} files.")


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


def init_h5(f: h5py.File, data: dict, maxn: int = 10**7):
    """Write dict to HDF5 datasets."""
    for k, v in data.items():
        f.create_dataset(k, data=v, maxshape=(maxn, *v.shape[1:]))


def append_h5(f: h5py.File, data: dict):
    """Write dict to HDF5 datasets. Assume datasets corresponding
    to the dict keys already exist and append to them."""
    assert set(f.keys()) == set(data.keys()), (
        f"Keys in existing HDF5 file {set(f.keys())} do not match "
        f"keys in data {set(data.keys())}.")
    for k, v in data.items():
        f[k].resize((f[k].shape[0] + v.shape[0]), axis=0)
        f[k][-v.shape[0]:] = v


# Data processing

def flatten_params(params: dict, max_len: int):
    # order keys
    params = {k: params[k] for k in layer_names() if k in params}
    flat = [vv.flatten() for v in params.values() for vv in v.values()]
    sizes = [len(v) for v in flat]
    flat = np.concatenate(flat)
    if len(flat) > max_len:
        raise DataError(f"Too many params (> {max_len})")
    flat = pad_to(np.array(flat), max_len, 0.)
    return flat, sizes


def unflatten_params(flat: ArrayLike, sizes: ArrayLike, d_model: int):
    """Inverse of flatten_params."""
    sizes = np.array(sizes)
    sizes = sizes[sizes > 0]
    # order keys
    split = np.split(flat, np.cumsum(sizes))
    params = dict()
    for k in layer_names():
        wshape = get_w_shape(k, d_model)
        if k.endswith('embed'):
            x = split.pop(0)
            params[k] = {'embeddings': x.reshape(wshape)}
        else:
            params[k] = {
                'b': split.pop(0),
                'w': split.pop(0).reshape(wshape),
            }
        if len(split) <= 1:
            break
    return params


def get_w_shape(layer_name: str, d: int):
    if (layer_name in ["pos_embed", "token_embed"] 
        or layer_name.endswith(("attn/linear", "mlp/linear_2"))):
        return (-1, d)
    elif layer_name.endswith(("key", "query", "value", "mlp/linear_1")):
        return (d, -1)
    else:
        raise ValueError(f"Unknown layer name: {layer_name}")


def dataset_to_arrays(
    data: list[dict],
    config: DatasetConfig,
) -> dict[str, np.ndarray]:
    """Process a list of dicts into a dict of arrays for model input.
    Deletes original datapoints (i.e. elements of data) to free memory.
    """
    out = defaultdict(list)

    while data:
        for k, v in datapoint_to_arrays(data.pop(0), config).items():
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


def datapoint_to_arrays(x: dict, config: DatasetConfig) -> dict[str, np.ndarray]:
    tokens = pad_to(
        np.array(x['tokens']),
        config.max_rasp_length, 
        pad_value=vocab.pad_id,
    )
    layer_idx = pad_to(np.array(x['layer_idx'], dtype=int), 
                       config.max_layers, pad_value=0)
    weights = np.array(x['weights'], dtype=NUMPY_DTYPE)
    pad_value = 0.05 if not config.compress else 0
    weights = pad_to(weights, config.max_weights_length, pad_value=pad_value)
    out = {
        'tokens': tokens,
        'weights': weights,
        'layer_idx': layer_idx,
    }
    out.update({k: v for k, v in x.items() 
                if (k not in out
                    and not isinstance(v, str)
                    and np.isscalar(v))})
    return out


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
        if tokens not in reference:
            reference.add(tokens)
            deduped.append(x)
        

    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({100*(len(data) - len(deduped)) / len(data)}%)")
    logger.info(f"Remaining new datapoints: {len(deduped)}")

    return deduped


def batched(iterable, n):
    """Batch data into lists of length n. 
    The last batch may be shorter.
    """
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


def get_filename(rng: np.random.Generator = None) -> str:
    rng = np.random.default_rng(rng)
    idx = rng.integers(0, 2**63)  # p(collision) is <1e-8 for 1e5 files
    return f"data_{idx}"


def save_h5(
    data: list[dict],
    savedir: Path | str,
    rng: np.random.Generator = None,
) -> None:
    os.makedirs(savedir, exist_ok=True)
    savepath = savedir / (get_filename(rng) + ".h5")
    logger.info(f"Saving {len(data)} "
                f"datapoints to {savepath}")
    keys = data[0].keys()
    assert all(set(x.keys()) == keys for x in data)
    out = {k: [x[k] for x in data] for k in keys}
    out = {k: np.stack(v) for k, v in out.items()}
    out = {k: to_int(v) if k in ("tokens", "n_sops", "n_layers") else v 
        for k, v in out.items()}
    with h5py.File(savepath, 'a', libver='latest') as f:
        for k, v in out.items():
            f.create_dataset(k, data=v)


def save_json(
    data: list[dict],
    savedir: Path | str,
    overwrite = False,
    filename: str = None,
    rng: np.random.Generator = None,
) -> None:
    """Save data to a json file.
    If filename is not set, use a counter to generate a new 
    filename.
    """
    os.makedirs(savedir, exist_ok=True)
    filename = filename or get_filename(rng)
    savepath = Path(savedir) / (filename + ".json")
    logger.info(f"Saving {len(data)} "
                f"datapoints to {savepath}")
    open_mode = "w" if overwrite else "x"
    with open(savepath, open_mode) as f:
        json.dump(data, f)


def layer_names() -> Generator[str, None, None]:
    yield "pos_embed"
    yield "token_embed"
    for i in range(100): # what would you want more than 100 layers for
        yield f"transformer/layer_{i}/attn/key"
        yield f"transformer/layer_{i}/attn/query"
        yield f"transformer/layer_{i}/attn/value"
        yield f"transformer/layer_{i}/attn/linear"
        yield f"transformer/layer_{i}/mlp/linear_1"
        yield f"transformer/layer_{i}/mlp/linear_2"


def acquire_lock(lockfile: Path | str) -> None:
    """Acquire a lockfile. If the lockfile already exists, wait and try again.
    """
    try:
        with open(lockfile, "x") as f:
            pass
    except FileExistsError:
        time.sleep(0.1)
        return acquire_lock(lockfile)
    return None


def release_lock(lockfile: Path | str) -> None:
    os.remove(lockfile)
    return None


class Lock:
    def __init__(self, lockfile: Path | str):
        self.lockfile = lockfile

    def __enter__(self):
        acquire_lock(self.lockfile)
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        release_lock(self.lockfile)
        return None


class DataError(Exception):
    pass
