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
        logger.warning(f"No h5 files found in {source_dir}.")
        return
    
    with h5py.File(target_file, "a", libver="latest") as target:
        for file in source_files:
            with h5py.File(file, "r", libver="latest") as source:
                if name not in target:
                    target.create_group(name)
                    init_h5(target[name], source)
                else:
                    append_h5(target[name], source)
            os.remove(file)
        
    logger.info(f"Merged {len(source_files)} files.")


def add_ids(dataset: Path) -> None:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset file {dataset} not found.")
    with h5py.File(dataset, "r+", libver="latest") as f:
        groups = set.intersection(set(f.keys()), {"train", "val", "test"})
        for group in groups:
            if "ids" in f[group]:
                del f[group]["ids"]
            f[group].create_dataset(
                name="ids", 
                data=np.arange(len(f[group]["tokens"])),
                maxshape=(None,),
            )


def make_test_splits(dataset: Path) -> None:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset file {dataset} not found.")
    def _save_split_to_new_group(f: h5py.File, n_split: int, group_name: str):
        assert group_name not in f, f"Group {group_name} already exists."
        assert n_split > 0
        assert all(n_split < len(v) for v in f["train"].values())
        split = {}
        for k, v in f["train"].items():
            split[k] = v[-n_split:]
            v.resize((v.shape[0] - n_split), axis=0)
        f.create_group(group_name)
        init_h5(f[group_name], split)

    split_frac = 0.03
    with h5py.File(dataset, "r+", libver="latest") as f:
        n_split = int(len(f['train/tokens']) * split_frac)
        _save_split_to_new_group(f, n_split, "val")
        _save_split_to_new_group(f, n_split, "test")


def init_h5(f: h5py.File, data: dict, maxn: int = 10**7):
    """Write dict to HDF5 datasets."""
    for k, v in data.items():
        v = np.array(v)
        f.create_dataset(k, data=v, maxshape=(maxn, *v.shape[1:]))


def append_h5(f: h5py.File, data: dict):
    """Write dict to HDF5 datasets. Assume datasets corresponding
    to the dict keys already exist and append to them."""
    for k, v in data.items():
        v = np.array(v)
        f[k].resize((f[k].shape[0] + v.shape[0]), axis=0)
        f[k][-v.shape[0]:] = v


# Data processing

def flatten_params(params: dict, config: DatasetConfig
                   ) -> tuple[np.ndarray, np.ndarray]:
    # order keys
    params = {k: params[k] for k in layer_names() if k in params}
    flat = [vv.flatten() for v in params.values() for vv in v.values()]
    sizes = [len(v) for v in flat]
    flat = np.concatenate(flat, dtype=NUMPY_DTYPE)
    maxw = config.max_weights_length
    if len(flat) > maxw:
        raise DataError(f"Too many params (> {maxw})")
    flat = pad_to(np.array(flat), maxw, 0.1)

    if len(sizes) > config.max_layers:
        raise DataError(f"Too many layers (> {config.max_layers})")
    sizes = pad_to(np.array(sizes), config.max_layers, 0)
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


def pad_to(x: np.ndarray, max_len: int, pad_value: int = 0):
    """Pad a 1D array to a given length. Not jittable."""
    assert len(x) <= max_len, f"Expected len(x) <= {max_len}, got {len(x)}."
    assert isinstance(x, np.ndarray), f"Expected np.ndarray, got {type(x)}."
    chex.assert_rank(x, 1)
    return np.pad(x, (0, max_len - len(x)), constant_values=pad_value)


# Data generation and json utils
def load_batches(loaddir: Path) -> list[dict]:
    """Load all json files in loaddir and return in a single list."""
    data = []
    for entry in os.scandir(loaddir):
        if entry.name.endswith(".json"):
            batch = load_json(entry.path)
            data.extend(batch)
        
    if len(data) == 0:
        raise FileNotFoundError(f"No json files found in {loaddir}.")
    return data


def load_json(filename: str) -> list[dict]:
    with open(filename, "r") as f:
        return json.load(f)


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
    if len(data) == 0:
        logger.warning("Got empty list - no data to save.")
        return None
    os.makedirs(savedir, exist_ok=True)
    savepath = savedir / (get_filename(rng) + ".h5")
    logger.info(f"Saving {len(data)} "
                f"datapoints to {savepath}")
    keys = data[0].keys()
    assert all(set(x.keys()) == keys for x in data)
    out = {k: [x[k] for x in data] for k in keys}
    out = {k: np.stack(v) for k, v in out.items()}
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
    MAX_TRIES = 1000
    for _ in range(MAX_TRIES):
        try:
            with open(lockfile, "x") as f:
                pass
            return
        except FileExistsError:
            time.sleep(0.1)
    raise FileExistsError(f"Could not acquire lockfile {lockfile} "
                          f"after {MAX_TRIES} tries ({MAX_TRIES*0.1}s).")


def release_lock(lockfile: Path | str) -> None:
    os.remove(lockfile)


class Lock:
    def __init__(self, lockfile: Path | str):
        self.lockfile = Path(lockfile)
        self.lockfile.parent.mkdir(exist_ok=True)

    def __enter__(self):
        acquire_lock(self.lockfile)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        release_lock(self.lockfile)


class DataError(Exception):
    pass


def track_idx_between_processes(config: DatasetConfig, name: str) -> int:
    """Helper function to keep track of an integer index between
    processes via a lockfile. Used loading batches of data
    when compiling or compressing compiled models.
    """
    # TODO: this could be a class instead, with a get_current_idx
    # method and a reset method.
    start, end = 0, config.compiling_batchsize
    tracker_file = config.paths.data_dir / name
    lockfile = config.paths.data_dir / f"{tracker_file}.lock"
    with Lock(lockfile):
        if not tracker_file.exists():
            with open(tracker_file, "w") as f:
                f.write(str(end))
        else:
            with open(tracker_file, "r") as f:
                start = int(f.read())
                end = start + config.compiling_batchsize
            
            with open(tracker_file, "w") as f:
                f.write(str(end))
    assert end > start, f"end: {end}, start: {start}"
    return start, end

