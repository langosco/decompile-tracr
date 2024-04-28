from pathlib import Path
from typing import Optional
import h5py
import numpy as np

import jax
import jax.flatten_util
from jax import numpy as jnp

from decompile_tracr.dataset import config


class DataLoader:
    """Generator that loads data from an HDF5 file and yields it in batches.
    Stores auxiliary information such as the shape of the dataset.
    """
    def __init__(
        self,
        loadfile: Path = config.data_dir / "full.h5",
        group: str = "train",
        batch_size: int = 32,
        process_fn: Optional[callable] = None,
        max_datapoints: Optional[int] = -1,
    ):
        with h5py.File(loadfile, "r") as f:
            n = f[f"{group}/tokens"].shape[0]
            self.shape = {k: v.shape for k, v in f[group].items()}
            _check_dataset_shapes(f[group], n)

        n = n if max_datapoints == -1 else min(n, max_datapoints)
        self.length = n // batch_size
        self.ndata = n // batch_size * batch_size  # skip last incomplete batch
        self.epoch_count = 0

        self.loadfile = loadfile
        self.group = group
        self.batch_size = batch_size
        self.process_fn = process_fn if process_fn is not None else lambda x: x
        self.max_datapoints = max_datapoints

        # run a dummy `process_fn` to check the output shape
        dummy_data = {k: np.zeros((batch_size, *v[1:]))
                       for k, v in self.shape.items()}
        dummy_data = process_fn(dummy_data)
        out_shapes = {k: v.shape for k, v in dummy_data.items()}
        for k in self.shape.keys():
            self.shape[k] = (self.shape[k][0],) + out_shapes[k][1:]

    def __iter__(self):
        """Load data from an HDF5 file and yield it in batches."""
        if not self.loadfile.exists():
            raise FileNotFoundError(f"File {self.loadfile} not found.")
        with h5py.File(self.loadfile, "r") as f:
            n = f[f"{self.group}/tokens"].shape[0]
            _check_dataset_shapes(f[self.group], n)
        
        for i in range(0, self.ndata, self.batch_size):
            with h5py.File(self.loadfile, "r") as f:
                data = {
                    k: v[i:i+self.batch_size] 
                    for k, v in f[self.group].items()
                }
                yield self.process_fn(data)
        
        self.epoch_count += 1

    def __len__(self):
        return self.length


def load_dataset_for_model_input(
    loadfile: Path = config.data_dir / "full.h5",
    ndata: int = -1,
    group: str = "train",
) -> dict[str, np.ndarray]:
    """Load dataset of tracr-compiled base models and process it 
    for model input. Returns a dict of np.ndarrays with keys 'weights',
    'tokens', 'n_sops', 'n_layers'.
    """
    if not loadfile.exists():
        raise FileNotFoundError(f"File {loadfile} not found.")
    with h5py.File(loadfile, "r") as f:
        _check_dataset_shapes(f[group], ndata)
        train_data = {k: v[:ndata] for k, v in f[group].items()}
    return train_data


def _check_dataset_shapes(g: h5py.Group, ndata: int):
    n = g["tokens"].shape[0]
    assert ndata == -1 or n >= ndata, (
        f"Expected at least {ndata} datapoints, got {n}")
    for k, v in g.items():
        assert v.shape[0] == n, (
            f"len(tokens) = {n}, but len({k}) = {v.shape[0]}")


@jax.jit
def symlog(x: np.ndarray, linear_thresh: float = 2.) -> np.ndarray:
    """Symmetric log transform.
    """
#    assert linear_thresh > 1., "linear_thresh must be > 1."
    slog = jnp.sign(x) * jnp.log(jnp.abs(x))
    return jnp.where(
        jnp.abs(x) < linear_thresh, 
        x * jnp.log(linear_thresh)/linear_thresh,
        slog
    )