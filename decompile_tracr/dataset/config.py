from typing import Optional
from pathlib import Path
import chex
from decompile_tracr.globals import on_cluster
from decompile_tracr.globals import hpc_storage_dir
from decompile_tracr.globals import module_path


def default_base_data_dir() -> Path:
    if on_cluster:
        data_dir = hpc_storage_dir / "lauro/rasp-data/"
    else:
        data_dir = module_path / "data/"
    return data_dir


class DatasetPaths:
    """Set up paths for the dataset and cache.
    Pipeline: .cache/programs -> programs -> .cache/compiled -> dataset.h5
    """
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.programs_cache  = data_dir / ".cache/programs"
        self.compiled_cache  = data_dir / ".cache/compiled"
        self.programs        = data_dir / "programs"  # for deduped programs
        self.dataset         = data_dir / "dataset.h5"


@chex.dataclass
class DatasetConfig:
    base_data_dir: Path = default_base_data_dir()
    ndata: int = 10_000
    program_length: int = 8
    max_rasp_length: int = 256
    max_weights_length: int = 65_536
    max_layers: int = 128
    compiling_batchsize: int = 100  # constrained by cpu mem
    compress: str = None  # "svd" or "autoencoder"
    n_augs: int = None  # number of augmentations
    source_data_dir: Path = None
    name: str = "default"

    def __post_init__(self):
        self.data_dir = self.base_data_dir / self.name
        self.paths = DatasetPaths(self.data_dir)

        if self.source_data_dir is not None:
            if self.compress is None:
                raise ValueError(
                    "source_data_dir specified but no compression "
                    "method specified."
                )
            self.source_paths = DatasetPaths(self.source_data_dir)
        else:
            if self.compress is not None:
                raise ValueError("No source directory specified.")

        if self.compress is not None:
            assert self.compress in ["svd", "autoencoder", "orthogonal"]
            assert self.n_augs is not None, (
                "Number of augmentations must be set.")
        else:
            assert self.n_augs is None, (
        "Augmentations are only possible when compressing.")


# Presets
def load_config(name: Optional[str] = None) -> DatasetConfig:
    if name is None:
        name = "default"
    return _presets[name]


base_data_dir = default_base_data_dir()

# TODO: instead of specifying name twice, do something better
_presets = {
    "default": DatasetConfig(),

    "small": DatasetConfig(
        ndata=100,
        program_length=5,
        max_rasp_length=128,
        max_weights_length=32_768,
        compiling_batchsize=100,
        name="small",
    ),

    "range": DatasetConfig(
        ndata=1_000,
        program_length=[4,5,6,7,8],
        max_rasp_length=128,
        max_weights_length=65_536,
        max_layers=50,
        compiling_batchsize=180,
        name="range",
    ),

    "compressed": DatasetConfig(
        compiling_batchsize=20,
        compress="orthogonal",
        n_augs=0,
        source_data_dir=base_data_dir / "default",
        name="compressed",
    ),

    "small_compressed": DatasetConfig(
        ndata=10_000,
        program_length=5,
        max_rasp_length=128,
        max_weights_length=32_768,
        compiling_batchsize=30,
        compress="svd",
        n_augs=0,
        source_data_dir=base_data_dir / "small",
        name="small_compressed",
    ),
}

