from typing import Optional
from pathlib import Path
import chex
from decompile_tracr.globals import on_cluster
from decompile_tracr.globals import hpc_storage_dir
from decompile_tracr.globals import module_path


def default_data_dir() -> Path:
    if on_cluster:
        data_dir = hpc_storage_dir / "lauro/rasp-data/"
    else:
        data_dir = module_path / "data/"
    return data_dir


class DatasetPaths:
    """Set up paths for the dataset and cache.
    Pipeline: .cache/programs -> programs -> .cache/compiled -> dataset.h5
    """
    def __init__(self, data_dir: Optional[str | Path] = None):
        if data_dir is None:
            data_dir = default_data_dir()
        self.data_dir = Path(data_dir)
        self.programs_cache  = data_dir / ".cache/programs"
        self.compiled_cache  = data_dir / ".cache/compiled"
        self.programs        = data_dir / "programs"  # for deduped programs
        self.dataset         = data_dir / "dataset.h5"


@chex.dataclass
class DatasetConfig:
    ndata: int = 100
    program_length: Optional[int] = 8
    max_rasp_length: Optional[int] = 256
    max_weights_length: Optional[int] = 65_536
    max_layers: Optional[int] = 25
    data_dir: Optional[Path] = None
    name: Optional[str] = "default"
    compiling_batchsize: Optional[int] = 180  # constrained by cpu mem
    compress: Optional[str] = None  # "svd" or "autoencoder"
    n_augs: Optional[int] = None  # number of augmentations

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = default_data_dir()
        self.paths = DatasetPaths(self.data_dir)

        if self.compress is not None:
            assert self.compress in ["svd", "autoencoder"]
            assert self.n_augs is not None, "Number of augmentations must be set."
        else:
            assert self.n_augs is None, ("Augmentations are only possible "
                                          "when compressing.")
        



# Presets
def load_config(name: Optional[str] = None) -> DatasetConfig:
    if name is None:
        name = "default"
    return _presets[name]


base_data_dir = default_data_dir()

_presets = {
    "default": DatasetConfig(
        data_dir=base_data_dir / "default",
    ),

    "small_compressed": DatasetConfig(
        ndata=10_000,
        program_length=5,
        max_rasp_length=128,
        max_weights_length=32_768,
        max_layers=15,
        compiling_batchsize=50,
        name="small",
        compress="svd",
        n_augs=10,
        data_dir=base_data_dir / "small_compressed",
    ),

    "range": DatasetConfig(
        ndata=1_000,
        program_length=[4,5,6,7,8],
        max_rasp_length=128,
        max_weights_length=65_536,
        max_layers=25,
        compiling_batchsize=180,
        name="range_4_8",
        data_dir=base_data_dir / "range",
    ),


    "small_compressed_no_augs": DatasetConfig(
        ndata=10_000,
        program_length=5,
        max_rasp_length=128,
        max_weights_length=32_768,
        max_layers=15,
        compiling_batchsize=200,
        name="small",
        compress="svd",
        n_augs=0,
        data_dir=base_data_dir / "small_compressed_no_augs",
    ),

}

