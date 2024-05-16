from typing import Optional
from pathlib import Path
import chex
from decompile_tracr.globals import on_cluster
from decompile_tracr.globals import hpc_storage_dir
from decompile_tracr.globals import module_path


# program len 5:
# MAX_RASP_LENGTH = 128
# MAX_WEIGHTS_LENGTH = 16_384


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
    program_length: Optional[int] = 10
    max_rasp_length: Optional[int] = 256
    max_weights_length: Optional[int] = 65_536
    max_layers: Optional[int] = 25
    data_dir: Optional[Path] = None
    name: Optional[str] = "default"
    compiling_batchsize: Optional[int] = 180  # constrained by cpu mem

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = default_data_dir()
        self.paths = DatasetPaths(self.data_dir)