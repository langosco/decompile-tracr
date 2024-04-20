import os
import sys
from pathlib import Path

# Set up global variables
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = Path("/rds/project/rds-eWkDxBhxBrQ")
global_disable_tqdm = not interactive


# Set up global constants
# per layer maximum size of RASP program and weights:

# program len 5:
MAX_RASP_LENGTH = 128
MAX_WEIGHTS_LENGTH = 16_384

# program len 10:
# MAX_RASP_LENGTH = 256
# MAX_WEIGHTS_LENGTH = 65_536


def set_data_dir():
    module_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../..'))
    module_path = Path(module_path)

    if on_cluster:
        data_dir = hpc_storage_dir / "lauro/rasp-data/"
    else:
        data_dir = module_path / "data/"
    return data_dir, module_path


data_dir, module_path = set_data_dir()

# Set up global default paths
# 1) data/unprocessed contains the output of the program
# sampler, i.e. tokenized RASP programs.
# 2) sampler output (and maybe other data such as example
# programs) is deduped and stored in data/deduped.
# 3) the programs in deduped are then compiled. We store
# rasp tokens + weights in data/full.

unprocessed_dir = data_dir / "unprocessed"
deduped_dir = data_dir / "deduped"
full_dataset_dir = data_dir / "full"