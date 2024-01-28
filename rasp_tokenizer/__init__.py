import os
import sys
from pathlib import Path

# Set up global variables
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = "/rds/project/rds-eWkDxBhxBrQ"

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))



# Set up global constants
# per layer maximums size of RASP program and weights:
# MAX_RASP_LENGTH = 128
# MAX_WEIGHTS_LENGTH = 16_384

MAX_RASP_LENGTH = 32
MAX_WEIGHTS_LENGTH = 8192


class Paths:
    def __init__(self):
        self.set_default_paths()

    def set_default_paths(self):
        if on_cluster:
            data_dir = os.path.join(hpc_storage_dir, "lauro/rasp-data/")
        else:
            data_dir = os.path.join(module_path, "scripts/hpc_data/")

        self.module_path = Path(module_path)
        self.data_dir = Path(data_dir)


paths = Paths()
