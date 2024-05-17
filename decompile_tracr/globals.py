import os
import sys
from pathlib import Path


on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = Path("/rds/project/rds-eWkDxBhxBrQ")
module_path = Path(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
disable_tqdm = not interactive