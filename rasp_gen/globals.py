import os
import sys
from pathlib import Path


on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = Path("/rds/project/rds-eWkDxBhxBrQ")
module_path = Path(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
try:
    in_foreground = os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())
except OSError:
    in_foreground = True
disable_tqdm = not interactive or not in_foreground