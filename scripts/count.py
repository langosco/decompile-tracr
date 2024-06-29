import argparse
import h5py
import jax
from decompile_tracr.dataset.config import load_config


parser = argparse.ArgumentParser(description='Sample RASP programs.')
parser.add_argument('--config', type=str, default=None,
                    help="Name of config file.")
args = parser.parse_args()
config = load_config(args.config)
dataset = config.paths.dataset
programs = config.paths.programs

if dataset.exists() and False:
    with h5py.File(dataset, 'r') as f:
        for k, v in f.items():
            print(f"Number of datapoints in {k}: {len(v['tokens']):,}")
else:
    if not programs.exists():
        raise FileNotFoundError(f"Dataset {dataset} not found")
    else:
        print(f"Dataset {dataset} not found. Counting programs "
              f"in {programs} instead.")
    with h5py.File(programs, 'r') as f:
        print(f"Number of datapoints: {len(f['tokens']):,}")