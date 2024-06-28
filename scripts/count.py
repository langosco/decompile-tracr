import argparse
import h5py
import jax
from decompile_tracr.dataset.config import load_config


parser = argparse.ArgumentParser(description='Sample RASP programs.')
parser.add_argument('--config', type=str, default=None,
                    help="Name of config file.")
args = parser.parse_args()
config = load_config(args.config)

if not config.paths.dataset.exists():
    raise FileNotFoundError(f"Dataset {config.paths.dataset} not found")

with h5py.File(config.paths.dataset, 'r') as f:
    for k, v in f.items():
        print(f"Number of datapoints in {k}: {len(v['tokens']):,}")
#    print()
#    print("Dataset shapes:")
#    for k, v in f.items():
#        for kk, vv in v.items():
#            print(f"    {k}/{kk}: {vv.shape}")

