import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import numpy as np
import matplotlib.pyplot as plt
from decompile_tracr.tokenizing.str_to_rasp import split_list
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import config


# script currently broken after refactoring

# TODO:
# - histogram of ops

parser = argparse.ArgumentParser(description='Data processing.')
parser.add_argument('--name', type=str, default="train", 
                    help="'train', 'test', 'test_5, 'test_6")
parser.add_argument('--no_plot', action='store_true', help="Don't plot histograms")
parser.add_argument('--max_datapoints', type=int, default=None, 
                    help="Limit number of datapoints")
args = parser.parse_args()

# Data loading
data = data_utils.load_batches(
    config.full_dataset_dir, 
    max_data=args.max_datapoints
)
all_rasp = data_utils.load_batches(config.unprocessed_dir)
deduped_rasp = []
for d in os.scandir(config.deduped_dir):
    if d.is_dir():
        deduped_rasp.extend(data_utils.load_batches(d))


# Preprocessing data for histograms
n_sops = [x['n_sops'] for x in data]
n_params = [len([w for layer in x['weights'] for w in layer]) for x in data]
n_tokens = [len(x['tokens']) for x in data]
n_layers = [len(x['weights'])-1 for x in data]

n_params_per_layer = [len(layer) for x in data for layer in x['weights'][1:]]

# Statistics
n_data_deduped = len(deduped_rasp)
n_data_rasp = len(all_rasp)

print(f"Total datapoints pre deduping (unprocessed): {n_data_rasp:,}")
print(f"Total datapoints pre compilation (deduped): {n_data_deduped:,}")
print(f"Total datapoints (compiled): {len(data):,}")
print(f"Total layers (compiled): {sum(n_layers):,}")
print()
assert sum(n_layers) == len(n_params_per_layer)

print(f"Min weights per model: {np.min(n_params):,}")
print(f"Max weights per model: {np.max(n_params):,}")
print("Min weights per layer:", np.min(n_params_per_layer))
print("Max weights per layer:", np.max(n_params_per_layer))


if not args.no_plot:
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    axs = axs.flatten()

    axs[0].set_title("Program length (# SOps per program)")
    axs[0].hist(n_sops, bins=range(0, 15))
    axs[0].set_xticks(range(0, 15))

    axs[1].set_title("Layers per model")
    axs[1].hist(n_layers)

    axs[2].set_title("Parameters per model")
    axs[2].hist(n_params)

    axs[3].set_title("Tokens per model")
    axs[3].hist(n_tokens)

    axs[4].set_title("Parameters per layer")
    axs[4].hist(n_params_per_layer)

#    axs[5].set_title("Tokens per layer")
#    axs[5].hist(n_tokens_per_layer, bins=range(2, 20))


    plt.tight_layout()
    plt.show()


# token duplicates
