import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import numpy as np
import matplotlib.pyplot as plt
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import config


# script currently broken after refactoring

# TODO:
# - histogram of ops

parser = argparse.ArgumentParser(description='Data processing.')
parser.add_argument('--name', type=str, default="train", 
                    help="'train', 'test', 'test_5, 'test_6")
args = parser.parse_args()

# Data loading
data = data_utils.load_batches(config.data_dir / "full")
print("Total datapoints:", len(data))


# Preprocessing data for histograms
n_sops = [x['n_sops'] for x in data]
n_params = [len([w for layer in x['weights'] for w in layer]) for x in data]
n_tokens = [len([t for layer in x['tokens'] for t in layer]) for x in data]
n_layers = [len(x['weights']) for x in data]

n_params_per_layer = [len(layer) for x in data for layer in x['weights']]
n_tokens_per_layer = [len(layer) for x in data for layer in x['tokens']]

print("Total layers:", sum(n_layers))
assert sum(n_layers) == len(n_params_per_layer)
assert sum(n_layers) == len(n_tokens_per_layer)


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

axs[5].set_title("Tokens per layer")
axs[5].hist(n_tokens_per_layer, bins=range(2, 20))


plt.tight_layout()
plt.show()


# Statistics
print("Min weights per model:", np.min(n_params))
print("Max weights per model:", np.max(n_params))
print("Min weights per layer", np.min(n_params_per_layer))
print("Max weights per layer", np.max(n_params_per_layer))
