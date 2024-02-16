import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import numpy as np
import matplotlib.pyplot as plt
from decompile_tracr.dataset.data_utils import load_deduped


# script currently broken after refactoring


parser = argparse.ArgumentParser(description='Data processing.')
parser.add_argument('--name', type=str, default="train", 
                    help="'train', 'test', 'test_5, 'test_6")
args = parser.parse_args()

# Data loading
data = load_deduped(name=args.name, flatten=False)
print("Total datapoints:", len(data))


# Preprocessing data for histograms
n_sops = [x['n_sops'] for x in data]
weights_per_model = [
    sum(len(layer['weights']) for layer in d['weights_and_tokens']
    ) for d in data
]
layer_lengths = [len(x['weights_and_tokens']) for x in data]
data_per_layer = [l for x in data for l in x['weights_and_tokens']]
token_lens = [len(x['rasp_tok']) for x in data_per_layer]
weight_lens = [len(x['weights']) for x in data_per_layer]


# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 15))
axs = axs.flatten()

axs[0].set_title("Distribution of program lengths")
axs[0].hist(n_sops, bins=range(0, 15))
axs[0].set_xticks(range(0, 15))

axs[1].set_title("Distribution of # parameters per model")
axs[1].hist(weights_per_model)

axs[2].set_title("Number of layers per model")
axs[2].hist(layer_lengths)

axs[3].set_title("Distribution of tokens per layer")
axs[3].hist(token_lens, bins=range(2, 20))

axs[4].set_title("Distribution of weight lengths per layer")
axs[4].hist(weight_lens)

plt.tight_layout()
plt.show()


# Statistics
print("Min weights per model:", np.min(weights_per_model))
print("Max weights per model:", np.max(weights_per_model))
print("Min weights per layer", np.min(weight_lens))
print("Max weights per layer", np.max(weight_lens))
