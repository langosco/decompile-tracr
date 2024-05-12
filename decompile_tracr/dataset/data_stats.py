import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from decompile_tracr.tokenizing.str_to_rasp import split_list
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import dataloading
from decompile_tracr.dataset import config


# script currently broken after refactoring

# TODO:
# - histogram of ops

parser = argparse.ArgumentParser(description='Data processing.')
parser.add_argument('--name', type=str, default="train", 
                    help="'train', 'test', 'test_5, 'test_6")
parser.add_argument('--no_plot', action='store_true', help="Don't plot histograms")
parser.add_argument('--max_datapoints', type=int, default=30_000, 
                    help="Limit number of datapoints")
args = parser.parse_args()


# Data loading
BS = 1000


def get_stats(x: dict):
    out = {}
    out['n_layers'] = x['n_layers']
    out['n_sops'] = x['n_sops']
    out['n_tokens'] = jnp.sum(x['tokens'] != vocab.pad_id, axis=-1)
    out['n_params'] = x['layer_idx'].max(axis=-1)
    return out


train_loader = dataloading.DataLoader(
    loadfile=config.data_dir / "full.h5",
    group="train",
    batch_size=BS,
    process_fn=get_stats,
    max_datapoints=args.max_datapoints,
)


stats = defaultdict(list)
for x in train_loader:
    for k, v in x.items():
        stats[k].extend(v.tolist())


print(f"Total training datapoints (compiled): {train_loader.ndata:,}")
# print(f"Total datapoints pre deduping (unprocessed): {n_data_rasp:,}")
# print(f"Total datapoints pre compilation (deduped): {n_data_deduped:,}")
#print(f"Total layers (compiled): {sum(n_layers):,}")

print(f"Min weights per model: {np.min(stats['n_params']):,}")
print(f"Max weights per model: {np.max(stats['n_params']):,}")


if not args.no_plot:
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 15))
    axs = axs.flatten()

    axs[0].set_title("Program length (# SOps per program)")
    axs[0].hist(stats['n_sops'], bins=range(0, 15))
    axs[0].set_xticks(range(0, 15))

    axs[1].set_title("Layers per model")
    axs[1].hist(stats['n_layers'])

    axs[2].set_title("Parameters per model")
    axs[2].hist(stats['n_params'])

    axs[3].set_title("Tokens per model")
    axs[3].hist(stats['n_tokens'])

    plt.tight_layout()
    plt.show()


# token duplicates
