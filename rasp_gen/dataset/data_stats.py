import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
from collections import defaultdict
from collections import Counter
import h5py

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import seaborn as sns

from rasp_gen.tokenize import vocab
from rasp_gen.tokenize import tokenizer
from rasp_gen.dataset import dataloading
from rasp_gen.dataset.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Data processing.')
    parser.add_argument('--name', type=str, default="train", 
                        help="'train', 'test', 'test_5, 'test_6")
    parser.add_argument('--noplot', action='store_true', help="Don't plot histograms")
    parser.add_argument('--max_datapoints', type=int, default=30_000, 
                        help="Limit number of datapoints")
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()
    return args



def stats_that_require_loading_tokens(args):
    config = load_config(args.config)
    file = config.paths.dataset
    with h5py.File(file, 'r') as f:
        split = "train"
        n = f[split]['tokens'].shape[0]
        tokens = f[split]['tokens'][:]
    
    print(f"Total training datapoints: {n:,}")
    counts = _get_sop_counts(tokens)
    counts = {tokenizer.decode_token(k): v for k, v in counts.items()}

    # plot
    if not args.noplot:
        x = np.arange(len(counts))
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs = axs.flatten()
        ax = axs[0]
        ax.bar(x, counts.values())
        ax.set_xticks(x)
        ax.set_xticklabels(counts.keys(), rotation=25)
        ax.set_xlabel("SOp counts")
        ax.set_ylabel("Count")


        counts = _get_encoding_counts(tokens)
        x = np.arange(len(counts))
        ax = axs[1]
        ax.bar(x, counts.values())
        ax.set_xticks(x)
        ax.set_xticklabels(counts.keys())
        ax.set_xlabel("Encoding counts")
        ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    return None



def _get_sop_counts(tokens: np.ndarray[int]):
    sop_tokens = tokenizer.encode(vocab.ops)
    _min, _max = np.min(sop_tokens), np.max(sop_tokens)
    tokens_that_encode_sops = tokens[(tokens >= _min) & (tokens <= _max)]
    return Counter(tokens_that_encode_sops)


def _get_encoding_counts(tokens: np.ndarray[int]):
    cat, num = tokenizer.encode(vocab.encodings)
    n_cat, n_num = np.sum(tokens == cat), np.sum(tokens == num)
    return dict(categorical=n_cat, numerical=n_num)


def stats_that_require_loading_weights(args: argparse.Namespace):
    BS = 32


    def get_stats(x: dict):
        out = {}
        out['n_layers'] = x['n_layers']
        out['n_sops'] = x['n_sops']
        out['n_tokens'] = jnp.sum(x['tokens'] != vocab.pad_id, axis=-1)
        out['n_params'] = x['layer_idx'].max(axis=-1)
        return out


    config = load_config(args.config)
    train_loader = dataloading.DataLoader(
        loadfile=config.paths.dataset,
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
    print(f"Mean weights per model: {np.mean(stats['n_params']):,.2f}")


    if not args.noplot:
        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(6, 5))
        axs = axs.flatten()

        axs[0].set_xlabel("Program length (# SOps per program)")
        axs[0].hist(stats['n_sops'], bins=range(0, 15))
        axs[0].set_xticks(range(0, 15))

        axs[1].set_xlabel("Layers per model")
        axs[1].hist(stats['n_layers'])

        axs[2].set_xlabel("Parameters per model")
        axs[2].hist(stats['n_params'])

        axs[3].set_xlabel("Tokens per model")
        axs[3].hist(stats['n_tokens'])

        plt.tight_layout()
        plt.show()


    # token duplicates

if __name__ == "__main__":
    args = parse_args()
    stats_that_require_loading_tokens(args)
    stats_that_require_loading_weights(args)