import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import dill

from decompile_tracr.sampling import sampling
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset import config 
from decompile_tracr.dataset.utils import sequential_count_via_lockfile


logger = setup_logger(__name__)


def save_batch(
        json_dataset: list,
        dill_dataset: list = None,
        savedir: Path = config.unprocessed_dir,
        keep_aux=False,
        overwrite=False,
        filename = None,
    ):
    """Save to a json and optionally a dill file."""
    logger.info(f"Saving {len(json_dataset)} generated "
                f"programs to {savedir}.")

    if filename is None:
        idx = sequential_count_via_lockfile(savedir / "count.txt")
        filename = f"data_{idx}"
    else:
        filename = Path(filename)
    
    open_mode = "w" if overwrite else "x"
    with open(savedir / f"{filename}.json", open_mode) as f:
        json.dump(json_dataset, f)
    
    if keep_aux:
        with open(savedir / f"{filename}.dill", open_mode + "b") as f:
            dill.dump(dill_dataset, f)


def sample_rasp(args):
    """Sample a program while catching and logging errors."""
    sampler = sampling.ProgramSampler(rng=rng)
    try:
        program, _ = sampler.sample(
            n_sops=args.n_sops, 
            min_length=args.min_length, 
            max_length=args.max_length
        )
    except sampling.SamplingError as e:
        logger.warning(f"Received sampling error: {e}.")
        program = sample_rasp(args)
    return program


def filter(by_layer: list[dict]):
    """Filter out bad programs."""
    return any(len(x) > config.MAX_RASP_LENGTH for x in by_layer)


def sample_loop(args, json_dataset, aux_dataset):
    """Modifies json_dataset and aux_dataset in-place."""
    for i in tqdm(range(args.ndata), disable=config.on_cluster):
        program = sample_rasp(args)
        tokens = tokenizer.tokenize(program)

        if filter(tokens):
            logger.warning(f"Skipping program {i} (too large).")
            continue

        json_dataset.append({
            "name": args.name,  # eg "train" or "test"
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })

        if args.keep_aux:
            aux_dataset.append({
                "rasp": program,  # rasp.SOp
            })


def parse_args():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--name', type=str, default="train")
    parser.add_argument('--n_sops', type=int, default=10, 
                        help='how many sops to sample per program.')
    parser.add_argument('--min_length', type=int, default=4, 
                        help='min nr of sops per program')
    parser.add_argument('--max_length', type=int, default=None, 
                        help='max nr of sops per program')
    parser.add_argument('--ndata', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--savedir', type=str, default=None,
                        help="override default save path (data/batches/...)")
    parser.add_argument('--keep_aux', action='store_true',
                        help="whether to include the rasp program "
                        "in the output data. rasp expressions are serialized "
                        "to a separate file using dill.")
    args = parser.parse_args()

    if args.savedir is None:
        args.savedir = config.unprocessed_dir
    else:
        args.savedir = Path(args.savedir)

    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # output data
    json_dataset = []
    aux_dataset = []

    try:
        sample_loop(args, json_dataset, aux_dataset)
    except KeyboardInterrupt:
        logger.info("Interrupted, saving dataset.")


    lengths = [x["n_sops"] for x in json_dataset]
    n_layers_per = [len(x["tokens"]) for x in json_dataset]

    logger.info(f"Generated {len(json_dataset)} programs.")
    logger.info(f"Min and max program length: {np.min(lengths)}, {np.max(lengths)}")
    logger.info(f"Average program length: {np.mean(lengths)}")
    logger.info(f"Average number of layers per program: {np.mean(n_layers_per)}")
    logger.info(f"Min and max number of layers per program: {np.min(n_layers_per)}, {np.max(n_layers_per)}")

    save_batch(json_dataset, aux_dataset, args.savedir, args.keep_aux)
