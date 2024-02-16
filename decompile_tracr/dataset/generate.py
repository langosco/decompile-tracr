import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from decompile_tracr.sampling import sampling
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset import config 
from decompile_tracr.dataset.data_utils import save_batch


logger = setup_logger(__name__)


def generate(
    rng: np.random.Generator, 
    ndata: int, 
    name: str, 
    savedir: Path = config.unprocessed_dir,
    **sampler_kwargs
) -> list[dict]:
    data = sample_loop(rng, ndata, name, **sampler_kwargs)
    save_batch(data, savedir)
    return data


def sample_rasp(
        rng: np.random.Generator,
        **sampler_kwargs,
):
    """Sample a program while catching and logging errors."""
    sampler = sampling.ProgramSampler(rng=rng)
    try:
        program, _ = sampler.sample(**sampler_kwargs)
    except sampling.SamplingError as e:
        logger.warning(f"Received sampling error: {e}.")
        program = sample_rasp(rng, **sampler_kwargs)
    return program


def filter(by_layer: list[dict]):
    """Filter out bad programs."""
    return any(len(x) > config.MAX_RASP_LENGTH for x in by_layer)


def sample_loop(rng, ndata, name: str, **sampler_kwargs):
    data = []
    for i in tqdm(range(ndata), disable=config.global_disable_tqdm):
        program = sample_rasp(rng, **sampler_kwargs)
        tokens = tokenizer.tokenize(program)

        if filter(tokens):
            logger.warning(f"Skipping program {i} (too large).")
            continue

        data.append({
            "name": name,  # eg "train" or "test"
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })
    return data
    

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
    args = parser.parse_args()

    if args.savedir is None:
        args.savedir = config.unprocessed_dir
    else:
        args.savedir = Path(args.savedir)

    return args


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sampler_kwargs = {
        "n_sops": args.n_sops, 
        "min_length": args.min_length, 
        "max_length": args.max_length
    }

    data = generate(rng, args.ndata, args.name, args.savedir, 
                             **sampler_kwargs)

    lengths = [x["n_sops"] for x in data]
    n_layers_per = [len(x["tokens"]) for x in data]

    logger.info(f"Generated {len(data)} programs.")
    logger.info(f"Min and max program length: {np.min(lengths)}, {np.max(lengths)}")
    logger.info(f"Average program length: {np.mean(lengths)}")
    logger.info(f"Average number of layers per program: {np.mean(n_layers_per)}")
    logger.info(f"Min and max number of layers per program: {np.min(n_layers_per)}, {np.max(n_layers_per)}")
