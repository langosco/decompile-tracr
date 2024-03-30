import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import psutil

from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.rasp import rasp

from decompile_tracr.sampling import sampling
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset import config 
from decompile_tracr.dataset.data_utils import save_batch


logger = setup_logger(__name__)
process = psutil.Process()
VERBOSE = False


def generate(
    rng: np.random.Generator, 
    ndata: int, 
    name: str, 
    savedir: Path = config.unprocessed_dir,
    program_length: int = 10,
) -> list[dict]:
    logger.info("Begin sampling RASP programs.")
    data = sample_loop(rng, ndata, name, program_length)
    logger.info(f"Done sampling {len(data)} RASP programs.")
    save_batch(data, savedir)
    return data


def sample_rasp(
        rng: np.random.Generator,
        program_length: int,
) -> rasp.SOp:
    """Sample a program while catching and logging errors."""
    try:
        program = sampling.sample(rng, program_length)
    except sampling.SamplingError as e:
        logger.warning(f"Received sampling error: {e}.")
        program = sample_rasp(rng, program_length)
    return program


def filter(by_layer: list[dict]):
    """Filter out bad programs."""
    return any(len(x) > config.MAX_RASP_LENGTH for x in by_layer)


def sample_loop(rng, ndata, name: str, program_length: int):
    data = []
    for i in tqdm(range(ndata), disable=config.global_disable_tqdm, 
                  desc="Sampling"):
        program = sample_rasp(rng, program_length)
        try:
            tokens = tokenizer.tokenize(program)
        except (InvalidValueSetError, NoTokensError) as e:
            logger.warning(f"Skipping program {i} ({e}).")

        if filter(tokens):
            logger.warning(f"Skipping program {i} (too large).")
            continue

        data.append({
            "name": name,  # eg "train" or "test"
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })

        if i % 100 == 0 and VERBOSE:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({len(data)} programs).")
    return data
    

def parse_args():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--name', type=str, default="train")
    parser.add_argument('--program_length', type=int, default=10, 
                        help='program length (nr of sops)')
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

    data = generate(rng, args.ndata, args.name, args.savedir, args.program_length)

    lengths = [x["n_sops"] for x in data]
    n_layers_per = [len(x["tokens"]) for x in data]

    logger.info(f"Generated {len(data)} programs.")
    logger.info(f"Min and max program length: {np.min(lengths)}, {np.max(lengths)}")
    logger.info(f"Average program length: {np.mean(lengths)}")
    logger.info(f"Average number of layers per program: {np.mean(n_layers_per)}")
    logger.info(f"Min and max number of layers per program: {np.min(n_layers_per)}, {np.max(n_layers_per)}")
