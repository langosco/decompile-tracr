import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from tqdm import tqdm
import argparse
import psutil

from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.rasp import rasp

from decompile_tracr.sampling import sampling
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.data_utils import save_batch
from decompile_tracr.globals import disable_tqdm
from decompile_tracr.tokenizing import vocab


logger = setup_logger(__name__)
process = psutil.Process()
VERBOSE = False


def generate(
    rng: np.random.Generator, 
    config: DatasetConfig,
) -> list[dict]:
    logger.info("Begin sampling RASP programs.")
    data = sample_loop(rng, config)
    logger.info(f"Done sampling {len(data)} RASP programs.")
    save_batch(data, config.paths.programs_cache)
    return data


def sample_loop(rng, config: DatasetConfig):
    data = []
    for i in tqdm(range(config.ndata), disable=disable_tqdm, 
                  desc="Sampling"):
        program = sample_rasp(rng, config.program_length)
        try:
            tokens = tokenizer.tokenize(program)
        except (InvalidValueSetError, NoTokensError) as e:
            logger.warning(f"Skipping program {i} ({e}).")
            continue

        if to_filter(tokens, config=config):
            logger.warning(f"Skipping program {i} (too long).")
            continue

        data.append({
            "name": config.name,  # eg "train" or "test"
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })

        if i % 100 == 0 and VERBOSE:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({len(data)} programs).")
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
        return sample_rasp(rng, program_length)
    return program


def to_filter(tokens: list[int], config: DatasetConfig):
    """Returns True for programs that are too long."""
    program_too_long = len(tokens) > config.max_rasp_length
    too_many_layers = 1 + tokens.count(vocab.eol_id) > config.max_layers
    return program_too_long or too_many_layers
    

def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--program_length', type=int, default=10, 
                        help='program length (nr of sops)')
    parser.add_argument('--ndata', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None,
                        help="Override default data directory.")
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    config.ndata = args.ndata
    config.program_length = args.program_length
    config.name = args.name
    config.data_dir = args.dir

    data = generate(rng, config)

    lengths = [x["n_sops"] for x in data]
    n_tokens_per_program = [len(x["tokens"]) for x in data]

    logger.info(f"Min and max program length: {np.min(lengths)}, "
                f"{np.max(lengths)}")
    logger.info(f"Average program length: {np.mean(lengths)}")
    logger.info(f"Average number of tokens per program: "
                f"{np.mean(n_tokens_per_program)}")
    logger.info(f"Min and max number of tokens per program: "
                f"{np.min(n_tokens_per_program)}, "
                f"{np.max(n_tokens_per_program)}")
