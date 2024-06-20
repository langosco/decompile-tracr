import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from tqdm import tqdm
import argparse
import psutil

from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.rasp import rasp

from decompile_tracr.sample import sample
from decompile_tracr.tokenize import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset.config import DatasetConfig, load_config
from decompile_tracr.dataset.data_utils import save_json
from decompile_tracr.globals import disable_tqdm
from decompile_tracr.tokenize import vocab
from decompile_tracr.tokenize.str_to_rasp import split_list


logger = setup_logger(__name__)
process = psutil.Process()
VERBOSE = False


def generate(
    rng: np.random.Generator, 
    config: DatasetConfig,
    disable_tqdm: bool = False,
) -> list[dict]:
    logger.info("Begin sampling RASP programs.")
    data = sample_loop(rng, config, disable_tqdm=disable_tqdm)
    logger.info(f"Done sampling {len(data)} RASP programs.")
    save_json(rng=rng, data=data, savedir=config.paths.programs_cache)
    return data


def sample_loop(rng, config: DatasetConfig, disable_tqdm=False):
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
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
            "n_layers": tokens.count(vocab.eol_id),
        })

        if i % 100 == 0 and VERBOSE:
            mem_info = process.memory_full_info()
            logger.info(f"Memory usage: {mem_info.uss / 1024**2:.2f} "
                        f"MB ({len(data)} programs).")
    return data


def sample_rasp(
    rng: np.random.Generator,
    program_length: int | list[int],
) -> rasp.SOp:
    """Sample a program while catching and logging errors."""
    try:
        program_length = int(program_length)
    except TypeError:
        program_length = rng.choice(list(program_length))

    try:
        program = sample.sample(rng, program_length)
    except sample.SamplingError as e:
        logger.warning(f"Received sampling error: {e}.")
        return sample_rasp(rng, program_length)
    return program


def to_filter(tokens: list[int], config: DatasetConfig):
    """Returns True for programs that are too long."""
    program_too_long = len(tokens) > config.max_rasp_length
    too_many_layers = 1 + tokens.count(vocab.eol_id) > config.max_layers
    too_many_ops_in_layer = config.compress and multiple_ops_in_one_layer(tokens)
    return program_too_long or too_many_layers or too_many_ops_in_layer


def multiple_ops_in_one_layer(tokens: list[int]):
    return any(n > 1 for n in ops_per_layer(tokens))


def ops_per_layer(tokens: list[int]):
    layers = split_list(tokens, vocab.eol_id)
    return [l.count(vocab.eoo_id) for l in layers]

    

def parse_args():
    parser = argparse.ArgumentParser(description='Sample RASP programs.')
    parser.add_argument('--ndata', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--config', type=str, default=None,
                        help="Name of config file.")
    args = parser.parse_args()
    if disable_tqdm:
        args.disable_tqdm = True
    return args


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    if args.ndata is not None:
        config.ndata = args.ndata

    data = generate(rng, config, disable_tqdm=args.disable_tqdm)

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
