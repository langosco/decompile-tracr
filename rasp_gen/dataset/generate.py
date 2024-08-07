import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
from tqdm import tqdm
import argparse

from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.rasp import rasp

from rasp_gen.sample import sample
from rasp_gen.tokenize import tokenizer
from rasp_gen.dataset.logger_config import setup_logger
from rasp_gen.dataset.config import DatasetConfig, load_config
from rasp_gen.dataset.data_utils import save_json
from rasp_gen.dataset import data_utils
from rasp_gen.dataset import Signals
from rasp_gen.globals import disable_tqdm
from rasp_gen.tokenize import vocab
from rasp_gen.tokenize.str_to_rasp import split_list


logger = setup_logger(__name__)
VERBOSE = False


def generate_batches(
    rng: np.random.Generator,
    config: DatasetConfig,
    ndata: int = 100,
    disable_tqdm: bool = False,
):
    logger.info("Begin sampling RASP programs.")
    bs = min(ndata, 200)
    nbatches = np.ceil(ndata / bs).astype(int)
    for i in range(nbatches):
        if i == nbatches - 1:
            bs = ndata - i * bs
        generate_batch(rng, bs, config=config, disable_tqdm=disable_tqdm)
        if Signals.sigterm:
            break


def generate_batch(
    rng, 
    batch_size: int,
    config: DatasetConfig, 
    disable_tqdm: bool = False,
) -> list[dict]:
    data = []
    for i in tqdm(range(batch_size), disable=disable_tqdm, desc="Sampling"):
        program = sample_rasp(rng, config.program_length)
        try:
            tokens = tokenizer.tokenize(program)
        except (InvalidValueSetError, NoTokensError) as e:
            logger.warning(f"Skipping program {i} ({e}).")
            continue

        if to_filter(tokens, config=config):
            logger.warning(f"Skipping program {i} (too long).")
            continue

        tokens = data_utils.pad_to(
            np.array(tokens),
            config.max_rasp_length, 
            pad_value=vocab.pad_id,
        ).tolist()

        data.append({
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
            "n_layers": tokens.count(vocab.eol_id),
        })

    if not Signals.n_sigterms >= 2:  # avoid saving after 2nd sigterm
        save_json(rng=rng, data=data, savedir=config.paths.programs_cache)

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
    parser.add_argument('--ndata', type=int, default=100)
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

    generate_batches(rng, config, ndata=args.ndata, 
                     disable_tqdm=args.disable_tqdm)