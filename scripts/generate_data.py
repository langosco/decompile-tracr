# Quick test script to generate a small dataset for metamodel training.
# Output is a list of dicts. Each dict corresponds to a program.
# Each dict contains keys
# - 'weights_and_tokens': a list of dicts, one per layer. 
#       Used for training the metamodel.
# - 'model': a tracr AssembledTransformerModel, inclduing the weights
# - 'rasp': the original program
# 
# Output gets saved to paths.data_dir / "batches" / "data_{idx}.pkl"
# TODO: clean up and make more readable

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jaxtyping import ArrayLike
import dill as pickle
from tqdm import tqdm
import signal
import argparse

from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel

from rasp_generator import sampling
from rasp_generator.utils import sample_test_input, print_program
from rasp_tokenizer import tokenizer
from rasp_tokenizer.compiling import COMPILER_BOS
from rasp_tokenizer.logger_config import setup_logger
from rasp_tokenizer import on_cluster
from rasp_tokenizer import paths
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH
from rasp_tokenizer.utils import sequential_count_via_lockfile


parser = argparse.ArgumentParser(description='Training run')
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--n_sops', type=int, default=15, 
                    help='how many sops to sample per program.')
parser.add_argument('--min_length', type=int, default=4, 
                    help='min nr of sops per program')
parser.add_argument('--max_length', type=int, default=15, 
                    help='max nr of sops per program')
parser.add_argument('--ndata', type=int, default=50)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()


SAVEDIR = paths.data_dir / "batches"
if args.savedir is not None:
    SAVEDIR = SAVEDIR / args.savedir


os.makedirs(SAVEDIR, exist_ok=True)
logger = setup_logger(__name__)
rng = np.random.default_rng(args.seed)
test_inputs = [sample_test_input(rng) for _ in range(50)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]


NUM_DATAPOINTS = args.ndata
MAX_COMPILE_TIME = 5  # seconds


def save_to_file(dataset):
    idx = sequential_count_via_lockfile(SAVEDIR / "count.txt")
    savepath = SAVEDIR / f"data_{idx}.pkl"
    logger.info(f"Saving generated programs to {savepath}.")
    with open(savepath, "xb") as f:
        pickle.dump(dataset, f)


class CompilationTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise CompilationTimeout(
        f"Program took longer than {MAX_COMPILE_TIME} seconds to compile.")

signal.signal(signal.SIGALRM, timeout_handler)


def is_compiled_model_invalid(
        expr: rasp.SOp, 
        model: AssembledTransformerModel,
) -> (bool, str):
    for test_input in test_inputs:
        try:
            rasp_out = expr(test_input)
        except ValueError as err:
            reason = f"Program raised ValueError on test input: {err}."
            return True, reason

        rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
        model_out = model.apply([COMPILER_BOS] + test_input).decoded[1:]

        if not np.allclose(
            model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3):
            reason = (
                f"Compiled program {expr.label} does not match RASP output."
            )
            return True, reason
        else:
            return False, None


def try_compile(program: rasp.SOp):
    try:
        signal.alarm(MAX_COMPILE_TIME)
        model, tokens, params = tokenizer.compile_and_tokenize(program)
        return model, tokens, params
    except (InvalidValueSetError, NoTokensError, CompilationTimeout) as err:
        logger.warning(f"Failed to compile program: {err}.")
        return None, None, None
    finally:
        signal.alarm(0)


def sample_and_compile():
    sampler = sampling.ProgramSampler(rng=rng)
    try:
        program, _ = sampler.sample(
            n_sops=args.n_sops, 
            min_length=args.min_length, 
            max_length=args.max_length
        )
    except sampling.SamplingError as e:
        logger.warning(f"Received sampling error: {e}.")
        return None, None, None, None
    try:
        model, tokens, params = try_compile(program)
    except Exception:  # catch everything else to print program
        logger.warning("Unkown exception during compilation.")
        logger.warning("Program:")
        print_program(program)
        print()
        save_to_file(program_dataset)
        raise
    return program, model, tokens, params


def to_flat_datapoints(
        tokens: dict[str, list[int]],
        params: dict[str, ArrayLike],
    ) -> (list[dict], tuple):
    by_layer = [
        dict(
            rasp_tok=tuple(tok),
            weights=tuple(params[layer].tolist()),
        ) for layer, tok in tokens.items()
    ]
    # design decisions:
    # - should we store datapoints per layer or per program?
    # - is it bad if we lose program info? (it's bad for test data, but not necessarily for training data)
    # - should we deduplicate based on program or based on layer?
    # - should we deduplicated based on rasp code or rasp + weights? (if by layer)
    return by_layer, tuple(x['rasp_tok'] for x in by_layer)


def filter(by_layer: list[dict]):
    """Filter out bad programs."""
    if (
        any(len(x['rasp_tok']) > MAX_RASP_LENGTH for x in by_layer) or
        any(len(x['weights']) > MAX_WEIGHTS_LENGTH for x in by_layer)
    ):
        return True
    else:
        return False


def sample_loop(dataset, all_rasp):
    for i in tqdm(range(NUM_DATAPOINTS), disable=on_cluster):
        program, model, tokens, params = sample_and_compile()
        if model is None:
            continue

        is_invalid, reason = is_compiled_model_invalid(program, model)
        if is_invalid:
            logger.warning(f"Validation error at program {i}: {reason}")
            continue

        by_layer, rasp_str = to_flat_datapoints(tokens, params)
        if rasp_str in all_rasp:
            # This doesn't deduplicate on the layer level, 
            # just on the program level.
            # It also doesn't dedupe between batches, which is why 
            # scripts/dedupe.py is necessary.
            logger.warning(f"Duplicate program {i} filtered.")
            continue
        elif filter(by_layer):
            logger.warning(f"Filtered out program {i} (too large).")
            continue
        else:
            all_rasp.add(rasp_str)
            datapoint = {
                "weights_and_tokens": by_layer,  # list of dicts
                "model": model,  # AssembledTransformerModel
                "rasp": program,  # rasp.SOp
            }
            dataset.append(datapoint)
            lengths.append(program.annotations['length'])


all_rasp = set()
lengths = []
program_dataset = []

try:
    sample_loop(program_dataset, all_rasp)
except KeyboardInterrupt:
    logger.info("Interrupted, saving dataset.")


n_layers_per = [len(x) for x in program_dataset]

logger.info(f"Generated {len(lengths)} programs.")
logger.info(f"Min and max program length: {np.min(lengths)}, {np.max(lengths)}")
logger.info(f"Average program length: {np.mean(lengths)}")
logger.info(f"Average number of layers per program: {np.mean(n_layers_per)}")
logger.info(f"Min and max number of layers per program: {np.min(n_layers_per)}, {np.max(n_layers_per)}")

save_to_file(program_dataset)
