# Quick test script to generate a 'dummy' dataset for metamodel training.

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jaxtyping import ArrayLike
import flax
import pickle
from tqdm import tqdm
import signal

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
from rasp_tokenizer.utils import sequential_count_via_lockfile


logger = setup_logger(__name__)
rng = np.random.default_rng(0)
test_inputs = [sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]

SAVEPATH = paths.data_dir / "train"
os.makedirs(SAVEPATH, exist_ok=True)


# per layer maximums:
MAX_RASP_LENGTH = 128
MAX_WEIGHTS_LENGTH = 8192

NUM_DATAPOINTS = 50
MAX_COMPILE_TIME = 5  # seconds


class CompilationTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise CompilationTimeout(
        f"Program took longer than {MAX_COMPILE_TIME} seconds to compile.")

signal.signal(signal.SIGALRM, timeout_handler)


def is_compiled_model_invalid(
        expr: rasp.SOp, 
        model: AssembledTransformerModel,
    ):
    for test_input in test_inputs:
        rasp_out = expr(test_input)
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
        signal.alarm(0)
        return model, tokens, params
    except (InvalidValueSetError, NoTokensError, CompilationTimeout) as err:
        logger.warning(f"Failed to compile program: {err}.")
        return None, None, None


def sample_and_compile():
    sampler = sampling.ProgramSampler(rng=rng)
    sampler.sample(target_length=15)
    try:
        model, tokens, params = try_compile(sampler.program)
    except Exception:  # catch everything else to print program
        logger.warning("\nUnkown exception during compilation.")
        logger.warning("Program:")
        print_program(sampler.program)
        print()
        save_to_file(dataset)
        raise
    return sampler.program, model, tokens, params


def to_flat_datapoints(
        tokens: dict[str, list[int]],
        params: dict[str, ArrayLike],
        program_id: int,
    ) -> (list[dict], tuple):
    by_layer = [
        dict(
            rasp=tuple(tok),
            weights=tuple(params[layer].tolist()),
            program_id=program_id,
        ) for layer, tok in tokens.items()
    ]
    # design decisions:
    # - should we store datapoints per layer or per program?
    # - is it bad if we lose program info? (it's bad for test data, but not necessarily for training data)
    # - should we deduplicate based on program or based on layer?
    # - should we deduplicated based on rasp code or rasp + weights? (if by layer)
    return by_layer, tuple(x['rasp'] for x in by_layer)


def filter(by_layer: list[dict]):
    """Filter out bad programs."""
    if (
        any(len(x['rasp']) > MAX_RASP_LENGTH for x in by_layer) or
        any(len(x['weights']) > MAX_WEIGHTS_LENGTH for x in by_layer)
    ):
        return True
    else:
        return False


def sample_loop(dataset, all_rasp):
    for i in tqdm(range(NUM_DATAPOINTS)):
        program, model, tokens, params = sample_and_compile()
        if model is None:
            continue

        is_invalid, reason = is_compiled_model_invalid(program, model)
        if is_invalid:
            logger.warning(f"Validation error at program {i}: {reason}")
            continue

        program_id = sequential_count_via_lockfile(SAVEPATH / "count.txt")
        prog, rasp_str = to_flat_datapoints(tokens, params, program_id=program_id)
        if rasp_str in all_rasp:
            # dedupe programs
            # note this doesn't deduplicate on the layer level
            continue
        elif filter(prog):
            continue
        else:
            all_rasp.add(rasp_str)
            dataset += prog


all_rasp = set()
dataset = []

try:
    sample_loop(dataset, all_rasp)
except KeyboardInterrupt:
    logger.info("Interrupted, saving dataset.")


def save_to_file(dataset):
    idx = sequential_count_via_lockfile(SAVEPATH / "count.txt")
    savepath = SAVEPATH / f"data_{idx}.pkl"
    logger.info(f"Saving generated programs to {savepath}.")
    with open(savepath, "xb") as f:
        pickle.dump(dataset, f)


save_to_file(dataset)