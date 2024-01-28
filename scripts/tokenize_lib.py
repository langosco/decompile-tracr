# TODO: clean up and make more readable

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jaxtyping import ArrayLike
import pickle
import signal

from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel

from rasp_generator.utils import sample_test_input, print_program
from rasp_tokenizer import tokenizer
from rasp_tokenizer.compiling import COMPILER_BOS
from rasp_tokenizer.logger_config import setup_logger
from rasp_tokenizer import paths
from rasp_tokenizer import lib
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH


logger = setup_logger(__name__)
rng = np.random.default_rng(0)
test_inputs = [sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]


SAVEPATH = paths.data_dir / "test"
os.makedirs(SAVEPATH, exist_ok=True)


NUM_DATAPOINTS = 50
MAX_COMPILE_TIME = 5  # seconds


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
        return model, tokens, params
    except (InvalidValueSetError, NoTokensError) as err:
        logger.warning(f"Failed to compile program: {err}.")
        return None, None, None
    finally:
        signal.alarm(0)


def safe_compile_and_tokenize(program):
    try:
        model, tokens, params = try_compile(program)
    except Exception:  # catch everything else to print program
        logger.warning("\nUnkown exception during compilation.")
        logger.warning("Program:")
        print_program(program)
        print()
        save_to_file(dataset)
        raise
    return program, model, tokens, params


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
    max_rasp_len = max(len(x['rasp']) for x in by_layer)
    max_weights_len = max(len(x['weights']) for x in by_layer)
    if max_rasp_len > MAX_RASP_LENGTH:
        logger.warning(f"RASP length too long, {max_rasp_len} > {MAX_RASP_LENGTH} "
                       f"at program {by_layer[0]['program_id']}.")
        return True
    elif max_weights_len > MAX_WEIGHTS_LENGTH:
        logger.warning(f"Weights length too long, {max_weights_len} > {MAX_WEIGHTS_LENGTH} "
                       f"at program {by_layer[0]['program_id']}.")
        return True
    return False


def compile_loop(dataset):
    for program_id, program in lib.examples.items():
        program, model, tokens, params = safe_compile_and_tokenize(program)
        if model is None:
            continue

        is_invalid, reason = is_compiled_model_invalid(program, model)
        if is_invalid:
            logger.warning(f"Validation error at program {program_id}: {reason}")
            continue

        prog, _ = to_flat_datapoints(tokens, params, program_id=program_id)

        logger.info(f"Compiled and tokenized program {program_id}.")

        if filter(prog):
            raise ValueError(f"Program {program_id} doesn't pass filter.")
        else:
            dataset += prog


def save_to_file(dataset):
    savepath = SAVEPATH / f"reference_programs.pkl"
    logger.info(f"Saving generated programs to {savepath}.")
    with open(savepath, "xb") as f:
        pickle.dump(dataset, f)


dataset = []
compile_loop(dataset)
save_to_file(dataset)
