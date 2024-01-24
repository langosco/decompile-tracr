# Quick test script to generate a 'dummy' dataset for metamodel training.

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jaxtyping import ArrayLike
import flax
import pickle
from tqdm import tqdm

from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel

from rasp_generator import sampling
from rasp_generator.utils import sample_test_input, print_program
from rasp_tokenizer.utils import RaspFlatDatapoint
from rasp_tokenizer import tokenizer
from rasp_tokenizer.compiling import COMPILER_BOS


rng = np.random.default_rng(0)
test_inputs = [sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]


# per layer maximums:
MAX_PROGRAM_LENGTH = 30
MAX_WEIGHTS_LENGTH = 5000

NUM_DATAPOINTS = 100


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
        model, tokens, params = tokenizer.compile_and_tokenize(program)
        return model, tokens, params
    except (InvalidValueSetError, NoTokensError) as err:
        print(f"Failed to compile program: {err}.")
        return None, None, None


def sample_and_compile():
    sampler = sampling.ProgramSampler(rng=rng)
    sampler.sample()
    try:
        model, tokens, params = try_compile(sampler.program)
    except Exception:  # catch everything else to print program
        print("\nUnkown exception during compilation.")
        print("Program:")
        print_program(sampler.program)
        print()
        raise
    return sampler.program, model, tokens, params



def to_flat_datapoints(tokens, params) -> list[RaspFlatDatapoint]:
    by_layer = [
        RaspFlatDatapoint(program=tok, weights=np.array(params[layer])) 
        for layer, tok in tokens.items()
        ]
    by_layer = [x for x in by_layer if len(x.program) > 0]
    by_layer = [x for x in by_layer if len(x.program) < MAX_PROGRAM_LENGTH]
    by_layer = [x for x in by_layer if len(x.weights) < MAX_WEIGHTS_LENGTH]
    return by_layer


results = []
for i in tqdm(range(NUM_DATAPOINTS)):
    program, model, tokens, params = sample_and_compile()
    if model is None:
        continue

    is_invalid, reason = is_compiled_model_invalid(program, model)
    if is_invalid:
        print(f"Validation error at program {i}: {reason}")
        continue

    results.extend(to_flat_datapoints(tokens, params))


savepath = "data/dummy_data.pkl"
print(f"Saving generated programs to {savepath}.")
with open(savepath, "xb") as f:
    pickle.dump(results, f)