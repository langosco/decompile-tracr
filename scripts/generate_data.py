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
from rasp_generator.utils import sample_test_input
from rasp_tokenizer import tokenizer
from rasp_tokenizer.compiling import COMPILER_BOS


rng = np.random.default_rng(0)


test_inputs = [sample_test_input(rng) for _ in range(100)]


@flax.struct.dataclass
class RaspFlatDatapoint:
    """Ready for training"""
    program: ArrayLike
    weights: ArrayLike


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


def sample():
    sampler = sampling.ProgramSampler(rng=rng)
    sampler.sample()
    model, tokens, params = tokenizer.compile_and_tokenize(sampler.program)
    return sampler.program, model, tokens, params


def to_flat_datapoints(tokens, params) -> list[RaspFlatDatapoint]:
    return [RaspFlatDatapoint(program=tok, weights=np.array(params[layer]))
            for layer, tok in tokens.items()]


results = []
for i in tqdm(range(1000)):
    try:
        program, model, tokens, params = sample()
    except (InvalidValueSetError, NoTokensError) as err:
        print(f"Failed to sample program: {err}.")

    is_invalid, reason = is_compiled_model_invalid(program, model)
    if is_invalid:
        print(f"Validation error at program {i}: {reason}")
        continue

    results.extend(to_flat_datapoints(tokens, params))


savepath = "data/dummy_data.pkl"
print(f"Saving generated programs to {savepath}.")
with open(savepath, "xb") as f:
    pickle.dump(results, f)