import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import numpy as np
import chex

from tracr.rasp import rasp
from tracr.compiler.compiling import compile_rasp_to_model
from tracr.compiler import assemble

from decompile_tracr.sampling import rasp_utils
from decompile_tracr.sampling import sampling
from decompile_tracr.sampling.validate import is_valid
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import lib
from decompile_tracr.dataset import compile

rng = np.random.default_rng(None)


TEST_INPUTS = [rasp_utils.sample_test_input(rng) for _ in range(100)]
LENGTH = 8
PROGRAMS = [sampling.sample(rng, program_length=LENGTH) for _ in range(30)]


@pytest.mark.parametrize("program", PROGRAMS)
def test_compile(program: rasp.SOp):
    assert rasp_utils.count_sops(program) == LENGTH, f"Program {program.label} has unexpected length."

    model = _compile(program)
    assert _validate_compiled(program, model), (
        f"Compiled program {program.label} does not "
         "match RASP output."
    )


    # Compare compile(program) to compile(detokenize(tokenize(program)))
    tokens = tokenizer.tokenize(program)
    reconstructed_program = tokenizer.detokenize(tokens)
    reconstructed_model = _compile(reconstructed_program)

    chex.assert_trees_all_close(model.params, reconstructed_model.params, 
                                rtol=1e-3, atol=1e-3)


def _validate_compiled(program: rasp.SOp, model: assemble.AssembledTransformerModel):
    return all(_is_close(program, model, x) for x in TEST_INPUTS)


def _is_close(program: rasp.SOp, model: assemble.AssembledTransformerModel, x: list):
    """Compare outputs on a single test input x."""
    # first make sure program passes initial screening:
    assert is_valid(program, x), f"Program {program.label} is not valid for input {x}"

    # then compare outputs of compiled model vs. RASP program:
    rasp_out = program(x)
    rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
    assert isinstance(x, list), f"Expected list, got {type(x)}"
    model_out = model.apply(["compiler_bos"] + x).decoded[1:]
    return np.allclose(model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3)


def _compile(program):
    return compile_rasp_to_model(
        program,
        vocab=set(range(5)),
        max_seq_len=5,
    )