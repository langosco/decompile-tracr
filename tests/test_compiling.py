import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu"
import pytest
import time
import numpy as np
import chex

from tracr.rasp import rasp
from tracr.compiler.compiling import compile_rasp_to_model
from tracr.compiler import assemble
from tracr.compiler import validating
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler.craft_model_to_transformer import NoTokensError

from rasp_gen.sample import rasp_utils
from rasp_gen.sample import sample
from rasp_gen.tokenize import tokenizer

rng = np.random.default_rng(None)


INPUTS = [rasp_utils.sample_test_input(rng) for _ in range(100)]
LENGTH = 8
PROGRAMS = [sample.sample(rng, program_length=LENGTH) for _ in range(50)]


@pytest.fixture(scope="module")
def data():
    programs = [_retokenize(p) for p in PROGRAMS]
    t = time.time()
    compiled = [_compile(p) for p in programs]
    compile_time = time.time() - t
    num_invalid = sum([c is None for c in compiled])

    compiled, programs = zip(
        *[(c, p) for c, p in zip(compiled, programs) 
          if c is not None and p is not None])

    return dict(
        programs=programs,
        compiled=compiled,
        compile_time=compile_time,
        num_invalid=num_invalid
    )


def test_timing(data):
    MAX = 5
    time_per_program = data['compile_time'] / len(data['programs'])
    assert time_per_program < MAX, (
        f"Compiling took {time_per_program:.3f} seconds per program. "
        f"Want less than {MAX} seconds.")


def test_invalid_programs(data):
    frac_invalid = data['num_invalid'] / len(PROGRAMS)
    assert frac_invalid < 0.05, (
        f"{data['num_invalid']}/{len(PROGRAMS)} programs failed to compile. "
        f"Want this to happen for less than 5% of programs.")

    
def test_dynamic_validate(data):
    """Before compiling, make sure that programs pass dynamic validation.
    """
    ps = data['programs']
    invalid = [
        _is_invalid_according_to_dynamic_validator(p) for p in ps
    ]

    assert not any(invalid), (
        f"{sum(invalid)}/{len(ps)} programs failed dynamic validation.")


def test_outputs_equal(data):
    """Check that the compiled program has 
    the same output as the original.
    """
    valid = [
        _is_close(p, model=m) for p, m in zip(data['programs'], data['compiled'])
    ]
    invalid_frac = 1 - np.mean(valid)
    invalid = sum([not v for v in valid])
    assert invalid_frac < 0.05, (
        f"{invalid}/{len(data['programs'])} compiled models fail "
        "to match the output of the original program."
    )


def test_recompile(data):
    """Compile program again and make sure the result is the same."""
    programs, compiled = data['programs'], data['compiled']
    new_compiled = [_compile(p) for p in programs]
    not_equal = 0
    for m1, m2 in zip(compiled, new_compiled):
        try:
            chex.assert_trees_all_close(m1.params, m2.params, rtol=1e-3, atol=1e-3)
        except AssertionError as e:
            not_equal += 1
    assert not_equal == 0, (
        f"{not_equal}/{len(programs)} programs produce different models "
        "when compiled a second time."
    )


def test_retokenize_and_recompile(data):
    """Compare compile(program) to compile(detokenize(tokenize(program)))"""
    programs, compiled = data['programs'], data['compiled']
    new_compiled = [_retokenize_and_compile(p) for p in programs]
    not_equal = 0
    for m1, m2 in zip(compiled, new_compiled):
        try:
            chex.assert_trees_all_close(m1.params, m2.params, rtol=1e-3, atol=1e-3)
        except AssertionError as e:
            not_equal += 1
    assert not_equal == 0, (
        f"{not_equal}/{len(programs)} programs produce different models "
        "when retokenized and compiled a second time."
    )


def _retokenize_and_compile(program: rasp.SOp):
    program = tokenizer.detokenize(tokenizer.tokenize(program))
    return _compile(program)


def _is_close_single(
        program: rasp.SOp, model: assemble.AssembledTransformerModel, x: list):
    """Compare outputs on a single test input x."""
    rasp_out = program(x)
    rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
    assert isinstance(x, list), f"Expected list, got {type(x)}"
    model_out = model.apply(["compiler_bos"] + x).decoded[1:]
    return np.allclose(model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3)


def _is_close(program, model):
    return all(_is_close_single(program, model, x) for x in INPUTS)


def _compile(program):
    if program is None:
        return None
    assert isinstance(program, rasp.SOp)
    try:
        return compile_rasp_to_model(
            program,
            vocab=set(range(5)),
            max_seq_len=5,
        )
    except (InvalidValueSetError, NoTokensError) as e:
        return None


def _is_invalid_according_to_dynamic_validator(program) -> bool:
    """Check if program passes dynamic validation."""
    return any(len(validating.validate(program, x)) > 0 for x in INPUTS)


def _retokenize(p: rasp.SOp) -> rasp.SOp:
    assert isinstance(p, rasp.SOp)
    try:
        return tokenizer.detokenize(tokenizer.tokenize(p))
    except (InvalidValueSetError, NoTokensError):
        return None