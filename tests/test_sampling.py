import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import time
import numpy as np
from collections import Counter
from jaxtyping import ArrayLike
import chex

from tracr.rasp import rasp
from tracr.compiler import validating

from decompile_tracr.sample import rasp_utils
from decompile_tracr.sample.rasp_utils import SamplingError
from decompile_tracr.sample import sample
from decompile_tracr.sample.validate import perform_checks
from decompile_tracr.tokenize import tokenizer
from decompile_tracr.tokenize import vocab
from decompile_tracr.dataset import lib

rng = np.random.default_rng(None)

LENGTH = 10
INPUTS = [rasp_utils.sample_test_input(rng, max_seq_len=5, min_seq_len=5) 
               for _ in range(1000)]

t = time.time()
PROGRAMS = [sample.sample(rng, program_length=LENGTH) for _ in range(100)]
total_time = time.time() - t
OUTPUTS_UNSANITIZED = [[p(x) for x in INPUTS] for p in PROGRAMS]  # can sometimes throw ValueError('key/tokens is None')
OUTPUTS_UNSANITIZED = np.array(OUTPUTS_UNSANITIZED, dtype=float)
OUTPUTS = np.nan_to_num(OUTPUTS_UNSANITIZED, nan=0)  # (programs, inputs, seq)


def test_timing():
    time_per_program = total_time / len(PROGRAMS)
    assert time_per_program < 0.5, (
        f"Sampling took {time_per_program:.3f} seconds per program.")


def test_length():
    for p in PROGRAMS:
        assert rasp_utils.count_sops(p) == LENGTH


def test_static_validate():
    errs = [validating.validate(p) for p in PROGRAMS]
    num_invalid = sum([len(e) > 0 for e in errs])
    assert num_invalid == 0, (
        f"{num_invalid}/{len(PROGRAMS)} programs "
        f"failed static validation.")


def test_dynamic_validate():
    invalid = [_is_invalid_according_to_dynamic_validator(p, INPUTS) 
               for p in PROGRAMS]
    num_invalid = sum(invalid)
    assert num_invalid == 0, (
        f"{num_invalid}/{len(PROGRAMS)} programs "
        f"failed dynamic validation.")


def test_constant_wrt_input():
    """Test that sampled programs are not constant wrt input."""
    are_constant = [_mostly_constant_wrt_input(o) for o in OUTPUTS]
    frac_constant = sum(are_constant) / len(OUTPUTS)
    assert frac_constant < 0.05, (
        f"{frac_constant*100}% of programs produce exactly "
        f"the same output for > 80% of test inputs."
    )


def test_low_var():
    """Test that sampled programs have a reasonable amount of variance wrt input"""
    are_low_var = [_low_var(o) for o in OUTPUTS]
    frac_low_var = sum(are_low_var) / len(OUTPUTS)
    assert frac_low_var < 0.05, (
        f"{frac_low_var*100}% of programs have low variance in output."
    )


def test_outputs_within_range(magnitude=1e4):
    """Test that program outputs are within a reasonable range."""
    print("outputs:", OUTPUTS)
    print("any none?", np.any(np.isnan(np.array(OUTPUTS, dtype=float))))
    assert np.all(np.abs(OUTPUTS) < magnitude), (
        f"Outputs are not within range (-{magnitude}, {magnitude})."
        f"Found min: {np.min(OUTPUTS)}, max: {np.max(OUTPUTS)}."
    )


def test_no_nones():
    """Test that programs do not return None too often."""
    too_many_nones = [
        _contains_too_many_nones(o) for o in OUTPUTS_UNSANITIZED]
    frac_nones = sum(too_many_nones) / len(OUTPUTS_UNSANITIZED)
    assert frac_nones < 0.05, (
        f"{frac_nones*100}% of programs return None too often."
    )


def _mostly_constant_wrt_input(outputs: ArrayLike) -> bool:
    """Check if program is constant wrt input. 
    Returns True if >80% of inputs produce exactly the same output.
    """
    chex.assert_shape(outputs, (len(INPUTS), 5))
    outputs = [tuple(x) for x in outputs]  # need hashable type
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs) > 0.8


def _low_var(outputs: ArrayLike, threshold=0.01) -> bool:
    """Check if program has low variance wrt input. 
    Returns True if stddev of outputs is below the threshold.
    """
    chex.assert_shape(outputs, (len(INPUTS), 5))
    return np.all(np.std(outputs, axis=0) < threshold)


def _contains_too_many_nones(outputs: ArrayLike) -> bool:
    """Check if program returns None too often."""
    chex.assert_shape(outputs, (len(INPUTS), 5))
    nones = np.isnan(outputs)
    more_than_a_fifth_none = nones.mean() > 0.2
    any_more_than_half_none = (nones.mean(axis=1) > 0.5).any()
    return more_than_a_fifth_none or any_more_than_half_none


def _is_invalid_according_to_dynamic_validator(
        program, inputs: list[list]) -> bool:
    """Check if program passes dynamic validation."""
    return any(len(validating.validate(program, x)) > 0 for x in inputs)