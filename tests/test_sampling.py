import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import numpy as np
from collections import Counter

from tracr.rasp import rasp

from decompile_tracr.sampling import rasp_utils
from decompile_tracr.sampling import sampling
from decompile_tracr.sampling.validate import is_valid
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import lib

rng = np.random.default_rng(None)

TEST_INPUTS = [rasp_utils.sample_test_input(rng, max_seq_len=5, min_seq_len=5) 
               for _ in range(100)]
LENGTH = 10
PROGRAMS = [sampling.sample(rng, program_length=LENGTH) for _ in range(30)]
OUTPUTS = [[p(x) for x in TEST_INPUTS] for p in PROGRAMS]
# replace None with 0s:
OUTPUTS = [tuple(0 if x is None else x for x in o) for p_outs in OUTPUTS for o in p_outs]


def test_sample():
    for p in PROGRAMS:
        assert rasp_utils.count_sops(p) == LENGTH


def test_validity_without_compiling():
    """Test that sampled programs are valid."""
    valid = [
        all(is_valid(p, x) for x in TEST_INPUTS) for p in PROGRAMS
    ]
    n_programs_valid = sum(valid)
    assert n_programs_valid / len(PROGRAMS) > 0.95, (
        f"Only {n_programs_valid} / {len(PROGRAMS) * 100}\% of programs are valid.")


def test_constant_wrt_input():
    """Test that sampled programs are not constant wrt input."""
    are_constant = [_constant_wrt_input(o) for o in OUTPUTS]
    frac_constant = sum(are_constant) / len(OUTPUTS)
    assert frac_constant < 0.05, (
        f"{frac_constant*100}% of programs produce the same output for > 80% of test inputs."
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


def _constant_wrt_input(outputs: list[tuple]) -> bool:
    """Check if program is constant wrt input. 
    Returns True if >80% of inputs produce exactly the same output.
    """
    counts = Counter(outputs)
    return counts.most_common(1)[0][1] / len(outputs) > 0.8


def _low_var(outputs: list[tuple], threshold=0.01) -> bool:
    """Check if program has low variance wrt input. 
    Returns True if stddev of outputs is below the threshold.
    """
    return np.std(outputs) < 0.1