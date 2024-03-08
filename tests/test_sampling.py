import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import numpy as np

from tracr.rasp import rasp

from decompile_tracr.sampling import rasp_utils
from decompile_tracr.sampling import sampling
from decompile_tracr.sampling.validate import is_valid
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import lib

rng = np.random.default_rng(None)

TEST_INPUTS = [rasp_utils.sample_test_input(rng) 
               for _ in range(100)]
LENGTH = 10
PROGRAMS = [sampling.sample(rng, program_length=LENGTH) for _ in range(30)]


def test_sample():
    for p in PROGRAMS:
        assert rasp_utils.count_sops(p) == LENGTH


def test_validity_without_compiling():
    """Test that sampled programs are valid."""
    valid = [
        all(is_valid(p, x) for x in TEST_INPUTS) for p in PROGRAMS
    ]
    n_programs_valid = sum(valid)
    assert n_programs_valid / len(PROGRAMS) > 0.95, (f"Only {n_programs_valid} / {len(PROGRAMS) * 100}\%"
                                                      "of programs are valid.")