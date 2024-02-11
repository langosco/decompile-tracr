import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest

from tracr.rasp import rasp

from rasp_generator import rasp_utils
from rasp_tokenizer import tokenizer
from rasp_tokenizer import lib


SAMPLE_PROGRAMS = lib.examples


@pytest.mark.parametrize("program", SAMPLE_PROGRAMS)
def test_round_trip(program: rasp.SOp):
    """Tokenize, then detokenize"""
    tokens = tokenizer.tokenize(program)
    reconstructed = tokenizer.detokenize(tokens)

    assert rasp_utils.is_equal(program, reconstructed, recursive=True), (
        f"Detokenized does not equal original.")
    
    retokenized = tokenizer.tokenize(reconstructed)
    assert tokens == retokenized, (
        f"tokenize(reconstructed) does not equal tokenize(original).")
