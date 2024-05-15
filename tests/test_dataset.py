import os
import pytest
import numpy as np
import shutil
import jax
import h5py

from decompile_tracr.sampling import rasp_utils
from decompile_tracr.sampling import sampling
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile
from decompile_tracr.dataset import data_utils


rng = np.random.default_rng()

@pytest.fixture()
def tokens():
    NDATA = 1000
    with h5py.File(config.data_dir / "full.h5", "r") as f:
        return f['train/tokens'][:NDATA]


def test_sampling():
    #data = data_utils.load_batches(base_dir / "unprocessed")
    pass


def test_deduplication():
    pass


def test_tokenization(tokens):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""

    for x in tokens:
        x = x.tolist(); x = x[:x.index(vocab.pad_id)]
        program = tokenizer.detokenize(x)
        retokenized = tokenizer.tokenize(program)
        assert x == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
        )


def test_compilation():
    # this is hard because we can't easily load a compiled model
    # once we saved the weights.
    pass