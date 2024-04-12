import os
import pytest
import numpy as np
import shutil
import jax


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


def test_sampling(base_dir, make_test_data):
    #data = data_utils.load_batches(base_dir / "unprocessed")
    pass


def test_deduplication(make_test_data):
    pass


def test_tokenization(base_dir, make_test_data):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""
    deduped_dirs = list((base_dir / "deduped").iterdir())
    deduped_dirs = [x for x in deduped_dirs if x.is_dir()]
    data = [data_utils.load_batches(d) for d in deduped_dirs]
    data = [x for ds in data for x in ds]

    for x in data:
        program = tokenizer.detokenize(x['tokens'])
        retokenized = tokenizer.tokenize(program)
        assert x['tokens'] == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
        )


def test_compilation(base_dir, make_test_data):
    # this is hard because we can't easily load a compiled model
    # once we saved the weights.
    pass


def test_weights_range(base_dir, make_test_data):
    """Test that compiled model weights are within a reasonable range."""
    LOWER, UPPER = -1e6, 1e6
    print("type:", type(base_dir))
    data = data_utils.load_batches(base_dir / "full")

    def _test_weights_range(data, split_layers: bool):
        data = data_utils.prepare_dataset(data, split_layers=split_layers)
        data = [x['weights'] for x in data]
        with jax.default_device(jax.devices("cpu")[0]):
            data = jax.flatten_util.ravel_pytree(data)[0]
        assert np.all(data > LOWER) and np.all(data < UPPER), (
            "Some parameters exceed bounds. Found parameters of sizes "
            f"(min, max): ({np.min(data)}, {np.max(data)})."
        )

    _test_weights_range(data, split_layers=False)
    _test_weights_range(data, split_layers=True)