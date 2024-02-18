import os
import pytest
import numpy as np
import shutil


from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile
from decompile_tracr.dataset import data_utils


BASE_DIR = config.data_dir / "test_make_dataset"
rng = np.random.default_rng(42)

@pytest.fixture(scope="module")
def make_test_data():
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(BASE_DIR)

    sampler_kwargs = {
        "n_sops": 10,
        "min_length": 4,
        "max_length": None
    }

    generate.generate(
        rng, 
        ndata=5, 
        name='testing_make_dataset', 
        savedir=BASE_DIR / "unprocessed",
        **sampler_kwargs,
    )

    tokenize_lib.tokenize_lib(savedir=BASE_DIR / "unprocessed")

    dedupe.dedupe(
        loaddir=BASE_DIR / "unprocessed",
        savedir=BASE_DIR / "deduped",
        batchsize=50,
    )

    compile.compile_all(
        loaddir=BASE_DIR / "deduped",
        savedir=BASE_DIR / "full",
    )
    return None


def test_sampling(make_test_data):
    pass



def test_deduplication(make_test_data):
    pass


def test_tokenization(make_test_data):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""
    deduped_dirs = list((BASE_DIR / "deduped").iterdir())
    deduped_dirs = [x for x in deduped_dirs if x.is_dir()]
    data = [data_utils.load_batches(d) for d in deduped_dirs]
    data = [x for ds in data for x in ds]

    for x in data:
        program = tokenizer.detokenize(x['tokens'])
        retokenized = tokenizer.tokenize(program)
        assert x['tokens'] == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
        )


def test_compilation(make_test_data):
    pass