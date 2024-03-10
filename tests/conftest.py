import os
import pytest
import numpy as np
import shutil

from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile


rng = np.random.default_rng(42)


@pytest.fixture(scope="session")
def base_dir():
    return config.data_dir / "test_make_dataset"


@pytest.fixture(scope="session")
def make_test_data():
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir)

    generate.generate(
        rng, 
        ndata=5, 
        name='testing_make_dataset', 
        savedir=base_dir / "unprocessed",
        program_length=10,
    )

    tokenize_lib.tokenize_lib(savedir=base_dir / "unprocessed")

    # to make sure that dedupe is working:
    for f in (base_dir / "unprocessed").iterdir():
        if f.is_file() and f.suffix == ".json":
            shutil.copy(f, base_dir / "unprocessed" / f"duplicate_{f.name}")

    dedupe.dedupe(
        loaddir=base_dir / "unprocessed",
        savedir=base_dir / "deduped",
        batchsize=50,
    )

    compile.compile_all(
        loaddir=base_dir / "deduped",
        savedir=base_dir / "full",
    )
    return None

