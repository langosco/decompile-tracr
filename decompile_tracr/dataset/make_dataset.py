# Desc: Script to generate a dataset for training a decompiler
# Usually you would want to instead run generate.py and compile.py
# in parallel (eg by running many instances of the scripts), but
# this script is useful for generating a sample dataset using a 
# single thread.

import numpy as np

from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile

rng = np.random.default_rng(42)


if __name__ == "__main__":
    generate.generate(
        rng, 
        ndata=20, 
        name='testing_make_dataset', 
        savedir=config.unprocessed_dir,
        program_length=10,
    )

    tokenize_lib.tokenize_lib(savedir=config.unprocessed_dir)

    dedupe.dedupe(
        loaddir=config.unprocessed_dir,
        savedir=config.deduped_dir,
        batchsize=1024,
    )

    compile.compile_all(
        loaddir=config.deduped_dir,
        savedir=config.full_dataset_dir,
    )