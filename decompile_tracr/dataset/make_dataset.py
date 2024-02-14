
import numpy as np

from decompile_tracr.dataset import config
from decompile_tracr.dataset import generate
from decompile_tracr.dataset import tokenize_lib
from decompile_tracr.dataset import dedupe
from decompile_tracr.dataset import compile

rng = np.random.default_rng(42)

sampler_kwargs = {
    "n_sops": 10,
    "min_length": 4,
    "max_length": None
}

generate.generate(
    rng, 
    ndata=10, 
    name='testing_make_dataset', 
    savedir=config.unprocessed_dir,
    **sampler_kwargs,
)

tokenize_lib.tokenize_lib(savedir=config.unprocessed_dir)

dedupe.dedupe(
    loaddir=config.unprocessed_dir,
    savedir=config.deduped_dir,
)

compile.compile(
    loaddir=config.deduped_dir,
    savedir=config.full_dataset_dir,
)