import jax
import numpy as np

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling
from decompile_tracr.dataset import compile
from decompile_tracr.training import autoencoder


rng = np.random.default_rng(0)
key = jax.random.key(0)


def test_autoencoder():
    # TODO
    if False:
        program_toks = tokenizer.tokenize(sampling.sample(rng, program_length=5))
        assembled_model = compile.compile_tokens_to_model(program_toks)

        # Train autoencoder to compress residual stream
        key, subkey = jax.random.split(key)
        train_out = autoencoder.train_autoencoder(
            subkey, assembled_model, nsteps=10)
        ae_state, log, ae_model = (
            train_out['state'], train_out['log'], train_out['model'],
        )
