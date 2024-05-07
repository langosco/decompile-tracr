import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import chex
from jaxtyping import ArrayLike
import numpy as np
import jax.numpy as jnp

from tracr.compiler.assemble import AssembledTransformerModel

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling
from decompile_tracr.dataset import compile
from decompile_tracr.training import autoencoder
from decompile_tracr.training import transformer

from metamodels_for_rasp.train import Updater


rng = np.random.default_rng(0)
key = jax.random.key(0)

# assume a tokenized rasp program is given
# - compile it to a model
# - train autoencoder on residuals
# - train transformer on encoded residuals

def train_autoencoder(key: jax.random.PRNGKey,
                      assembled_model: AssembledTransformerModel):
    key, subkey = jax.random.split(key)
    updater, state = autoencoder.init_autoencoder(subkey, assembled_model)
    get_residuals = autoencoder.get_residuals_sampler(
        assembled_model, seq_len=5, batch_size=10)
    log = []

    for _ in range(100000):
        key, subkey = jax.random.split(key)
        batch = get_residuals(subkey)
        state, aux = updater.update(state, batch)
        log.append(aux)
    
    return state, log


program_toks = tokenizer.tokenizer(sampling.sample(rng, program_length=5))
assembled_model = compile.compile_tokens_to_model(program_toks)

autoencoder_state, _ = train_autoencoder(key, assembled_model)


# train transformer on encoded residuals
key, subkey = jax.random.split(key)
model, updater, state = transformer.init_transformer(subkey, assembled_model)





