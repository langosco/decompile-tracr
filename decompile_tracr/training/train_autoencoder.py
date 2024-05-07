import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.sampling import sampling
from decompile_tracr.dataset import compile
from decompile_tracr.training import autoencoder



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    key = jax.random.key(0)
    toks = tokenizer.tokenize(sampling.sample(rng, 5))
    m = compile.compile_tokens_to_model(toks)
    key, subkey = jax.random.split(key)
    updater, state = autoencoder.init_autoencoder(subkey, m)
    get_residuals = autoencoder.get_residuals_sampler(m, seq_len=5, batch_size=10)
    log = []

    for step in range(100000):
        key, subkey = jax.random.split(key)
        residuals, embeddings = get_residuals(subkey)
        batch = jnp.concatenate([residuals, embeddings])
        state, aux = updater.update(state, batch)
        log.append(aux)


    plt.plot([l["train/loss"] for l in log])
    plt.yscale('log')
    plt.xscale('log')

    print('final loss:', log[-1]["train/loss"])