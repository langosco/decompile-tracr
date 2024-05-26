import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import numpy as np

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.sample import sample
from decompile_tracr.dataset import compile
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.compress import autoencoder
from decompile_tracr.compress import transformer

if __name__ == "__main__":
    logger = setup_logger(__name__)
    rng = np.random.default_rng(0)
    key = jax.random.key(0)


    program_toks = tokenizer.tokenize(sample.sample(rng, program_length=5))
    assembled_model = compile.compile_tokens_to_model(program_toks)


    logger.info("Training autoencoder to compress residual stream...")
    key, subkey = jax.random.split(key)
    train_out = autoencoder.train_autoencoder(
        subkey, assembled_model, nsteps=10)
    ae_state, ae_model = train_out['state'], train_out['model']


    def encode(x):
        return ae_model.apply({'params': ae_state.params}, x, method=ae_model.encode)


    get_batch = transformer.DataGenerator(
        assembled_model=assembled_model,
        encode=encode,
        batch_size=32,
        seq_len=6,
    )


    logger.info(
        "Training transformer to match the compiled model layer-by-layer...")
    model, state, log = transformer.train_transformer(
        subkey, 
        get_batch=get_batch, 
        args=transformer.TransformerTrainingArgs(
            nsteps=1000,
            learning_rate=1e-3,
        ),
    )

    logger.info(f"Final loss: {log[-1]['train/loss']}")