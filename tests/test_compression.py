import pytest
import jax
import numpy as np
import jax.numpy as jnp

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.sample import sample
from decompile_tracr.dataset import compile
from decompile_tracr.compress import compress
from decompile_tracr.compress.utils import AssembledModelInfo

rng = np.random.default_rng(0)

@pytest.fixture(scope='module')
def assembled_model():
    program_toks = tokenizer.tokenize(
        sample.sample(rng, program_length=5))
    assembled_model = compile.compile_tokens_to_model(program_toks)
    return assembled_model


@pytest.fixture(scope='module')
def autoencoder(assembled_model):
    h = AssembledModelInfo(model=assembled_model).d_model // 1.2
    compressed_model, aux = compress.train_autoencoder(
        jax.random.key(0), assembled_model, nsteps=10, 
        dtype=jnp.float32, hidden_size=int(h))
    return compressed_model, aux


@pytest.fixture(scope='module')
def svd(assembled_model):
    h = AssembledModelInfo(model=assembled_model).d_model // 1.2
    compressed_model, aux = compress.train_svd(
        jax.random.key(0), assembled_model, hidden_size=int(h))
    return compressed_model, aux


def test_autoencoder_methods(autoencoder):
    """Autoencoder methods are consistent."""
    cmodel, aux = autoencoder
    ae, state = aux['model'], aux['state']
    h, d = cmodel.hidden_size, cmodel.original_size

    def encode_decode(x):
        return ae.apply({'params': state.params}, x)

    def encode(x):
        return ae.apply({'params': state.params}, x, 
                           method=ae.encode)

    def decode(x):
        return ae.apply({'params': state.params}, x, 
                           method=ae.decode)
    
    x = rng.normal(size=(10, d))
    z = rng.normal(size=(10, h))
    assert np.all(encode(x) == cmodel.encode_activations(x))
    assert np.all(decode(z) == cmodel.decode_activations(z))
    assert np.all(encode_decode(x) == decode(encode(x)))
    assert np.all(encode_decode(x) == 
                  cmodel.decode_activations(
                      cmodel.encode_activations(x)))


def test_svd_methods(svd):
    cmodel, aux = svd
    svd = aux['svd']
    h, d = cmodel.hidden_size, cmodel.original_size
    x = rng.normal(size=(10, d))
    z = rng.normal(size=(10, h))

    assert np.all(cmodel.encode_activations(x) == svd.transform(x))
    assert np.all(
        cmodel.decode_activations(z) == 
        svd.inverse_transform(z)
    )
    assert np.all(
        cmodel.decode_activations(cmodel.encode_activations(x)) == 
        svd.inverse_transform(svd.transform(x))
    )