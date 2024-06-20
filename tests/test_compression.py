import pytest
import jax
import numpy as np
import jax.numpy as jnp

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.sample import sample
from decompile_tracr.dataset import compile
from decompile_tracr.compress import compress
from decompile_tracr.compress.utils import AssembledModelInfo
from decompile_tracr.dataset.reconstruct import ModelFromParams

rng = np.random.default_rng(0)
HIDDEN_SIZE = 15

@pytest.fixture(scope='module')
def x() -> AssembledModelInfo:
    program_toks = tokenizer.tokenize(
        sample.sample(rng, program_length=5))
    m = compile.compile_(tokenizer.detokenize(program_toks))
    x = AssembledModelInfo(m)
    x.params = m.params
    return x


@pytest.fixture(scope='module')
def autoencoder(x):
    model = ModelFromParams(x.params, num_heads=x.num_heads)
    return compress.train_autoencoder(jax.random.key(0), model, nsteps=10, 
        dtype=jnp.float32, hidden_size=HIDDEN_SIZE)


@pytest.fixture(scope='module')
def svd(x):
    model = ModelFromParams(x.params, num_heads=x.num_heads)
    return compress.train_svd(model=model, hidden_size=HIDDEN_SIZE)


def test_autoencoder_methods(autoencoder):
    """Autoencoder methods are consistent."""
    wenc, wdec, aux = autoencoder
    c = compress.Compresser(wenc, wdec)
    ae, state = aux['model'], aux['state']
    h, d = c.hidden_size, c.original_size

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
    assert np.all(encode(x) == c.encode_activations(x))
    assert np.all(decode(z) == c.decode_activations(z))
    assert np.all(encode_decode(x) == decode(encode(x)))
    assert np.all(encode_decode(x) == 
                  c.decode_activations(c.encode_activations(x)))


def test_svd_methods(svd):
    wenc, wdec, aux = svd
    c = compress.Compresser(wenc, wdec)
    svd = aux['svd']
    h, d = c.hidden_size, c.original_size
    x = rng.normal(size=(10, d))
    z = rng.normal(size=(10, h))

    assert np.all(c.encode_activations(x) == svd.transform(x))
    assert np.all(
        c.decode_activations(z) == 
        svd.inverse_transform(z)
    )
    assert np.all(
        c.decode_activations(c.encode_activations(x)) == 
        svd.inverse_transform(svd.transform(x))
    )