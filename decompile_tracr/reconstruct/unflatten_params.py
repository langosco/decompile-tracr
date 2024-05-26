import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jaxtyping import ArrayLike
import numpy as np
import jax.numpy as jnp
import haiku as hk
import chex

from tracr.compiler.validating import validate
from tracr.rasp.rasp import Map, SequenceMap, LinearSequenceMap, Select, Aggregate, Comparison, SelectorWidth, indices, tokens 
from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.compiler import craft_graph_to_model
from tracr.compiler import rasp_to_graph
from tracr.compiler import lib as tracr_lib
from tracr.compiler import assemble
from tracr.transformer import model


from decompile_tracr.dataset import lib
from decompile_tracr.dataset import data_utils
from decompile_tracr.dataset import dataloading
from decompile_tracr.dataset import config
from decompile_tracr.dataset import compile as comp
from decompile_tracr.tokenize import tokenizer
from decompile_tracr.tokenize import vocab
from decompile_tracr.sample import sample
from decompile_tracr.sample import rasp_utils
from decompile_tracr.sample.map_primitives import FunctionWithRepr
from decompile_tracr.tokenize.str_to_rasp import split_list
from decompile_tracr.dataset.compile import get_weights


def _compile(program):
    return compiling.compile_rasp_to_model(
        program,
        vocab=set(range(5)),
        max_seq_len=5,
    )


rng = np.random.default_rng(0)
key = jax.random.key(0)

def unflatten_embed(embed: ArrayLike):
    embed = embed.reshape(7+6, -1)
    d_model = embed.shape[-1]
    return {
        "token_embed": {"embeddings": embed[:7]},
        "pos_embed": {"embeddings": embed[7:]},
    }, d_model


flat_embed = data_utils.get_params(m.params, layer_name="embed")
out, d = unflatten_embed(flat_embed)
assert np.all(out['token_embed']['embeddings'] == m.params['token_embed']['embeddings'])
assert np.all(out['pos_embed']['embeddings'] == m.params['pos_embed']['embeddings'])
import jax.flatten_util


def unflatten_attn(attn: ArrayLike, d_model: int, layer: int):
    assert (len(attn) - d_model) % (3 + 4*d_model) == 0
    hidden = (len(attn) - d_model) // (3 + 4*d_model)

    prefix = f'transformer/layer_{layer}/attn/'
    shapes = {
        'key': {'b': (hidden,), 'w': (d_model, hidden)},
        'query': {'b': (hidden,), 'w': (d_model, hidden)},
        'value': {'b': (hidden,), 'w': (d_model, hidden)},
        'linear': {'b': (d_model,), 'w': (hidden, d_model)},
    }
    dummy = {prefix + k: {kk: jnp.zeros(vv) for kk, vv in v.items()} 
             for k, v in shapes.items()}
    _, unflatten = data_utils.get_params(
        dummy, layer_name=f"layer_{layer}/attn", include_unflatten_fn=True)
    names = ['key', 'query', 'value', 'linear']
    names = [prefix + k for k in names]
    return {k: v for k, v in zip(dummy.keys(), unflatten(attn))}


attn = data_utils.get_params(m.params, layer_name="layer_0/attn")
out = unflatten_attn(attn, d, 0)

for k, v in out.items():
    for kk, vv in v.items():
        assert np.all(vv == m.params[k][kk])


def unflatten_mlp(mlp, d_model, layer):
    assert (len(mlp) - d_model) % (1 + 2*d_model) == 0
    hidden = (len(mlp) - d_model) // (1 + 2*d_model)
    prefix = f'transformer/layer_{layer}/mlp/'

    shapes = {
        'linear_1': {'b': (hidden,), 'w': (d_model, hidden)},
        'linear_2': {'b': (d_model,), 'w': (hidden, d_model)},
    }
    dummy = {prefix + k: {kk: jnp.zeros(vv) for kk, vv in v.items()}
                for k, v in shapes.items()}
    _, unflatten = data_utils.get_params(
        dummy, layer_name=f"layer_{layer}/mlp", include_unflatten_fn=True)
    return {k: v for k, v in zip(dummy.keys(), unflatten(mlp))}


mlp = data_utils.get_params(m.params, layer_name="layer_0/mlp")
out = unflatten_mlp(mlp, d, 0)

for k, v in out.items():
    for kk, vv in v.items():
        assert np.all(vv == m.params[k][kk])


def unflatten_params(datapoint: dict):
    layer_idx, weights = datapoint["layer_idx"], datapoint["weights"]
    layers = _split_layers(weights, layer_idx)

    params = {}
    embed, d_model = unflatten_embed(layers[0])
    params.update(embed)

    for i, layer in enumerate(layers[1:]):
        if i % 2 == 0:
            params.update(unflatten_attn(layer, d_model, i//2))
        else:
            params.update(unflatten_mlp(layer, d_model, i//2))
    
    return params


def _split_layers(w, l):
    layers = []
    start_idx = 0
    for end_idx in l:
        if end_idx == 0:
            break
        layers.append(w[start_idx:end_idx])
        start_idx = end_idx
    return layers


# Main problems for recovering the model from the weights:
# To actually run the model, we need to reconstruct the model's forward pass.
# Difficulties:
# - The unembedding parameters are not provided in the parameter dict,
# so we would have to reconstruct them from the weights (in particular from
# the embedding weights). We would also have to figure out whether the output
# is numerical or categorical.
# You can recover a CompiledModel object via
# ```
# m = _compile(p)
# true_compiled_model = hk.transform(m.get_compiled_model).apply({}, key)
# ```
# Then true_compiled_model.use_unembed_argmax will tell you whether the
# output is categorical or numerical. But of course we don't have access
# to m here.
# - When a model has >1 attention heads, the attention weights
# corresponding to the heads are concatenated. We would have to figure out
# the number of heads from the model's parameters and then split the weights
# accordingly.
# - Here's a sketch of how we would do it if we had access to num_heads and
# unembed:
# ```
# key_size = reconstructed['transformer/layer_0/attn/key']['b'].shape
# mlp_hidden_size = reconstructed['transformer/layer_0/mlp/linear_1']['b'].shape
# num_heads = ...
# 
# model_config = model.TransformerConfig(
#     num_heads=num_heads,
#     num_layers=len(x['layer_idx'])//2,
#     key_size=key_size,
#     mlp_hidden_size=mlp_hidden_size,
#     dropout_rate=0.,
#     activation_function=jax.nn.relu,
#     layer_norm=False,
#     causal=False,
# )
#
# def get_compiled_model():
#     def unembed(x): return x
#     is_output_categorical = ...
# 
#     embed_modules = assemble.EmbeddingModules(
#         token_embed=hk.Embed(
#             embedding_matrix=reconstructed['token_embed']['embeddings'], 
#             name="token_embed"
#         ),
#         pos_embed=hk.Embed(
#             embedding_matrix=reconstructed['pos_embed']['embeddings'],
#             name="pos_embed"
#         ),
#         unembed=hk.to_module(unembed)(),
#     )
# 
# 
#     return model.CompiledTransformerModel(
#         transformer=model.Transformer(model_config),
#         token_embed=embed_modules.token_embed,
#         position_embed=embed_modules.pos_embed,
#         unembed=embed_modules.unembed,
#         use_unembed_argmax=is_output_categorical)
# 
# 
# compiled_model = hk.transform(get_compiled_model).apply({}, key)
# ```
# This would still need some extra work - we want to get an 
# AssembledTransformerModel - but that should be fairly easy if we had
# the rest.


if __name__ is "__main__":
    for _ in range(100):
        try:
            t = tokenizer.tokenize(sample.sample(rng, 10))
            p = tokenizer.detokenize(t)
            m = _compile(p)
            w = get_weights(t, max_weights_len=config.MAX_WEIGHTS_LENGTH)
        except:
            continue
        x = {
            "weights": jnp.array(np.concatenate(w)),
            "layer_idx": jnp.array(np.cumsum([len(_) for _ in w])),
        }

        reconstructed_params = unflatten_params(x)
        chex.assert_trees_all_equal_shapes_and_dtypes(reconstructed_params, m.params)
        chex.assert_trees_all_equal(reconstructed_params, m.params)