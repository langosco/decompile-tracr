import flax
from jaxtyping import ArrayLike
from collections import defaultdict
import networkx as nx
import jax.flatten_util
from tracr.compiler import craft_graph_to_model
from tracr.rasp import rasp
from tracr.compiler import nodes

import rasp_tokenizer.vocab as tokenizer_vocab


def get_layer_name_from_number(layer_number: int) -> str:
    """Naming scheme: layer_0/attn, layer_0/mlp, layer_1/attn, etc."""
    new_layer_number = layer_number // 2
    if layer_number % 2 == 0:
        return f"layer_{new_layer_number}/attn"
    else:
        return f"layer_{new_layer_number}/mlp"


def get_nodes_by_layer(
        program_graph: nx.DiGraph,
        sources: list[nodes.Node]
    ) -> dict[list[str]]:
    """Convert a RASP program to a dict that maps every layer
    to corresponding RASP operations performed by that layer."""
    # this is a dict nodes -> layer number:
    nodes_to_layers = craft_graph_to_model._allocate_modules_to_layers(program_graph, sources)


    # we want a dictionary the other way around, i.e. 
    # layer_name -> list of nodes:
    n_layers = max(nodes_to_layers.values()) + 1
    if n_layers % 2 != 0:
        n_layers += 1  # must be even

    layers_to_nodes = {get_layer_name_from_number(i): []
                       for i in range(n_layers)}
    for node_id, layer in nodes_to_layers.items():
        layername = get_layer_name_from_number(layer)
        if node_id.startswith("aggregate") or node_id.startswith("selector_width"):
            # Include selector as well.
            # Note this will double count if a selector appears as
            # an argument of multiple aggregates or selector-widths
            # There's probably no way around this, since the same
            # selector can appear in multiple layers.
            selector_id = list(program_graph.predecessors(node_id))[0]
            assert selector_id.startswith("select_")
            layers_to_nodes[layername].append(selector_id)
        layers_to_nodes[layername].append(node_id)

    return layers_to_nodes


def add_variable_names_to_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Allocate variable names from the tokenizer vocabulary
    to the nodes in the graph."""
    sop_labels = sorted(list(graph.nodes))
    sop_names = iter(tokenizer_vocab.sop_variables)
    sel_names = iter(tokenizer_vocab.selector_variables)

    for label in sop_labels:
        if label in ['tokens', 'indices']:
            continue

        if label.startswith('select'):
            graph.nodes[label]["token"] = next(sel_names)
        else:
            graph.nodes[label]["token"] = next(sop_names)
    return graph


def get_encoding(graph: nx.DiGraph, node_id: int) -> str:
    if node_id.startswith('select_'):
        return None  # Selectors don't have an encoding
    else:
        expr = graph.nodes[node_id]["EXPR"]
        return expr.annotations['encoding'].value


def get_classname(graph: nx.DiGraph, node_id: int) -> str:
    expr = graph.nodes[node_id]["EXPR"]
    return type(expr).__name__


def get_variable_name(graph: nx.DiGraph, node_id: int) -> str:
    if node_id in ['tokens', 'indices']:
        return node_id
    else:
        node = graph.nodes[node_id]
        return node["token"]


def get_args(graph: nx.DiGraph, node_id: int) -> list[str]:
    expr = graph.nodes[node_id]["EXPR"]
    pred_ids = list(graph.predecessors(node_id))
    variable_args = [get_variable_name(graph, i) for i in pred_ids]

    if isinstance(expr, rasp.Select):
        other_args = [expr.predicate.name]
    elif isinstance(expr, rasp.LinearSequenceMap):
        other_args = [expr.fst_fac, expr.snd_fac]
    elif isinstance(expr, (rasp.Map, rasp.SequenceMap)):
        other_args = [repr(expr.f)]
    else:
        other_args = []

    return other_args + variable_args


def rasp_graph_to_layerwise_representation(
        graph: nx.DiGraph,
        sources,
    ) -> dict[list[str]]:
    """Return a representation of the program as a list of ops (per 
    layer) in order to tokenize it."""

    layers_to_nodes = get_nodes_by_layer(graph, sources)
    graph = add_variable_names_to_graph(graph)

    layerwise_program = {}

    for layer, node_ids in layers_to_nodes.items():
        flat_layer = ["START"]
        for node_id in node_ids:
            flat_layer.append(get_variable_name(graph, node_id))
            flat_layer.append(get_encoding(graph, node_id))
            flat_layer.append(get_classname(graph, node_id))
            flat_layer.extend(get_args(graph, node_id))
        flat_layer.append("END")
        flat_layer = [x for x in flat_layer if x is not None]
        layerwise_program[layer] = flat_layer
    
    return layerwise_program


def get_params(params: dict, layer_name: str) -> jax.Array:
    """
    params: hk parameters as returned by tracr compiler in model.params.
    layer_name: name of the layer to extract parameters for.
    
    Assume layer_name is in format `layer_n/attn` or `layer_n/mlp`.
    Return parameters for that layer as a 1-d array.
    """
    prefix = f'transformer/{layer_name}'

    if layer_name.endswith('attn'):
        layer_params = [
            params[f"{prefix}/{k}"] for k in ['key', 'query', 'value', 'linear']
        ]
    elif layer_name.endswith('mlp'):
        layer_params = [
            params[f'{prefix}/{k}'] for k in ['linear_1', 'linear_2']
        ]
    else:
        raise ValueError(f'Unknown layer name {layer_name}.')
    
    return jax.flatten_util.ravel_pytree(layer_params)[0]


@flax.struct.dataclass
class RaspFlatDatapoint:
    """Holds a single datapoint consisting of a tokenized
    program and a flat array of weights."""
    program: ArrayLike
    weights: ArrayLike

