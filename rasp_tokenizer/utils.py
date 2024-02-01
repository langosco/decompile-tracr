import fcntl
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

    for label in sop_labels:
        if label in ['tokens', 'indices']:
            graph.nodes[label]["token"] = label
        elif label.startswith('select_'):
            assert isinstance(graph.nodes[label]["EXPR"], rasp.Select)
            pass
#            graph.nodes[label]["token"] = next(sel_names)
        else:
            assert not isinstance(
                graph.nodes[label]["EXPR"], rasp.Select)
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
    if isinstance(expr, rasp.Select):
        raise ValueError("No need to get classnames for selectors, "
                         "you're probably doing something wrong.")
    elif isinstance(expr, rasp.Aggregate):
        return "SelectAggregate"
    elif isinstance(expr, rasp.SelectorWidth):
        return "SelectorWidth"
    else:
        return type(expr).__name__


def get_variable_name(graph: nx.DiGraph, node_id: int) -> str:
    node = graph.nodes[node_id]
    return node["token"]


def get_args(graph: nx.DiGraph, node_id: int) -> list[str]:
    expr = graph.nodes[node_id]["EXPR"]
    assert expr.label == node_id

    if isinstance(expr, rasp.Map):
        args = [repr(expr.f), get_variable_name(graph, expr.inner.label)]
    elif isinstance(expr, rasp.LinearSequenceMap):
        args = [
            get_variable_name(graph, expr.fst.label), 
            get_variable_name(graph, expr.snd.label),
            str(expr.fst_fac),
            str(expr.snd_fac),
        ]
    elif isinstance(expr, rasp.SequenceMap):
        args = [
            repr(expr.f), 
            get_variable_name(graph, expr.fst.label), 
            get_variable_name(graph, expr.snd.label),
        ]
    elif isinstance(expr, rasp.Aggregate):
        args = [
            *get_args(graph, expr.selector.label),
            get_variable_name(graph, expr.sop.label),
        ]
    elif isinstance(expr, rasp.SelectorWidth):
        args = get_args(graph, expr.selector.label)
    elif isinstance(expr, rasp.Select):
        args = [
            get_variable_name(graph, expr.keys.label),
            get_variable_name(graph, expr.queries.label),
            expr.predicate.name,
        ]
    else:
        raise ValueError(f"Unknown expression type {type(expr)}.")

    return args



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
        flat_layer = []
        flat_layer.append(tokenizer_vocab.BOS)
        for node_id in node_ids:
            if node_id.startswith('select_'):
                # Skip selectors, they are included as arguments
                # of the corresponding aggregate or selector-width
                continue
            if not flat_layer[-1] == tokenizer_vocab.BOS:
                flat_layer.append(tokenizer_vocab.SEP)
            flat_layer.append(get_variable_name(graph, node_id))
            flat_layer.append(get_encoding(graph, node_id))
            flat_layer.append(get_classname(graph, node_id))
            flat_layer.extend(get_args(graph, node_id))
        flat_layer.append(tokenizer_vocab.EOS)
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


def sequential_count_via_lockfile(countfile="/tmp/counter.txt"):
    """Increment a counter in a file. 
    Use a lockfile to ensure atomicity. If the file doesn't exist, 
    create it and start the counter at 1."""
    try:
        with open(countfile, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            f.seek(0)
            counter_str = f.read().strip()
            counter = 1 if not counter_str else int(counter_str) + 1

            f.seek(0)
            f.truncate()  # Clear the file content
            f.write(str(counter))
            f.flush()

            fcntl.flock(f, fcntl.LOCK_UN)

        return counter
    except ValueError as e:
        raise ValueError(f"Invalid counter value: {counter_str}. {e}")
