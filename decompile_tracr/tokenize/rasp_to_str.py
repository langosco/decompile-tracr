# Description: Convert a RASP program to a string representation
# by layer. For example:
#    SelectorWidth(Select(tokens, tokens, rasp.Comparison.LT))
# becomes:
#    layer_0/attn: ['BOS', 'sop_0', 'categorical', 'SelectorWidth', 
#        'tokens', 'tokens', 'LT', 'EOS']
#    layer_0/mlp: ['BOS', 'EOS']


import networkx as nx
from typing import Sequence

from tracr.compiler import basis_inference
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.compiler import nodes
from tracr.craft import bases

from tracr.compiler import craft_graph_to_model
from tracr.rasp import rasp
from tracr.compiler import nodes

from decompile_tracr.tokenize import vocab

Node = nodes.Node


def rasp_to_str(program: rasp.SOp) -> list[str]:
    graph, sources = get_rasp_graph(program)
    rasp_str = rasp_graph_to_str(graph, sources)
    validate_rasp_str(rasp_str)
    return rasp_str


def get_rasp_graph(program: rasp.SOp) -> tuple[nx.DiGraph, Sequence[Node]]:
    dummy_vocab = {0}
    dummy_max_seq_len = 1
    dummy_bos="bos"
    dummy_mlp_exactness = 1

    extracted = rasp_to_graph.extract_rasp_graph(program)
    graph, sources, sink = extracted.graph, extracted.sources, extracted.sink

    basis_inference.infer_bases(
        graph,
        sink,
        vocab=dummy_vocab,
        max_seq_len=dummy_max_seq_len,
    )

    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, dummy_bos),
        mlp_exactness=dummy_mlp_exactness,
    )
    return graph, sources


def rasp_graph_to_str(
    graph: nx.DiGraph,
    sources: Sequence[Node],
) -> list[str]:
    """Convert a rasp graph to a representation as list of string tokens.
    """
    node_to_layer = craft_graph_to_model._allocate_modules_to_layers(
        graph, sources)
    layers_to_nodes = get_nodes_by_layer(node_to_layer)
    sop_names = iter(vocab.sop_variables)

    rep = [vocab.BOS]
    for nodes in layers_to_nodes.values():
        node_reps = [node_to_str(graph, sop) for sop in nodes]

        # assign sop names:
        for node_rep, node_id in sorted(zip(node_reps, nodes)):
            varname = next(sop_names)
            graph.nodes[node_id]["varname"] = varname
            rep.extend([varname] + node_rep)
        
        rep.append(vocab.EOL)
    rep.append(vocab.EOS)
    return rep


def get_nodes_by_layer(node_to_layer: dict[str, int]) -> dict[str, list[str]]:
    """Requires that node_to_layer is a dict that maps 
    node_id -> layer_number, e.g. {"map_1": 1, "map_2": 1}.
    This function returns an inverted mapping, i.e. a dictionary that maps
    layer -> set of node_ids, e.g. {"layer_0/mlp": {"map_1", "map_2"}}.
    """
    n_layers = max(node_to_layer.values()) + 1
    if n_layers % 2 != 0:
        n_layers += 1  # must be even

    layers_to_nodes = {get_layer_name_from_number(i): set()
                       for i in range(n_layers)}

    for node_id, layer in node_to_layer.items():
        assert isinstance(node_id, str)
        assert isinstance(layer, int)
        layers_to_nodes[get_layer_name_from_number(layer)
                        ].add(node_id)
    
    return layers_to_nodes


def node_to_str(graph: nx.DiGraph, node_id: str) -> list[str]:
    """Tokenize a single node, eg 
    'map_1' -> ['numerical', 'Map', 'lambda x: x + 1', 'tokens']."""
    return [
        get_encoding(graph, node_id),
        get_classname(graph, node_id),
        *get_args(graph, node_id),
        vocab.EOO,
    ]


def get_encoding(graph: nx.DiGraph, node_id: str) -> str:
    """Returns encoding ('numerical' or 'categorical')."""
    expr = graph.nodes[node_id]["EXPR"]
    return expr.annotations['encoding'].value


def get_classname(graph: nx.DiGraph, node_id: str) -> str:
    """Get name of Op, e.g. 'Map' for rasp.Map, or SelectAggregate
    for rasp.Aggregate."""
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


def get_args(graph: nx.DiGraph, node_id: str) -> list[str]:
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


def get_variable_name(graph: nx.DiGraph, node_id: str) -> str:
    """Assumes variable names have been added to the graph.
    by `add_variable_names_to_graph`."""
    node = graph.nodes[node_id]
    if isinstance(node["EXPR"], rasp.TokensType):
        return vocab.inputs[0]
    elif isinstance(node["EXPR"], rasp.IndicesType):
        return vocab.inputs[1]
    else:
        return node["varname"]


def get_layer_name_from_number(layer_number: int) -> str:
    """Naming scheme: layer_0/attn, layer_0/mlp, layer_1/attn, etc."""
    new_layer_number = layer_number // 2
    if layer_number % 2 == 0:
        return f"layer_{new_layer_number}/attn"
    else:
        return f"layer_{new_layer_number}/mlp"


class InvalidRASPStringError(ValueError):
    pass


def validate_rasp_str(rasp_str: str):
    """Validate a string representation of a RASP program.
    """
    if not isinstance(rasp_str, list):
        raise InvalidRASPStringError("Input must be a list (of strings).")
    elif not isinstance(rasp_str[0], str):
        raise InvalidRASPStringError("Input must be a list of strings.")
    
    if not rasp_str.count(vocab.EOL) % 2 == 0:
        raise InvalidRASPStringError(f"Input must have an even number of "
                f"layers. Received: {len(rasp_str.count(vocab.EOL))}.")

    if len(rasp_str) < 2:
        raise InvalidRASPStringError("Program must include at least two "
                                     "tokens: BOS and EOS")
    elif rasp_str[0] != vocab.BOS:
        raise InvalidRASPStringError("Program must start with BOS.")
    elif rasp_str[-1] not in (vocab.EOS, vocab.PAD):
        raise InvalidRASPStringError("Program must end with EOS or PAD.")

    if rasp_str.count(vocab.BOS) != 1 or rasp_str.count(vocab.EOS) != 1:
        raise InvalidRASPStringError("Program must have exactly one "
                                     "BOS and one EOS.")
    
    padding = rasp_str[rasp_str.index(vocab.EOS)+1:]
    if padding != [vocab.PAD] * len(padding):
        raise InvalidRASPStringError("Detected non-padding tokens "
                                     "following EOS.")
    
    for x in rasp_str:
        if x not in vocab.vocab:
            raise InvalidRASPStringError(f"Unkown token: {x}. "
            "(Sometimes this happens when you hand-craft example programs "
            "that are not sampled using the vocabulary).")

    variable_names = set([x for x in rasp_str if x in vocab.sop_variables])
    variable_names_in_vocab_order = vocab.sop_variables[:len(variable_names)]
    if not variable_names == set(variable_names_in_vocab_order):
        raise InvalidRASPStringError("Variable names be assigned from 0 "
            "in increasing order: {variable_names_in_vocab_order}. "
            f"Found: {variable_names}.")