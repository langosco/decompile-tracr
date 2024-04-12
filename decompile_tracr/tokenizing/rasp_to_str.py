# Description: Convert a RASP program to a string representation
# by layer. For example:
#    SelectorWidth(Select(tokens, tokens, rasp.Comparison.LT))
# becomes:
#    layer_0/attn: ['BOS', 'sop_0', 'categorical', 'SelectorWidth', 
#        'tokens', 'tokens', 'LT', 'EOS']
#    layer_0/mlp: ['BOS', 'EOS']


import networkx as nx

from tracr.compiler import basis_inference
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases

from tracr.compiler import craft_graph_to_model
from tracr.rasp import rasp
from tracr.compiler import nodes

from decompile_tracr.tokenizing import vocab


def rasp_to_str(program: rasp.SOp) -> list[str]:
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

    rasp_str = rasp_graph_to_str(graph, sources)
    validate_rasp_str(rasp_str)
    return rasp_str


def rasp_graph_to_str(
    graph: nx.DiGraph,
    sources,
) -> list[str]:
    """Convert a rasp graph to a representation as list of string tokens.
    """
    layers_to_nodes = get_sops_by_layer(graph, sources)
    graph = add_variable_names_to_graph(graph)

    rep = [vocab.BOS]

    for node_ids in layers_to_nodes.values():
        for node_id in node_ids:
            rep.append(get_variable_name(graph, node_id))
            rep.append(get_encoding(graph, node_id))
            rep.append(get_classname(graph, node_id))
            rep.extend(get_args(graph, node_id))
            rep.append(vocab.EOO)
        rep.append(vocab.EOL)
    rep.append(vocab.EOS)
    return rep


def get_sops_by_layer(
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
        layers_to_nodes[get_layer_name_from_number(layer)
                        ].append(node_id)

    return layers_to_nodes


def add_variable_names_to_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Allocate variable names from the tokenizer vocabulary
    to the nodes in the graph."""
    # Assign variable names in graph-topological order, 
    # independent of SOp labels. At compile time, SOps 
    # are stored in the residual stream ordered by SOp label, 
    # so this can result in two equally tokenized programs 
    # having different compiled weights.
    sop_labels = nx.topological_sort(graph)
    sop_names = iter(vocab.sop_variables)

    for label in sop_labels:
        assert 'token' not in graph.nodes[label]
        assert graph.nodes[label]["EXPR"].label == label

        if isinstance(graph.nodes[label]["EXPR"], rasp.TokensType):
            graph.nodes[label]["token"] = "tokens"
        elif isinstance(graph.nodes[label]["EXPR"], rasp.IndicesType):
            graph.nodes[label]["token"] = "indices"
        elif label.startswith('select_'):
            assert isinstance(graph.nodes[label]["EXPR"], rasp.Select)
        else:
            assert not isinstance(
                graph.nodes[label]["EXPR"], rasp.Select)
            graph.nodes[label]["token"] = next(sop_names)
    return graph


def get_variable_name(graph: nx.DiGraph, node_id: int) -> str:
    """Assumes variable names have been added to the graph.
    by `add_variable_names_to_graph`."""
    node = graph.nodes[node_id]
    return node["token"]


def get_encoding(graph: nx.DiGraph, node_id: int) -> str:
    """Returns encoding ('numerical' or 'categorical')."""
    expr = graph.nodes[node_id]["EXPR"]
    return expr.annotations['encoding'].value


def get_classname(graph: nx.DiGraph, node_id: int) -> str:
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