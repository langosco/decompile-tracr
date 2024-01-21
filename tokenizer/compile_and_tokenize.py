from typing import Set

import networkx as nx

from tracr.compiler import assemble
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.compiler import validating
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.rasp import rasp


import tokenizer


COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"


def compile_rasp_to_model_and_return_graph(
    program: rasp.SOp,
    vocab: Set[rasp.Value],
    max_seq_len: int,
    causal: bool = False,
    compiler_bos: str = COMPILER_BOS,
    compiler_pad: str = COMPILER_PAD,
    mlp_exactness: int = 100,
) -> (assemble.AssembledTransformerModel, nx.DiGraph, list[nodes.Node]):
    """Exactly the same as `tracr.compiling.compile_rasp_to_model`,
    but returns the program graph in addition to the compiled model."""

    if compiler_bos in vocab:
        raise ValueError(
            "Compiler BOS token must not be present in the vocab. "
            f"Found '{compiler_bos}' in {vocab}"
        )

    if compiler_pad in vocab:
        raise ValueError(
            "Compiler PAD token must not be present in the vocab. "
            f"Found '{compiler_pad}' in {vocab}"
        )

    # Perform static validation to fail fast. This catches most programs that
    # tracr is unable to compile.
    unsupported_exprs = validating.static_validate(program)

    if unsupported_exprs:
        error_message = "\n".join(
            (f"{expr.expr.name}: {expr.reason}" for expr in unsupported_exprs)
        )
        error_message = f"Unsupported RASP expressions:\n{error_message}"
        raise NotImplementedError(error_message)

    extracted = rasp_to_graph.extract_rasp_graph(program)
    graph, sources, sink = extracted.graph, extracted.sources, extracted.sink

    basis_inference.infer_bases(
        graph,
        sink,
        vocab,
        max_seq_len,
    )

    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
        mlp_exactness=mlp_exactness,
    )

    craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

    compiled_model = craft_model_to_transformer.craft_model_to_transformer(
        craft_model=craft_model,
        graph=graph,
        sink=sink,
        max_seq_len=max_seq_len,
        causal=causal,
        compiler_bos=compiler_bos,
        compiler_pad=compiler_pad,
    )
    return compiled_model, graph, sources


def rasp_to_layerwise_representation(program_graph: nx.DiGraph, sources: list[nodes.Node]):
    """Convert a RASP program to a representation that maps every layer
    to corresponding RASP operations performed by that layer."""
    nodes_to_layers = craft_graph_to_model._allocate_modules_to_layers(program_graph, sources)

    # we want a dictionary the other way around, i.e. mapping from layer to RASP operations
    n_layers = max(nodes_to_layers.values()) + 1
    if n_layers % 2 != 0:
        n_layers += 1  # n_layers is always even (tracr will add dummy MLP block at the end)
    layers_to_nodes = {layer: [] for layer in range(n_layers)}
    for node_id, layer in nodes_to_layers.items():
        if node_id.startswith("aggregate") or node_id.startswith("selector_width"):
            # include selector as well
            # note this will double count if a selector appears as
            # an argument of multiple aggregates or selector-widths
            # There's probably no way around this, since the same
            # selector can appear in multiple layers.
            selector_id = list(program_graph.predecessors(node_id))[0]
            assert selector_id.startswith("select_")
            layers_to_nodes[layer].append(selector_id)
        layers_to_nodes[layer].append(node_id)

    return layers_to_nodes


def add_variable_names_to_graph(graph):
    """Allocate variable names from the tokenizer vocabulary
    to the nodes in the graph."""
    sop_labels = sorted(list(graph.nodes))
    sop_names = iter(tokenizer.vocab.sop_variables)
    sel_names = iter(tokenizer.vocab.selector_variables)

    for label in sop_labels:
        if label in ['tokens', 'indices']:
            continue

        if label.startswith('select'):
            graph.nodes[label]["token"] = next(sel_names)
        else:
            graph.nodes[label]["token"] = next(sop_names)
    return graph


def get_encoding(graph, node_id):
    if node_id.startswith('select_'):
        return None  # Selectors don't have an encoding
    else:
        expr = graph.nodes[node_id]["EXPR"]
        return expr.annotations['encoding'].value


def get_classname(graph, node_id):
    expr = graph.nodes[node_id]["EXPR"]
    return type(expr).__name__


def get_variable_name(graph, node_id):
    if node_id in ['tokens', 'indices']:
        return node_id
    else:
        node = graph.nodes[node_id]
        return node["token"]


def get_args(graph, node_id):
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


def flatten_program(graph: nx.DiGraph, sources) -> list:
    """Return a flat representation of the program
    in order to tokenize it."""

    layers_to_nodes = rasp_to_layerwise_representation(graph, sources)
    graph = add_variable_names_to_graph(graph)

    flat_representation = {}

    for layer, node_ids in layers_to_nodes.items():
        flat_layer = []
        for node_id in node_ids:
            flat_layer.append(get_variable_name(graph, node_id))
            flat_layer.append(get_encoding(graph, node_id))
            flat_layer.append(get_classname(graph, node_id))
            flat_layer.extend(get_args(graph, node_id))
            flat_layer.append("SEP")
        flat_layer = [x for x in flat_layer if x is not None]
        flat_representation[layer] = flat_layer
    
    return flat_representation


def tokenize(flat_program):
    vocab = tokenizer.vocab.vocab
    tokenized_program = {}
    for layer, layer_data in flat_program.items():
        tokenized_program[layer] = [vocab.index(x) for x in layer_data]
    return tokenized_program


def compile_and_tokenize(
        program: rasp.SOp,
        vocab={0,1,2,3,4,5},
        max_seq_len=5
    ):

    model, graph, sources = compile_rasp_to_model_and_return_graph(
        program,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )

    flat_program = flatten_program(graph, sources)
    tokens = tokenize(flat_program)
    return model, tokens