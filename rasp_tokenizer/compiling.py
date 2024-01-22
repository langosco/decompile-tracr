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
