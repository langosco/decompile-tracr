import os
from tracr.rasp import rasp
from tracr.compiler import compiling, rasp_to_graph
import numpy as np
from tracr.compiler.validating import validate
from typing import Union, TypeVar
from lauro_inverse_tracr import map_primitives, sampling
import networkx as nx
import inspect


def get_args(fn: callable):
    """Return the arguments of a function. Ignores keyword arguments."""
    argspec = inspect.getfullargspec(fn)
    n_kwargs = len(argspec.defaults)
    args = argspec.args
    args = args[:-n_kwargs] if n_kwargs > 0 else args
    if 'self' in args:
        args = args[1:]
    return args


def print_expr(expr: rasp.RASPExpr, test_input=None):
    args = ", ".join([arg.label for arg in expr.children])

    if isinstance(expr, rasp.Select):
        args += f", predicate={expr.predicate}"
    elif isinstance(expr, (rasp.Map, rasp.SequenceMap)):
        args = f"{expr.f}, " + args
    elif isinstance(expr, rasp.LinearSequenceMap):
        args = args + f", {expr.fst_fac}, {expr.snd_fac}"

    print_str = f"{expr.label} = {type(expr).__name__}({args})" #            Type: {expr.annotations['type']}")
    if test_input is not None and not isinstance(expr, rasp.Select):
        print_str += f"    # output: {expr(test_input)}"
#        print_str = f"{expr(test_input)}" + " ~ " + print_str
    elif not isinstance(expr, rasp.Select):
        print_str += f"    # type: {expr.annotations['type']}"
    print(print_str)
    return None


def print_program(output: rasp.SOp, test_input=None):
    graph = rasp_to_graph.extract_rasp_graph(output)
    sorted_nodes = list(nx.topological_sort(graph.graph))

    for node_id in sorted_nodes:
        if node_id in ["tokens", "indices"]:
            continue

        expr = graph.graph.nodes[node_id]["EXPR"]
        print_expr(expr, test_input=test_input)