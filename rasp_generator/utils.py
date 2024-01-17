from tracr.rasp import rasp
from tracr.compiler import rasp_to_graph
from tracr.compiler.validating import validate
import networkx as nx


def print_expr(expr: rasp.RASPExpr, test_input=None):
    """Print an annotated rasp expression in a human-readable format."""
    args = ", ".join([arg.label for arg in expr.children])

    if isinstance(expr, rasp.Select):
        args += f", predicate={expr.predicate}"
    elif isinstance(expr, rasp.LinearSequenceMap):
        args = args + f", {expr.fst_fac}, {expr.snd_fac}"
    elif isinstance(expr, (rasp.Map, rasp.SequenceMap)):
        args = f"{expr.f}, " + args

    print_str = f"{expr.label} = {type(expr).__name__}({args})" #            Type: {expr.annotations['type']}")
    if test_input is not None and not isinstance(expr, rasp.Select):
        print_str += f"    # output: {expr(test_input)}"
#        print_str = f"{expr(test_input)}" + " ~ " + print_str
    elif not isinstance(expr, rasp.Select):
        print_str += f"    # type: {expr.annotations['type']}"
    print(print_str)
    return None


def print_program(program: rasp.SOp, test_input=None):
    """Sort the nodes in a program topologically and print them in order."""
    graph = rasp_to_graph.extract_rasp_graph(program)
    sorted_nodes = list(nx.topological_sort(graph.graph))

    for node_id in sorted_nodes:
        if node_id in ["tokens", "indices"]:
            continue

        expr = graph.graph.nodes[node_id]["EXPR"]
        print_expr(expr, test_input=test_input)


def fraction_none(x: list):
    """Return the fraction of elements in x that are None."""
    return sum([1 for elem in x if elem is None]) / len(x)