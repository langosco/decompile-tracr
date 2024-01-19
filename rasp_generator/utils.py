from tracr.rasp import rasp
from tracr.compiler import rasp_to_graph
from tracr.compiler.validating import validate
import networkx as nx


class EmptyScopeError(Exception):
    pass


def annotate_type(sop: rasp.SOp, type: str):
    """Annotate a SOp with a type."""
    # important for compiler:
    if type in ["bool", "float"]:
        sop = rasp.numerical(sop)
    elif type in ["categorical"]:
        sop = rasp.categorical(sop)
    else:
        raise ValueError(f"Unknown type {type}.")
    
    # ignored by compiler but used by program sampler:
    sop = rasp.annotate(sop, type=type)
    return sop


def filter_by_type(sops: list[rasp.SOp], type: str = None):
    """Return the subset of SOps that are of a given type"""
    filtered = sops
    if type is not None:
        filtered = [sop for sop in filtered if sop.annotations["type"] == type]
    if len(filtered) == 0:
        raise EmptyScopeError(f"No SOps of type {type} in scope.")
    return filtered


def filter_by_constraints(sops: list[rasp.SOp], constraints: list[callable]):
    """Return the subset of SOps that satisfy a set of constraints.
    A constraint is a callable that takes a SOp and return a boolean."""
    filtered = sops
    for constraint in constraints:
        filtered = [v for v in filtered if constraint(v)]

    if len(filtered) == 0:
        raise EmptyScopeError("No SOps in scope satisfy constraints.")

    return filtered


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