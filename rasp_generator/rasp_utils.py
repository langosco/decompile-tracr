from typing import Optional
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
        raise EmptyScopeError(f"Filter failed. No SOps of type {type} in scope.")
    return filtered


def filter_by_constraints(
        sops: list[rasp.SOp], 
        constraints: list[callable],
        constraints_name: Optional[str] = None):
    """Return the subset of SOps that satisfy a set of constraints.
    A constraint is a callable that takes a SOp and return a boolean."""
    filtered = sops
    for constraint in constraints:
        filtered = [v for v in filtered if constraint(v)]

    if len(filtered) == 0:
        err_msg = "Filter failed. No SOps in scope satisfy constraints."
        if constraints_name is not None:
            err_msg += f" Constraints: {constraints_name}."
        raise EmptyScopeError(err_msg)

    return filtered


def no_none_in_values(sop: rasp.SOp, test_inputs: list[list]):
    """Return True if the SOp never contains None on any of the test inputs."""
    values = set()
    for x in test_inputs:
        values = values.union(sop(x))
    return not None in values


def print_expr(expr: rasp.RASPExpr, test_input=None, full=False):
    """Print an annotated rasp expression in a human-readable format."""
    args = ", ".join([arg.label for arg in expr.children])

    if isinstance(expr, rasp.Select):
        args += f", predicate={expr.predicate}"
    elif isinstance(expr, rasp.LinearSequenceMap):
        args = args + f", {expr.fst_fac}, {expr.snd_fac}"
    elif isinstance(expr, (rasp.Map, rasp.SequenceMap)):
        args = f"{expr.f}, " + args

    print_str = f"{type(expr).__name__}({args})"

    if full and not isinstance(expr, rasp.Select):
        enc = expr.annotations["encoding"].value
        print_str = f"rasp.{enc}({print_str})"

    print_str = f"{expr.label} = {print_str}"


    if test_input is not None and not isinstance(expr, rasp.Select):
        print_str += f"    # output: {expr(test_input)}"
#        print_str = f"{expr(test_input)}" + " ~ " + print_str
    elif not isinstance(expr, rasp.Select) and not full:
        try:
            print_str += f"    # type: {expr.annotations['type']}"
        except KeyError:
            print_str += f"    # type: {expr.annotations['encoding']}"
    
    print(print_str)
    return None


def print_program(program: rasp.SOp, test_input=None,
                  full=False):
    """Sort the nodes in a program topologically and 
    print them in order. if full, print the full program
    in valid RASP syntax."""
    graph = rasp_to_graph.extract_rasp_graph(program)
    sorted_nodes = list(nx.topological_sort(graph.graph))

    for node_id in sorted_nodes:
        if node_id in ["tokens", "indices"]:
            continue

        expr = graph.graph.nodes[node_id]["EXPR"]
        print_expr(expr, test_input=test_input, full=full)


def fraction_none(x: list):
    """Return the fraction of elements in x that are None."""
    return sum([1 for elem in x if elem is None]) / len(x)


def sample_test_input(rng, vocab={0,1,2,3,4}, max_seq_len=5):  # to validate.py ?
    seq_len = rng.choice(range(1, max_seq_len+1))
    return rng.choice(list(vocab), size=seq_len).tolist()


def count_sops(program: rasp.SOp):
    """Return the number of SOps in a program (don't count Selects).
    Note that this is not the same as the *depth* of a node in the graph,
    which is always smaller.
    """
    nodes = rasp_to_graph.extract_rasp_graph(program).graph.nodes
    sops = [not isinstance(nodes[x]['EXPR'], rasp.Select) for x in nodes]
    return sum(sops)


def is_equal(sop1: rasp.SOp, sop2: rasp.SOp):
    """Two rasp expressions are equal if
    1) they are of the same type,
    2) they are annotated with the same variable type (bool, 
    float, categorical),
    3) they have the same args
    """
    if type(sop1) != type(sop2):
        return False
    elif sop1.annotations["type"] != sop2.annotations["type"]:
        return False
    elif sop1 is sop2:
        return True
    
    if isinstance(sop1, rasp.Map):
        return (sop1.f == sop2.f and
                sop1.inner is sop2.inner)
    elif isinstance(sop1, rasp.SequenceMap):
        return (sop1.f == sop2.f and
                sop1.fst is sop2.fst and
                sop1.snd is sop2.snd)
    elif isinstance(sop1, rasp.LinearSequenceMap):
        return (sop1.fst_fac == sop2.fst_fac and
                sop1.snd_fac == sop2.snd_fac and
                sop1.fst is sop2.fst and
                sop1.snd is sop2.snd)
    elif isinstance(sop1, rasp.Select):
        return (sop1.keys is sop2.keys and
                sop1.queries is sop2.queries and
                sop1.predicate is sop2.predicate)
    elif isinstance(sop1, rasp.Aggregate):
        return (sop1.selector is sop2.selector and
                sop1.sop is sop2.sop)
    elif isinstance(sop1, rasp.SelectorWidth):
        return sop1.selector is sop2.selector
    else:
        raise ValueError(f"Unknown SOp type {type(sop1)}.")
