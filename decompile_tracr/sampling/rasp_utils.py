import numpy as np
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
    if type is not None:
        mask = [1 if sop.annotations["type"] == type else 0 for sop in sops]
    else:
        mask = [1 for _ in sops]
    if len(mask) == 0:
        raise EmptyScopeError(f"Filter failed. No SOps of type {type} in scope.")
    return np.array(mask)


def filter_nones(sops: list[rasp.SOp], test_inputs: list[list]):
    mask = [1 if no_none_in_values(sop, test_inputs) else 0 for sop in sops]
    if len(mask) == 0:
        raise EmptyScopeError("Filter failed. All SOps contain None on some test input.")
    return np.array(mask)


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


def sample_test_input(rng, vocab={0,1,2,3,4}, max_seq_len=5, min_seq_len=1):  # to validate.py ?
    seq_len = rng.choice(range(min_seq_len, max_seq_len+1))
    return rng.choice(list(vocab), size=seq_len).tolist()


def count_sops(program: rasp.SOp):
    """Return the number of SOps in a program (don't count Selects).
    Note that this is not the same as the *depth* of a node in the graph,
    which is always smaller.
    """
    nodes = rasp_to_graph.extract_rasp_graph(program).graph.nodes
    sops = [not isinstance(nodes[x]['EXPR'], rasp.Select) for x in nodes]
    return sum(sops)


def is_equal(sop1: rasp.SOp, sop2: rasp.SOp, recursive=True,
             verbose=False):
    """Two rasp expressions are equal if
    1) they are the same op (eg both Maps)
    2) they have the same encoding
    3) they have the same args
    """
    if type(sop1) != type(sop2):
        return False
    elif sop1 is sop2:
        return True
    
    if isinstance(sop1, (rasp.TokensType, rasp.IndicesType)):
        return True

    def eq(x, y):
        if recursive:
            return is_equal(x, y, recursive=True, verbose=verbose)
        else:
            return x is y
    
    if isinstance(sop1, rasp.Map):
        out = (sop1.f == sop2.f and
               eq(sop1.inner, sop2.inner))
    elif isinstance(sop1, rasp.LinearSequenceMap):
        out = (sop1.fst_fac == sop2.fst_fac and
               sop1.snd_fac == sop2.snd_fac and
               eq(sop1.fst, sop2.fst) and
               eq(sop1.snd, sop2.snd))
    elif isinstance(sop1, rasp.SequenceMap):
        out = (sop1.f == sop2.f and
               eq(sop1.fst, sop2.fst) and
               eq(sop1.snd, sop2.snd))
    elif isinstance(sop1, rasp.Select):
        out = (eq(sop1.keys, sop2.keys) and
               eq(sop1.queries, sop2.queries) and
               sop1.predicate == sop2.predicate)
    elif isinstance(sop1, rasp.Aggregate):
        out = (eq(sop1.selector, sop2.selector) and 
               eq(sop1.sop, sop2.sop))
    elif isinstance(sop1, rasp.SelectorWidth):
        out = eq(sop1.selector, sop2.selector)
    else:
        raise ValueError(f"Unknown SOp type {type(sop1)}.")
    
    if verbose and not out:
        print(f"{sop1.label} != {sop2.label}")
    
    return out