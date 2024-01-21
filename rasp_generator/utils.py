from typing import Optional
from tracr.rasp import rasp
from tracr.compiler import rasp_to_graph
from tracr.compiler.validating import validate
import networkx as nx


class EmptyScopeError(Exception):
    pass


class FunctionWithRepr:
    """Minimal wrapper around a function that allows to 
    represent it as a string."""
    def __init__(self, fn_str: str):
        """
        fn_str: function in form of eval-able string, e.g. 'lambda x: x+1'."""
        self.fn_str = fn_str

    def __repr__(self):
        return self.fn_str
    
    def __call__(self, *args, **kwargs):
        return eval(self.fn_str)(*args, **kwargs)
    
    def compose(self, other: "FunctionWithRepr"):
        """Compose two functions."""
        return FunctionWithRepr(f"(lambda x: {self.fn_str})(({other.fn_str})(x))")


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
        try:
            print_str += f"    # type: {expr.annotations['type']}"
        except KeyError:
            print_str += f"    # type: {expr.annotations['encoding']}"
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


def sample_test_input(rng, vocab={0,1,2,3,4}, max_seq_len=5):
    seq_len = rng.choice(range(1, max_seq_len+1))
    return rng.choice(list(vocab), size=seq_len).tolist()


import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=5):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# # USAGE
# tracemalloc.start()
#
# counts = Counter()
# fname = '/usr/share/dict/american-english'
# with open(fname) as words:
#     words = list(words)
#     for word in words:
#         prefix = word[:3]
#         counts[prefix] += 1
# print('Top prefixes:', counts.most_common(3))
#
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)