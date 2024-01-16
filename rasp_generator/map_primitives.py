# The RASP operations rasp.Map, rasp.SequenceMap, and rasp.LinearSequenceMap
# apply a function elementwise to one or two SOps (sequence inputs).
# For example, rasp.Map(lambda x: x+1, rasp.tokens) will add 1 to each
# element of the input sequence.
# When we sample an operation, we need to determine what function to apply.
# This depends on the type of the input SOp. We currently distinguish 
# three "types": categorical, float, and bool.
# For example, a function that maps categorical --> bool might be
# lambda x, y: x == y.

# Tracr natively uses two types: categorical and numerical. The correspondence
# to our types is as follows:

# Tracr       | Ours
# --------------------
# categorical | categorical
# numerical   | float, bool

# CONSTRAINTS
# - numerical values are not allowed to be negative (because of ReLU)


from tracr.rasp import rasp
import numpy as np


TYPES = [
    "bool",
    "float",
    "categorical",
]


def get_map_fn(input_type: str) -> (callable, str):
    """Given the type of the input SOp, return a function that is
    valid for that type and its output type."""
    if input_type == "bool":
        output_type = np.random.choice(["bool", "categorical"])
        fns = BOOL_TO_BOOL if output_type == "bool" else BOOL_TO_CAT
    elif input_type == "float":
        output_type = np.random.choice(TYPES)
        if output_type == "bool":
            fns = FLOAT_TO_BOOL
        elif output_type == "float":
            fns = FLOAT_TO_FLOAT
        elif output_type == "categorical":
            fns = FLOAT_TO_CAT
    elif input_type == "categorical":
        output_type = np.random.choice(TYPES)
        if output_type == "bool":
            fns = CAT_TO_BOOL
        elif output_type == "float":
            fns = CAT_TO_FLOAT
        elif output_type == "categorical":
            fns = CAT_TO_CAT
    else:
        raise ValueError(f"Got sop annotated with unkown type {input_type}.")

    return np.random.choice(fns), output_type


# Tracr only supports categorical inputs.
# These are integers, but they'll be treated as categorical.
VOCAB_SIZE = 5
VOCAB = list(range(VOCAB_SIZE))


class FunctionWithRepr:
    """Minimal wrapper around a function that allows us
    to represent it as a string."""
    def __init__(self, fn_str: str):
        """
        fn_str: function in form of eval-able string, e.g. 'lambda x: x+1'."""
        self.fn_str = fn_str

    def __repr__(self):
        return self.fn_str
    
    def __call__(self, *args, **kwargs):
        return eval(self.fn_str)(*args, **kwargs)


# categorical --> categorical
# (note that range == domain. might change this later)
CAT_TO_CAT = [FunctionWithRepr(f"lambda x: x + {n % VOCAB_SIZE}") for n in VOCAB]

# categorical --> float
CAT_TO_FLOAT = [  # note that negative floats are not allowed (bc of ReLUs)
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: x+0.5"),
    FunctionWithRepr("lambda x: x+1"),
    FunctionWithRepr("lambda x: x*2"),
    FunctionWithRepr("lambda x: x/2"),
    FunctionWithRepr("lambda x: x**2"),
    FunctionWithRepr("lambda x: x**0.5"),
]

# categorical --> bool
CAT_TO_BOOL = (
    [FunctionWithRepr(f"lambda x: x == {n}") for n in VOCAB] +
    [FunctionWithRepr(f"lambda x: x != {n}") for n in VOCAB] +
    [FunctionWithRepr(f"lambda x: x > {n}") for n in VOCAB] +
    [FunctionWithRepr(f"lambda x: x < {n}") for n in VOCAB]
)


# bool --> bool
# (maybe remove? not very interesting)
BOOL_TO_BOOL = [
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: not x"),
]

# bool --> float
pass


BOOL_TO_CAT = BOOL_TO_BOOL


# float --> float
FLOAT_TO_FLOAT = CAT_TO_FLOAT

# float --> bool
# (really here I should sample threshold)
threshold = 0
FLOAT_TO_BOOL = [
    FunctionWithRepr(f"lambda x: x > {threshold}"),
    FunctionWithRepr(f"lambda x: x < {threshold}"),
]

# float --> categorical
FLOAT_TO_CAT = [
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: int(x)")
]


# rasp.SequenceMap only supports categorical --> categorical
NONLINEAR_SEQMAP_FNS = [
    FunctionWithRepr("lambda x, y: x*y"),
#    FunctionWithRepr("lambda x, y: x/y"),  # need to avoid y = 0
#    FunctionWithRepr("lambda x, y: x**1/y"),  # need to avoid y = 0
]


COMPARISONS = [
    rasp.Comparison.EQ, 
#    rasp.Comparison.FALSE,
    rasp.Comparison.TRUE,
    rasp.Comparison.GEQ,
    rasp.Comparison.GT, 
    rasp.Comparison.LEQ,
    rasp.Comparison.LT, 
    rasp.Comparison.NEQ,
]
