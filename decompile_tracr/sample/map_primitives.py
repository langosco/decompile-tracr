# The RASP operations rasp.Map, rasp.SequenceMap, and rasp.LinearSequenceMap
# apply a function elementwise to one or two SOps (sequence inputs).
# For example, rasp.Map(lambda x: x+1, rasp.tokens) will add 1 to each
# element of the input sequence.
# When we sample a Map, SequenceMap, or LinearSequenceMap, we need to 
# determine what function to apply. This depends on the type of the 
# input SOp. We currently distinguish three "types": categorical, float, and bool.
# For example, a function that maps categorical x categorical --> bool might be
# lambda x, y: x == y.

# Tracr natively uses two types: categorical and numerical. The correspondence
# to our types is as follows:

# Tracr       | Ours
# --------------------
# categorical | categorical
# numerical   | float, bool  (a bool is a numerical that only takes values 0 or 1)


# CONSTRAINTS
# - numerical values are not allowed to be negative (because of ReLU)


from tracr.rasp import rasp


class FunctionWithRepr:
    """Minimal wrapper around a function that allows to 
    represent it as a string."""
    def __init__(self, fn_str: str):
        """
        fn_str: function in form of eval-able string, e.g. 'lambda x: x+1'."""
        self.fn_str = fn_str
        self.fn = eval(fn_str)

    def __repr__(self):
        return self.fn_str
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def compose(self, other: "FunctionWithRepr"):
        """Compose two functions."""
        return FunctionWithRepr(f"(lambda x: {self.fn_str})(({other.fn_str})(x))")
    
    def __eq__(self, other):
        return self.fn_str == other.fn_str


TYPES = [
    "bool",
    "float",
    "categorical",
]


# Tracr only supports categorical inputs.
# These are integers, but they'll be treated as categorical.
VOCAB_SIZE = 10
VOCAB = list(range(VOCAB_SIZE))


# categorical --> categorical
# (note that range == domain. might change this later)
#CAT_TO_CAT = [FunctionWithRepr(f"lambda x: x + {n % VOCAB_SIZE}") for n in VOCAB]  # TODO remvoe
CAT_TO_CAT = [
    FunctionWithRepr(f"lambda x: x + 1"),
    FunctionWithRepr(f"lambda x: x - 1"),
]

# categorical --> float
CAT_TO_FLOAT = [  # note that negative floats are not allowed (bc of ReLUs)
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: x + 0.5"),
    FunctionWithRepr("lambda x: x + 1"),
    FunctionWithRepr("lambda x: x * 2"),
#    FunctionWithRepr("lambda x: x / 2"),
#    FunctionWithRepr("lambda x: x ** 2"),
]

# categorical --> bool
CAT_TO_BOOL = (
    [FunctionWithRepr(f"lambda x: x == {n}") for n in (0,1,2,3)] +
    [FunctionWithRepr(f"lambda x: x != {n}") for n in (2,3)] +
    [FunctionWithRepr(f"lambda x: x > {n}") for n in (2,3)] +
    [FunctionWithRepr(f"lambda x: x < {n}") for n in (2,3)]
)


# bool --> bool
# (maybe remove? not very interesting)
BOOL_TO_BOOL = [
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: not x"),
]

# bool --> float and bool --> categorical
# BOOL_TO_FLOAT = BOOL_TO_BOOL
# BOOL_TO_CAT = BOOL_TO_BOOL


# float --> float
# FLOAT_TO_FLOAT = CAT_TO_FLOAT

# float --> bool
# TODO: consider using more thresholds
threshold = 3.5
FLOAT_TO_BOOL = [
    FunctionWithRepr(f"lambda x: x > {threshold}"),
    FunctionWithRepr(f"lambda x: x < {threshold}"),
    FunctionWithRepr(f"lambda x: x == 0"),
]


# float --> categorical
FLOAT_TO_CAT = [
    FunctionWithRepr("lambda x: x"),
    FunctionWithRepr("lambda x: int(x)")
]


FREQ_TO_BOOL = [  # freq = outputs from numerical Aggregate
    FunctionWithRepr(f"lambda x: x > 0.5"),
]


# rasp.SequenceMap only supports categorical --> categorical
NONLINEAR_SEQMAP_FNS = [
    FunctionWithRepr("lambda x, y: x * y"),
    FunctionWithRepr(f"lambda x, y: x + y % {VOCAB_SIZE}"),
#    FunctionWithRepr(f"lambda x, y: x - y % {VOCAB_SIZE}"),
    FunctionWithRepr(f"lambda x, y: x - y"),  # remove?
    FunctionWithRepr(f"lambda x, y: x or y"),
    FunctionWithRepr(f"lambda x, y: x and y"),
#    FunctionWithRepr("lambda x, y: x/y"),  # need to avoid y = 0
#    FunctionWithRepr("lambda x, y: x**1/y"),  # need to avoid y = 0 and complex numbers
]


LINEAR_SEQUENCE_MAP_WEIGHTS = [-3, -2, -1, 1, 2, 3]


ALL_FNS = (
    BOOL_TO_BOOL +
    FLOAT_TO_BOOL + FLOAT_TO_CAT +
    CAT_TO_BOOL + CAT_TO_FLOAT + CAT_TO_CAT + 
    FREQ_TO_BOOL + NONLINEAR_SEQMAP_FNS
)


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


FUNCTIONS_BY_SIGNATURE = {
    "bool --> bool": BOOL_TO_BOOL,
    "bool --> float": BOOL_TO_BOOL,
    "bool --> categorical": BOOL_TO_BOOL,
    "float --> bool": FLOAT_TO_BOOL,
    "float --> float": CAT_TO_FLOAT,
    "float --> categorical": FLOAT_TO_CAT,
    "categorical --> bool": CAT_TO_BOOL,
    "categorical --> float": CAT_TO_FLOAT,
    "categorical --> categorical": CAT_TO_CAT,
    "freq --> bool": FREQ_TO_BOOL,
    "freq --> float": CAT_TO_FLOAT,
    "freq --> categorical": FLOAT_TO_CAT,
}


def get_map_fn(rng, input_type: str, output_types=TYPES) -> tuple[callable, str]:
    """
    Randomly determine an output domain (ie type), then sample a function
    from the set of functions that map from input_type --> output_type.
    """
    output_type = rng.choice(output_types)
    fn_scope = FUNCTIONS_BY_SIGNATURE[f"{input_type} --> {output_type}"]
    return rng.choice(fn_scope), output_type
