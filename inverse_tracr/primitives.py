import os
from tracr.rasp import rasp
from tracr.compiler import compiling
import numpy as np
from tracr.compiler.validating import validate


# Tracr only supports categorical inputs.
# These are integers, but they'll be treated as categorical.
VOCAB_SIZE = 10
VOCAB = list(range(VOCAB_SIZE))


def make_length():
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    return rasp.SelectorWidth(all_true_selector)


# categorical --> categorical
# (for now all internal variables are in the same vocab as the input. might change this later)
CATEGORICAL_MAP_FNS = [lambda x: x + n % VOCAB_SIZE for n in range(VOCAB_SIZE)]

# categorical --> float
CATEGORICAL_TO_FLOAT_MAP_FNS = [
    lambda x: x,
    lambda x: x+1,
    lambda x: x-1,
    lambda x: x*2,
    lambda x: x/2,
    lambda x: x**2,
    lambda x: x**0.5,
]

# categorical --> bool
CATEGORICAL_TO_BOOL_MAP_FNS = (
    [lambda x: x == n for n in range(VOCAB_SIZE)] +
    [lambda x: x != n for n in range(VOCAB_SIZE)] +
    [lambda x: x > n for n in range(VOCAB_SIZE)] +
    [lambda x: x >= n for n in range(VOCAB_SIZE)] +
    [lambda x: x < n for n in range(VOCAB_SIZE)] +
    [lambda x: x <= n for n in range(VOCAB_SIZE)]
)


# bool --> bool
# (maybe remove? not very interesting)
BOOL_MAP_FNS = [
    lambda x: x,
    lambda x: not x,
]

# bool --> float
pass

# bool --> categorical
BOOL_TO_CATEGORICAL_MAP_FNS = (
    lambda x: x,
    lambda x: not x,
)


# float --> float
FLOAT_MAP_FNS = CATEGORICAL_TO_FLOAT_MAP_FNS

# float --> bool
# (really here I should sample threshold)
threshold = 0
FLOAT_TO_BOOL_MAP_FNS = [
    lambda x: x > threshold,
    lambda x: x >= threshold,
    lambda x: x < threshold,
    lambda x: x <= threshold,
    lambda x: x == threshold,
    lambda x: x != threshold,
]

# float --> categorical
FLOAT_TO_CAT_MAP_FNS = [
    lambda x: x
]






NONLINEAR_SEQMAP_FNS = [
    lambda x, y: x,
    lambda x, y: y,
    lambda x, y: x+y,
    lambda x, y: x-y,
    lambda x, y: x*y,
    lambda x, y: x/y,
    lambda x, y: x**y,
    lambda x, y: x**1/y,
]

PREDICATES = [
    rasp.Comparison.EQ, 
    rasp.Comparison.FALSE,
    rasp.Comparison.TRUE,
    rasp.Comparison.GEQ,
    rasp.Comparison.GT, 
    rasp.Comparison.LEQ,
    rasp.Comparison.LT, 
    rasp.Comparison.NEQ,
]
