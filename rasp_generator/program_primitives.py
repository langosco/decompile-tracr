# Desc: Contains program primitives, that is small programs that
# can be sampled from to create larger programs.
# That is, instead of sampling just from the basic RASP operations,
# we can sample from a richer set of primitives.
# (Currently we don't support this, but it's a good idea to have)


import os
from tracr.rasp import rasp
from tracr.compiler import compiling
import numpy as np
from tracr.compiler.validating import validate

def make_length():
    all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    return rasp.SelectorWidth(all_true_selector).named("length")


def reverse_selector() -> rasp.Selector:
    length = make_length()
    reversed_indices = (length - rasp.indices - 1).named("reversed_indices")
    return rasp.Select(rasp.indices, reversed_indices, rasp.Comparison.EQ)



# sel = Select(sop1, sop2, predicate)
# sop = Aggregate(sel, sop_agg)
# rewrite:
# sop = SelectAggregate(sop1, sop2, predicate, sop_agg)