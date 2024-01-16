

# RASP operations need to satisfy certain constraints. For example, the
# input to a selector cannot contain None values. These constraints are
# used during sampling to ensure that the sampled operations are valid.


def does_not_contain_none(sop, test_input):
    """The input SOp to rasp.Selector is not allowed to have None
    values."""
    seq = sop(test_input)
    return None not in set(seq)


