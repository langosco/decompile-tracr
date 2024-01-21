from rasp_generator import map_primitives


ops = [
    "Map",
    "SequenceMap",
    "LinearSequenceMap",
    "Select",
    "Aggregate"
]

encodings = [
    "categorical", 
    "numerical"
]

sop_variables = [
    f"SOp_{i}" for i in range(20)
]

selector_variables = [
    f"Selector_{i}" for i in range(10)
]

maps = [
    repr(fn) for fn in map_primitives.ALL_FNS
]

comparisons = [
    comparison.name for comparison in map_primitives.COMPARISONS
]

inputs = [
    "tokens",
    "indices",
]

vocab = \
    ops + \
    encodings + \
    sop_variables + \
    selector_variables + \
    maps + \
    comparisons + \
    inputs + \
    ["SEP"]
