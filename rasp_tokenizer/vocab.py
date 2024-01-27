from rasp_generator import map_primitives


ops = [
    "Map",
    "SequenceMap",
    "LinearSequenceMap",
    "Select",
    "Aggregate",
    "SelectorWidth",
]

encodings = [
    "categorical", 
    "numerical"
]

sop_variables = [
    f"SOp_{i}" for i in range(40)
]

selector_variables = [
    f"Selector_{i}" for i in range(20)
]

maps = [
    repr(fn) for fn in map_primitives.ALL_FNS
]

comparisons = [
    comparison.name for comparison in map_primitives.COMPARISONS
]

linear_sequence_map_weights = map_primitives.LINEAR_SEQUENCE_MAP_WEIGHTS

inputs = [
    "tokens",
    "indices",
]


vocab = (
    ["PAD", "START", "END"] +
    ops +
    encodings +
    sop_variables +
    selector_variables +
    maps +
    linear_sequence_map_weights +
    comparisons +
    inputs
)


size = len(vocab)
pad_id = vocab.index("PAD")
