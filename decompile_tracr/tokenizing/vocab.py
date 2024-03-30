from decompile_tracr.sampling import map_primitives


ops = [
    "Map",
    "SequenceMap",
    "LinearSequenceMap",
    "SelectAggregate",
    "SelectorWidth",
]

encodings = [
    "categorical", 
    "numerical"
]

sop_variables = [
    f"sop_{i}" for i in range(25)
]

maps = [
    repr(fn) for fn in map_primitives.ALL_FNS
]

comparisons = [
    comparison.name for comparison in map_primitives.COMPARISONS
]

linear_sequence_map_weights = [
    str(x) for x in map_primitives.LINEAR_SEQUENCE_MAP_WEIGHTS]

inputs = [
    "tokens",
    "indices",
]


PAD = "PAD"
BOS = "BOS"
EOS = "EOS"
SEP = "SEP"


vocab = (
    [PAD, BOS, EOS,  SEP] +
    ops +
    encodings +
    sop_variables +
    maps +
    linear_sequence_map_weights +
    comparisons +
    inputs
)

vocab = tuple(sorted(list(set(vocab))))

size = len(vocab)
pad_id = vocab.index(PAD)
bos_id = vocab.index(BOS)
eos_id = vocab.index(EOS)
