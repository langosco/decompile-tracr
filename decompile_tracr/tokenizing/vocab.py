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
    f"sop_{i:02d}" for i in range(10)
]
assert sorted(sop_variables) == sop_variables

maps = sorted(list(set([
    repr(fn) for fn in map_primitives.ALL_FNS
])))

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
EOO = "EOO"  # end of op
EOL = "EOL"  # end of layer


vocab = (
    [PAD, BOS, EOS, EOO, EOL] +
    ops +
    encodings +
    sop_variables +
    maps +
    linear_sequence_map_weights +
    comparisons +
    inputs
)

vocab = tuple(vocab)
assert len(vocab) == len(set(vocab)), "vocab has duplicates"

size = len(vocab)
pad_id = vocab.index(PAD)
bos_id = vocab.index(BOS)
eos_id = vocab.index(EOS)
eoo_id = vocab.index(EOO)
eol_id = vocab.index(EOL)
