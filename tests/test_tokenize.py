import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import networkx as nx

from tracr.rasp import rasp
from tracr.compiler import rasp_to_graph

from decompile_tracr.sampling import rasp_utils
from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.tokenizing import vocab
from decompile_tracr.dataset import lib


SAMPLE_PROGRAMS = lib.examples
SAMPLE_PROGRAMS_TOKENIZED = [tokenizer.tokenize(x) for x in SAMPLE_PROGRAMS]


@pytest.mark.parametrize("program", SAMPLE_PROGRAMS)
def test_round_trip(program: rasp.SOp):
    """Tokenize, then detokenize"""
    tokens = tokenizer.tokenize(program)
    reconstructed = tokenizer.detokenize(tokens)

    assert rasp_utils.is_equal(program, reconstructed, recursive=True), (
        f"Detokenized does not equal original.")
    
    retokenized = tokenizer.tokenize(reconstructed)
    assert tokens == retokenized, (
        f"tokenize(reconstructed) does not equal tokenize(original).")


@pytest.mark.parametrize("tokens", SAMPLE_PROGRAMS_TOKENIZED)
def test_sop_names(tokens: list):
    """
    Test that ops are always preceded by an
    encoding and a variable called SOp_{n}
    """
    decoded = tokenizer.decode(tokens)
    for idx, tok in enumerate(decoded):
        if tok in ["Map", "SequenceMap", "LinearSequenceMap", 
                   "Aggregate", "SelectorWidth"]:
            assert decoded[idx-1] in ["numerical", "categorical"], (
                f"SOp {tok} preceded by '{decoded[idx-1]}' instead of "
                "numerical or categorical."
            )

            assert decoded[idx-2].startswith("sop_"), (
                f"SOp {tok} named '{decoded[idx-2]}' instead of "
                "sop_n."
            )
    
    # todo: test that variable names are assigned correctly
#    program = tokenizer.detokenize(tokens)
#    graph = rasp_to_graph.extract_rasp_graph(program)
#    sop_names_sorted = list(nx.topological_sort(graph.graph))




@pytest.mark.parametrize("tokens", SAMPLE_PROGRAMS_TOKENIZED)
def test_bos_and_eos(tokens: list):
    """Test that each sequence begins with BOS and ends with EOS"""
    decoded = tokenizer.decode(tokens)
    assert decoded[0] == vocab.BOS, f"Expected BOS, got {decoded[0]}"
    assert decoded[-1] == vocab.EOS, f"Expected EOS, got {decoded[-1]}"