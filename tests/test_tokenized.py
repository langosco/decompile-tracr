import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import dill as pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from pathlib import Path
import pytest

from rasp_tokenizer.data_utils import load_deduped, load_batches, save_deduped
from rasp_tokenizer.data_utils import process_data, split_dict_data
from rasp_tokenizer import paths

from rasp_tokenizer import tokenizer
from rasp_tokenizer import vocab



data = load_deduped(name="train")
tokens = [x["rasp_tok"] for x in data]
decoded = [tokenizer.decode(x) for x in tokens]


@pytest.mark.parametrize("ops", decoded)
def test_select_names(ops: list):
    """Test that selectors are always called Selector_{n}"""
    for idx, tok in enumerate(ops):
        assert isinstance(tok, str), f"Expected string, got {tok}"
        if tok == "Select":
            assert ops[idx-1].startswith("Selector_"), (
                f"Selector {tok} named {ops[idx-1]} instead of "
                "Selector_n."
            )


@pytest.mark.parametrize("ops", decoded)
def test_sop_names(ops: list):
    """Test that SOps are always called SOp_{n}"""
    for idx, tok in enumerate(ops):
        if tok in ["Map", "SequenceMap", "LinearSequenceMap", 
                   "Aggregate", "SelectorWidth"]:
            assert ops[idx-1] in ["numerical", "categorical"], (
                f"SOp {tok} preceded by {ops[idx-1]} instead of "
                "numerical or categorical."
            )

            assert ops[idx-2].startswith("SOp_"), (
                f"SOp {tok} named {ops[idx-1]} instead of "
                "SOp_n."
            )


@pytest.mark.parametrize("ops", decoded)
def test_bos_and_eos(ops: list):
    """Test that each sequence begins with BOS and ends with EOS"""
    assert ops[0] == vocab.BOS, f"Expected BOS, got {ops[0]}"
    assert ops[-1] == vocab.EOS, f"Expected EOS, got {ops[-1]}"