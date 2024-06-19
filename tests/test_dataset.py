import pytest
import numpy as np
import h5py
import chex

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.tokenize import vocab
from decompile_tracr.dataset.config import load_config, DatasetConfig
from decompile_tracr.dataset import logger_config
from decompile_tracr.dataset.dataloading import load_dataset
from decompile_tracr.dataset.reconstruct import ModelFromParams
from decompile_tracr.dataset.make_dataset import make_dataset, merge
from decompile_tracr.dataset.config import DatasetConfig
from decompile_tracr.dataset.config import default_data_dir
from decompile_tracr.dataset.compile import compile_
from decompile_tracr.compress.metrics import Embed, Unembed
from decompile_tracr.compress.utils import AssembledModelInfo
from decompile_tracr.dataset import data_utils


# Load from default dataset and do some sanity checks.


rng = np.random.default_rng()
logger = logger_config.setup_logger(__name__)
DATASETS = [
    "default",
]


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_splits(dataset_name: str):
    config = load_config(dataset_name)
    pth = config.paths.dataset
    with h5py.File(pth, "r") as f:
        assert "train" in f, f"No train split found in {pth}"
        assert "val" in f, f"No validation split found in {pth}"


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_deduplication(dataset_name: str):
    config = load_config(dataset_name)
    train, val, test = _load_tokens(config)
    ltrain, lval, ltest = len(train), len(val), len(test)

    assert ltrain > 0, f"Train split is empty for dataset {dataset_name}."
    assert lval > 0, f"Validation split is empty for dataset {dataset_name}."

    train = set([tuple(x.tolist()) for x in train])
    val = set([tuple(x.tolist()) for x in val])
    test = set([tuple(x.tolist()) for x in test])

    intersection = set.intersection(train, val)
    assert intersection == set(), (
        f"Found {len(intersection)} duplicates between train "
        f"and validation splits for dataset {dataset_name}."
    )
    
    intersection = set.intersection(train, test)
    assert intersection == set(), (
        f"Found {len(intersection)} duplicates between train "
        f"and test splits for dataset {dataset_name}."
    )

    intersection = set.intersection(val, test)
    assert intersection == set(), (
        f"Found {len(intersection)} duplicates between validation "
        f"and test splits for dataset {dataset_name}."
    )

    if len(train) < ltrain:
        logger.info(f"Found {ltrain - len(train)} duplicates in train split.")
    if len(val) < lval:
        logger.info(f"Found {lval - len(val)} duplicates in validation split.")
    if len(test) < ltest:
        logger.info(f"Found {ltest - len(test)} duplicates in test split.")


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_tokenization(dataset_name: str):
    """De-tokenize, then re-tokenize. Ensure that the re-tokenized
    program is the same as the original."""
    config = load_config(dataset_name)
    train, val, test = _load_tokens(config, n=10_000)
    tokens = train
    if val != []:
        tokens = np.concatenate([train, val])
    if test != []:
        tokens = np.concatenate([tokens, test])

    for x in tokens:
        x = x.tolist()
        x = x[:x.index(vocab.pad_id)]
        program = tokenizer.detokenize(x)
        retokenized = tokenizer.tokenize(program)
        assert x == retokenized, (
            f"tokenize(detokenize(x)) does not equal x."
            f"Original: {x}\n"
            f"Retokenized: {retokenized}"
        )


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_unflatten(dataset_name: str):
    """Load weights, unflatten, and compare to original params.
    Then check outputs of the reconstructed model.
    """
    config = load_config(dataset_name)
    data = load_dataset(config.paths.dataset, ndata=5)
    for i, w in enumerate(data['weights']):
        x = {k: v[i] for k, v in data.items()}
        params = data_utils.unflatten_params(
            w, sizes=x['layer_idx'], d_model=x['d_model'])

        # recompile and compare
        m = compile_(tokenizer.detokenize(data['tokens'][i]))
        info = AssembledModelInfo(model=m)
        chex.assert_trees_all_equal(params, m.params)
        assert info.num_heads == x['n_heads']
        assert info.num_layers == x['n_layers']
        assert info.d_model == x['d_model']

        # run forward pass through reconstructed and recompiled models
        # recompiled model:
        input_ = ["compiler_bos", 4, 2, 3]
        out_compiled = m.apply(input_).decoded

        # reconstructed model:
        embed, unembed = Embed(assembled_model=m), Unembed(assembled_model=m)
        model = ModelFromParams(params, num_heads=x['n_heads'])
        input_tokens = np.array([m.input_encoder.encode(input_)])
        out_reconstr = model.from_tokens.apply(params, input_tokens)
        unembedded = unembed(out_reconstr.output)
        decoded = m.output_encoder.decode(np.squeeze(unembedded).tolist())

        assert out_compiled[1:] == decoded[1:], (
            f"Reconstructed model {i} output does not match original output. "
            f"Expected: {out_compiled[1:]}, got: {decoded[1:]}"
        )


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_datapoint_attributes(dataset_name: str):
    KEYS = set([
        'categorical_output', 'd_model', 'layer_idx', 'n_heads', 
        'n_layers', 'n_sops', 'tokens', 'weights',
    ])
    config = load_config(dataset_name)
    data = load_dataset(config.paths.dataset, ndata=0)
    assert set(data.keys()) == KEYS, (
        f"Expected keys {KEYS}, got {set(data.keys())}."
    )


def _load_tokens(config: DatasetConfig, n: int = -1):
    path = config.paths.dataset
    with h5py.File(path, "r") as f:
        train = f['train/tokens'][:n]

        try:
            val = f['val/tokens'][:n]
        except KeyError:
            val = []

        try:
            test = f['test/tokens'][:n]
        except KeyError:
            test = []

    return train, val, test