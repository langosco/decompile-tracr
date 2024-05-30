import pytest
import numpy as np
import h5py

from decompile_tracr.tokenize import tokenizer
from decompile_tracr.tokenize import vocab
from decompile_tracr.dataset.config import load_config, DatasetConfig
from decompile_tracr.dataset import logger_config


# Load from default dataset and do some sanity checks.


rng = np.random.default_rng()
logger = logger_config.setup_logger(__name__)
DATASETS = [
    "small_compressed",
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
    tokens = np.concatenate([train, val, test])

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


def test_compilation():
    # this is hard because we can't easily load a compiled model
    # once we saved the weights.
    pass


def _load_tokens(config: DatasetConfig, n: int = -1):
    path = config.paths.dataset
    with h5py.File(path, "r") as f:
        train = f['train/tokens'][:n]
        val = f['val/tokens'][:n]
        try:
            test = f['test/tokens'][:n]
        except:
            logger.info(f"No test split found in {path}.")
            test = None
    return train, val, test