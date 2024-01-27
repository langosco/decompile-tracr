import pickle
from collections import defaultdict

import numpy as np
import chex

from meta_transformer import preprocessing
import meta_transformer.utils

from rasp_tokenizer import vocab
from rasp_tokenizer import paths
from rasp_tokenizer.logger_config import setup_logger


logger = setup_logger(__name__)


def load_data():
    train_path = paths.data_dir / "train" / "data.pkl"
    test_path = paths.data_dir / "test" / "50.pkl"

    logger.info(f"Loading train/val data from {train_path}.")
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    
    logger.info(f"Loading test data from {test_path}.")
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    return train_data, test_data


def pad_to(x: np.ndarray, max_len: int, pad_value: int = 0):
    """Pad a 1D array to a given length. Not jittable."""
    x = np.array(x)
    assert len(x) <= max_len
    chex.assert_rank(x, 1)
    return np.pad(x, (0, max_len - len(x)), constant_values=pad_value)


def process_single_datapoint(
        x: dict[str, tuple],
        d_model: int,
        max_rasp_len: int = 32,
        max_weights_len: int = 8192,
    ):
    """Process a single datapoint for model input.
    1) Rasp tokens: pad to max rasp length.
    2) Weights: pad to max_weights_len, then chunk.
    """
    if len(x['rasp']) > max_rasp_len:
        raise ValueError(f"Program length ({len(x['rasp'])}) exceeds "
                         f"max program length ({max_rasp_len}).")
    elif len(x['weights']) > max_weights_len:
        raise ValueError(f"Weights length ({len(x['weights'])}) exceeds "
                         f"max weights length ({max_weights_len}).")
    
    w_mean = np.mean(x['weights'])
    if np.abs(w_mean) > 1.5:
        return None
    
    weights = pad_to(x['weights'], max_weights_len)
    weights = preprocessing.pad_and_chunk(weights, d_model)  # (n_chunks, d_model)
    return {
        "rasp": pad_to(x['rasp'], max_rasp_len, pad_value=vocab.pad_id),
        "weights": weights,
        "id": x['id'],
    }


def process_data(
        data: list[list[dict]],
        d_model: int,
        max_rasp_len: int = 32,
        max_weights_len: int = 8192,
    ):
    n = len(data)
    out = defaultdict(list)
    for x in data:
        x_proc = process_single_datapoint(
            x, d_model, max_rasp_len, max_weights_len)

        if x_proc is None:
            continue

        for k, v in x_proc.items():
            out[k].append(v)

    out = {k: np.stack(v) for k, v in out.items()}
    logger.info(f"Filtered out {n - len(out['rasp'])} datapoints "
                f"({round(100 * (n - len(out['rasp'])) / n, 2)}%). "
                f"Total remaining: {len(out['rasp'])}.")
    # clip weights
    out["weights"] = np.clip(out["weights"], -1, 1)
    chex.assert_shape(out["rasp"], (None, max_rasp_len))
    chex.assert_shape(out["weights"], (None, max_weights_len//d_model, d_model))
    assert len(out["rasp"]) == len(out["weights"])
    return out


def split_dict_data(data: dict, val_ratio: float = 0.1):
    """Split a dictionary of data arrays into train and validation sets.
    """
    train, val = {}, {}
    for k, v in data.items():
        train[k], val[k] = meta_transformer.utils.split_data(
            v, val_ratio)
    return train, val