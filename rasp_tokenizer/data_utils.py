import os
import dill as pickle
from collections import defaultdict

import numpy as np
import chex

from meta_transformer import preprocessing
import meta_transformer.utils

from rasp_tokenizer import vocab
from rasp_tokenizer import paths
from rasp_tokenizer.logger_config import setup_logger


logger = setup_logger(__name__)


def load_batch(filename: str) -> list[list[dict]]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_batches(loaddir = None) -> list[dict]:
    """
    Load batches created by scripts/generate_data.py
    and merge into a single list. 
    Assume: programs are not deduplicated across
    batches.
    """
    path = paths.data_dir / "batches"
    if loaddir is not None:
        path = path / loaddir
    data = []
    for entry in os.scandir(path):
        if entry.name.endswith(".pkl"):
            data.extend(load_batch(entry.path))
    return data


def dedupe(data: list[dict]) -> list[dict]:
    """Deduplicate programs by RASP string.
    Assume data is a list as returned by load_batches().
    """
    all_rasp = set()
    deduped = []
    for program in data:
        rasp_str = tuple(
            x['rasp_tok'] for x in program['weights_and_tokens']
        )

        if rasp_str in all_rasp:
            continue  # found duplicate
        else:
            all_rasp.add(rasp_str)
            deduped.append(program)
    logger.info(f"Deduplicated: {len(deduped)} programs.")
    logger.info(f"Removed: {len(data) - len(deduped)} programs. "
                f"({100*(len(data) - len(deduped)) / len(data)}%)")
    return deduped


def save_deduped(data: list[dict], savedir = "train"):
    """Save data after deduplication."""
    path = paths.data_dir / "deduped" 
    if savedir is not None:
        path = path / savedir
    os.makedirs(path, exist_ok=True)
    path = path / "data.pkl"
    logger.info(f"Saving deduplicated programs to {path}.")
    with open(path, "xb") as f:
        pickle.dump(data, f)


def load_deduped(name="train", keep_model_and_program=False):
    path = paths.data_dir / "deduped" / name / "data.pkl"
    logger.info(f"Loading data from {path}.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not keep_model_and_program:
        data = flatten(data)
    return data


def load_data_for_training():
    """Load train and test data ready for processing."""
    train = load_deduped("train")
    test = load_deduped("test")
    return train, test


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
    if len(x['rasp_tok']) > max_rasp_len:
        raise ValueError(f"Program length ({len(x['rasp_tok'])}) exceeds "
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
        "rasp_tok": pad_to(x['rasp_tok'], max_rasp_len, pad_value=vocab.pad_id),
        "weights": weights,
        "program_id": x['program_id'],
    }


def to_int(array: np.ndarray):
    int_arr = array.astype(np.int32)
    assert np.allclose(array, int_arr), f"Array is not integer: {array}"
    return int_arr


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
    out = {k: to_int(v) if k in ("rasp_tok", "program_id") else v 
           for k, v in out.items()}
    logger.info(f"Filtered out {n - len(out['rasp_tok'])} datapoints "
                f"({round(100 * (n - len(out['rasp_tok'])) / n, 2)}%). "
                f"Total remaining: {len(out['rasp_tok'])}.")
    # clip weights
    out["weights"] = np.clip(out["weights"], -100, 100)
    chex.assert_shape(out["rasp_tok"], (None, max_rasp_len))
    chex.assert_shape(out["weights"], (None, max_weights_len//d_model, d_model))
    assert len(out["rasp_tok"]) == len(out["weights"])
    return out


def split_dict_data(data: dict, val_ratio: float = 0.1):
    """Split a dictionary of data arrays into train and validation sets.
    """
    train, val = {}, {}
    for k, v in data.items():
        train[k], val[k] = meta_transformer.utils.split_data(
            v, val_ratio)
    return train, val


def flatten(data: list[dict]) -> list[dict]:
    """
    Convert the list of models to a list of layers. Throw away
    model-level information like the rasp program and the compiled
    model.
    """
    out = []
    for program_id, program in enumerate(data):
        for layer in program["weights_and_tokens"]:
            layer['program_id'] = program_id
            out.append(layer)
    return out