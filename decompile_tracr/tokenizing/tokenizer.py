# Desc: encode / decode a RASP program into a sequence of tokens.

import numpy as np
from tracr.rasp import rasp

from decompile_tracr.tokenizing import rasp_to_str, str_to_rasp
import decompile_tracr.tokenizing.vocab as voc


def encode_token(x: str) -> int:
    try:
        return voc.vocab.index(x)
    except ValueError as e:
        assert e.args[0] == "tuple.index(x): x not in tuple"
        raise ValueError(f"Not in vocab: {x}")


def decode_token(x: int) -> str:
    return voc.vocab[x]


def encode(x: list[str]) -> list[int]:
    return [encode_token(tok) for tok in x]


def decode(x: list[int]) -> list[str]:
    return [decode_token(tok) for tok in x]


def tokenize(program: rasp.SOp) -> list[int]:
    """Tokenize a RASP program."""
    if not isinstance(program, rasp.SOp):
        raise ValueError("Input must be a RASP program.")

    by_layer = rasp_to_str.rasp_to_str(program)
    return encode(by_layer)


def detokenize(tokens: list[int]) -> rasp.SOp:
    if not isinstance(tokens[0], (int, np.integer)):
        raise ValueError(f"Input elements must be integers. Got "
                         f"{type(tokens[0])}.")

    decoded = decode(tokens)
    return str_to_rasp.str_to_rasp(decoded)