# Desc: encode / decode a RASP program into a sequence of tokens.

from tracr.rasp import rasp

from decompile_tracr.tokenizing import rasp_to_str, str_to_rasp
import decompile_tracr.tokenizing.vocab as voc


def encode(flat_program: list[str]):
    return [voc.vocab.index(x) for x in flat_program]


def decode(tokenized_program: list[int]):
    return [voc.vocab[x] for x in tokenized_program]


def tokenize(program: rasp.SOp) -> list[list[int]]:
    """
    Tokenize a RASP program.
    Output is a list of lists, where each list is 
    a layer of the program.
    """
    if not isinstance(program, rasp.SOp):
        raise ValueError("Input must be a RASP program.")

    by_layer = rasp_to_str.rasp_to_str(program)
    return [encode(l) for l in by_layer]


def detokenize(tokens: list[list[int]]) -> rasp.SOp:
    if not (isinstance(tokens, list) and isinstance(tokens[0], list) 
            and isinstance(tokens[0][0], int)):
        raise ValueError("Input must be a list of lists of ints.")

    decoded = [decode(x) for x in tokens]
    return str_to_rasp.str_to_rasp(decoded)