# Desc: encode / decode a RASP program into a sequence of tokens.

import jax

from tracr.compiler import rasp_to_graph
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler import basis_inference
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp


from rasp_tokenizer import rasp_to_str, str_to_rasp
import rasp_tokenizer.vocab as voc


def encode(flat_program: list[str]):
    return [voc.vocab.index(x) for x in flat_program]


def decode(tokenized_program: list[int]):
    return [voc.vocab[x] for x in tokenized_program]


def tokenize(program: rasp.SOp) -> dict[str, list[int]]:
    by_layer = rasp_to_str.rasp_to_str(program)
    return {layer: encode(v) for layer, v in by_layer.items()}


def detokenize(tokens_by_layer: dict[str, list[int]]):
    decoded_by_layer = {
        layer: decode(x) for layer, x in tokens_by_layer.items()
    }
    return str_to_rasp.str_to_rasp(decoded_by_layer)