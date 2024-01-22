# Desc: encode / decode a RASP program into a sequence of tokens.

from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel
import jax

from rasp_tokenizer.compiling import compile_rasp_to_model_and_return_graph
from rasp_tokenizer import utils
from rasp_tokenizer.vocab import vocab as tokenizer_vocab


def encode(flat_program: list[str]):
    return [tokenizer_vocab.index(x) for x in flat_program]


def decode(tokenized_program: list[int]):
    return [tokenizer_vocab[x] for x in tokenized_program]


def compile_and_tokenize(
        program: rasp.SOp,
        vocab: set[int] = {0,1,2,3,4,5},
        max_seq_len: int = 5,
    ) -> (AssembledTransformerModel, 
          dict[str, list[int]],
          dict[str, jax.Array]):
    """
    1) Compile the RASP program into a haiku model.
    2) Represent the program as a list of ops per layer.
    3) Tokenize the program, resulting in a list of tokens per layer.
    4) Extract the parameters corresponding to each layer.
    """

    model, graph, sources = compile_rasp_to_model_and_return_graph(
        program,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )

    by_layer = utils.rasp_graph_to_layerwise_representation(
        graph, sources)

    tokens_by_layer = {
        layername: encode(ops) for layername, ops in by_layer.items()
    }

    params_by_layer = {
        layername: utils.get_params(model.params, layername) 
            for layername in by_layer.keys()
    }

    return model, tokens_by_layer, params_by_layer