# Desc: encode / decode a RASP program into a sequence of tokens.

from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel
import jax

from rasp_tokenizer.compiling import compile_rasp_to_model_and_return_graph
from rasp_tokenizer import utils
import rasp_tokenizer.vocab as voc


def encode(flat_program: list[str]):
    return [voc.vocab.index(x) for x in flat_program]


def decode(tokenized_program: list[int]):
    return [voc.vocab[x] for x in tokenized_program]


def compile_and_tokenize(
        program: rasp.SOp,
        vocab: set[int] = {0,1,2,3,4},
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


#        flat_layer = ["START"]
#        for node_id in node_ids:
#            flat_layer.append(get_variable_name(graph, node_id))
#            flat_layer.append(get_encoding(graph, node_id))
#            flat_layer.append(get_classname(graph, node_id))
#            flat_layer.extend(get_args(graph, node_id))
#        flat_layer.append("END")



def get_next_op(str_tokens: list[str]):
    raise NotImplementedError

    if len(str_tokens) == 0:
        raise ValueError("Empty program.")
    elif str_tokens[0] in ("START", "END"):
        str_tokens = str_tokens[1:]
        return None
    
    str_tokens.reverse()

    var_name = str_tokens.pop()
    maybe_encoding = str_tokens.pop()
    if maybe_encoding in ("categorical", "numerical"):
        encoding = maybe_encoding
        classname = str_tokens.pop()
        assert classname in voc.ops
    else:
        encoding = None
        classname = maybe_encoding
        assert classname == "Select"
    
    args = ...

    return encodings[encoding](
        ops[classname](*args)
    )




def tokens_to_rasp(tokens: list[int]):
    str_tokens = decode(tokens)[1:-1]
    while len(str_tokens) > 0:
        op = get_next_op(str_tokens)
    return op




def detokenize(tokens_by_layer: dict[str, list[int]]):
    ops_by_layer = {
        layername: decode(tokens) for layername, tokens in tokens_by_layer.items()
    }

