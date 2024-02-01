# TODO: clean up and make more readable

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
from jaxtyping import ArrayLike
import pickle
import signal

from tracr.compiler.craft_model_to_transformer import NoTokensError
from tracr.compiler.basis_inference import InvalidValueSetError
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel

from rasp_generator.utils import sample_test_input, print_program
from rasp_tokenizer import tokenizer
from rasp_tokenizer.compiling import COMPILER_BOS
from rasp_tokenizer.logger_config import setup_logger
from rasp_tokenizer import paths
from rasp_tokenizer import lib
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH, MAX_WEIGHTS_LAYER_MEAN
from rasp_tokenizer.data_utils import save_batch, to_flat_datapoints


logger = setup_logger(__name__)
rng = np.random.default_rng(0)
test_inputs = [sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]


SAVEPATH = paths.data_dir / "deduped" / "lib"
os.makedirs(SAVEPATH, exist_ok=True)


def filter(by_layer: list[dict]):
    max_rasp_len = max(len(x['rasp_tok']) for x in by_layer)
    max_weights_len = max(len(x['weights']) for x in by_layer)
    if max_rasp_len > MAX_RASP_LENGTH:
        logger.warning(f"RASP length too long, "
                       f"{max_rasp_len} > {MAX_RASP_LENGTH}.")
        return True
    elif max_weights_len > MAX_WEIGHTS_LENGTH:
        logger.warning(f"Weights length too long, "
                       f"{max_weights_len} > {MAX_WEIGHTS_LENGTH} ")
        return True
    
    means_by_layer = [np.mean(x['weights']) for x in by_layer]
    maxw = np.abs(means_by_layer).max()
    if maxw > MAX_WEIGHTS_LAYER_MEAN:
        logger.warning(f"Max weight value too large, "
                       f"{maxw} > {MAX_WEIGHTS_LAYER_MEAN}.")
        return True
    return False


def compile_loop():
    json_dataset = []
    aux_dataset = []
    for program_id, program in enumerate(lib.examples):
        model, tokens, params = tokenizer.compile_and_tokenize(program)
        by_layer, rasp_str = to_flat_datapoints(tokens, params)

        filter(by_layer)

        logger.info(f"Compiled and tokenized program {program_id}.")

        json_dataset.append({
            "weights_and_tokens": by_layer,  # list of dicts
            "name": "lib",  # train vs test
            "n_sops": program.annotations['length'],  # nr of sops
        })

        aux_dataset.append({
            "rasp": program,  # rasp.SOp
            "model": model,  # AssembledTransformerModel
        })
    return json_dataset, aux_dataset


json_dataset, aux_dataset = compile_loop()

save_batch(
    json_dataset, 
    aux_dataset, 
    savedir=SAVEPATH,
    keep_aux=True,
    filename="data",
)
