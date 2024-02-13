import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset import config
from decompile_tracr.dataset import lib
from decompile_tracr.dataset.sample_rasp import filter
from decompile_tracr.dataset.sample_rasp import save_batch


logger = setup_logger(__name__)


def tokenize_loop():
    json_dataset = []
    aux_dataset = []
    for program_id, program in enumerate(lib.examples):
        tokens = tokenizer.tokenize(program)
        if filter(tokens):
            logger.warning(f"Program {program_id} is too long.")

        logger.info(f"Compiled and tokenized program {program_id}.")

        json_dataset.append({
            "name": "lib",
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })

        aux_dataset.append({
            "rasp": program,  # rasp.SOp
        })
    return json_dataset, aux_dataset


json_dataset, aux_dataset = tokenize_loop()


SAVEPATH = config.unprocessed_dir
os.makedirs(SAVEPATH, exist_ok=True)


save_batch(
    json_dataset, 
    aux_dataset, 
    savedir=SAVEPATH,
    keep_aux=True,
    overwrite=True,
    filename="lib",
)
