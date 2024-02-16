import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from decompile_tracr.tokenizing import tokenizer
from decompile_tracr.dataset.logger_config import setup_logger
from decompile_tracr.dataset import config
from decompile_tracr.dataset import lib
from decompile_tracr.dataset.generate import filter
from decompile_tracr.dataset.data_utils import save_batch


logger = setup_logger(__name__)


def tokenize_loop():
    data = []
    for program_id, program in enumerate(lib.examples):
        tokens = tokenizer.tokenize(program)
        if filter(tokens):
            logger.warning(f"Program {program_id} is too long.")

        data.append({
            "name": "lib",
            "n_sops": program.annotations['length'],  # nr of sops
            "tokens": tokens,
        })
    
    logger.info(f"Compiled all {len(data)} example programs.")
    return data


def tokenize_lib(savedir=config.unprocessed_dir):
    data = tokenize_loop()
    save_batch(
        data=data, 
        savedir=savedir,
        overwrite=True,
        filename="lib",
    )


if __name__ == "__main__":
    tokenize_lib()