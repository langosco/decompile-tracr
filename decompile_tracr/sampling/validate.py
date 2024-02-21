from typing import Optional
from tracr.rasp import rasp
import numpy as np
from tracr.compiler.validating import validate
from tracr.compiler import compiling
from decompile_tracr.sampling import map_primitives, rasp_utils


# TODO: move this fn to a different module, maybe validate.py?
def validate_custom_types(expr: rasp.SOp, test_input):
    """We use custom types - bool, float, and categorical - to enforce
    constraints on the output of SOps. If those constraints are violated,
    this function raises an error.
    """
    out = expr(test_input)
    if expr.annotations["type"] == "bool":
        # bools are numerical and only take values 0 or 1
        if not set(out).issubset({0, 1, None}):
            raise ValueError(f"Bool SOps may only take values 0 or 1. Instead, received {out}")
        elif not rasp.numerical(expr):
            raise ValueError(f"Bool SOps must be numerical. Instead, {expr} is categorical.")
    elif expr.annotations["type"] == "float":
        # floats are numerical and can take any value
        if not rasp.numerical(expr):
            raise ValueError(f"Float SOps must be numerical. Instead, {expr} is categorical.")
    elif expr.annotations["type"] == "categorical":
        if not rasp.categorical(expr):
            raise ValueError(f"{expr} is annotated as type=categorical, but is actually numerical.")


def validate_compilation(expr: rasp.SOp, test_input: list):
    model = compiling.compile_rasp_to_model(
        expr,
        vocab={0,1,2,3,4},
        max_seq_len=5,
        compiler_bos="BOS"
    )

    rasp_out = expr(test_input)
    rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
    model_out = model.apply(["BOS"] + test_input).decoded[1:]

    if not np.allclose(model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3):
        raise ValueError(f"Compiled program {expr.label} does not match RASP output.\n"
                            f"Compiled output: {model_out}\n"
                            f"RASP output: {rasp_out}\n"
                            f"Test input: {test_input}\n"
                            f"SOp: {expr}")

