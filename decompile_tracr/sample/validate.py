from collections import Counter
import numpy as np

from decompile_tracr.sample import rasp_utils
from decompile_tracr.sample.rasp_utils import SamplingError
from decompile_tracr.dataset.logger_config import setup_logger

from tracr.rasp import rasp
from tracr.compiler import validating
from tracr.compiler import compiling

logger = setup_logger(__name__)


def validate_custom_types(expr: rasp.SOp, values: list) -> None:
    """We use custom types - bool, float, and categorical - to enforce
    constraints on the output of SOps. If those constraints are violated,
    this function raises an error.

    Usage: 
    p = rasp_program()
    test_inputs = [1,2,3,4]
    validate_custom_types(p, p(test_inputs))
    """
    if expr.annotations["type"] == "bool":
        # bools are numerical and only take values 0 or 1
        if not set(values).issubset({0, 1, None}):
            raise ValueError(f"Bool SOps may only take values 0 or 1. Instead, received {values}")
        elif not rasp.numerical(expr):
            raise ValueError(f"Bool SOps must be numerical. Instead, {expr} is categorical.")
    elif expr.annotations["type"] == "float":
        # floats are numerical and can take any value
        if not rasp.numerical(expr):
            raise ValueError(f"Float SOps must be numerical. Instead, {expr} is categorical.")
    elif expr.annotations["type"] == "categorical":
        if not rasp.categorical(expr):
            raise ValueError(f"{expr} is annotated as type=categorical, but is actually numerical.")


def validate_compilation(expr: rasp.SOp, test_input: list) -> None:
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


def perform_checks(program, inputs: list[list]):
    """Given a sampled program, perform checks to see if we need to resample.
    """
    if len(validating.validate(program)) > 0:
        raise SamplingError("Program failed static validation.")

    try:
        outputs = [program(x) for x in inputs]
    except ValueError as e:
        if e.args[0] in ["key is None!", "query is None!"]:
            raise SamplingError(f"Program {program} is invalid "
                                "due to Nones in Select.")
        elif e.args[0].startswith("Only types int, bool, and float "
                                  "are supported for aggregation."):
            raise SamplingError(f"Program {program} is invalid "
                                "due to Nones in Aggregate.")
        else:
            raise

    if outputs == inputs:
        raise SamplingError("Program is the identity.")

    if any(rasp_utils.fraction_none(x) > 0.5 for x in outputs):
        raise SamplingError("Program returns None too often.")

    if is_constant(outputs):
        raise SamplingError("Program returns the same value too often.")
    
    tracr_dynamic_validate(program, inputs)

    return None


def tracr_dynamic_validate(program, inputs: list[list]):
    for x in inputs:
        if len(validating.validate(program, x)) > 0:
            raise SamplingError(f"Program failed dynamic validation.")


def is_constant(values: list[list[int | float]]) -> bool:
    """Return True if the values are constant or almost constant.
    If values have inhomogeneus lengths, they are treated as
    incomparable and only the subset of elements that are
    of the most frequent length are compared.
    """
    # make sure there is a reasonable number of values
    if len(values) < 10:
        logger.debug(
            f"is_constant: Too few values to determine if program is "
            f"constant. Got len(values) = {len(values)}, need > 10."
        )
        return False

    lens = [len(x) for x in values]
    if not all(l == lens[0] for l in lens):
        most_common_len = Counter(lens).most_common(1)[0][0]
        values = [x for x in values if len(x) == most_common_len]
        return is_constant(values)
    else:
        values = [[0 if x is None else x for x in v] for v in values]
        values = np.array(values)
        return values.std(axis=0).sum() < 0.5