
# Sample a program composed of RASP instructions:
# - Map
# - SequenceMap
# - LinearSequenceMap (compiled exactly into MLP weights)
# - Select
# - Aggregate
# - SelectorWidth


# Desiderata for sampled programs:
# - it follows the Tracr constraints so it compiles correctly
# - it has a reasonable number of instructions
# - it doesn't do anything trivial (eg return a constant)
# - it uses comparisons when appropriate (eg don't < compare two strings)


from typing import Optional
from tracr.rasp import rasp
import numpy as np
from tracr.compiler.validating import validate
from tracr.compiler import compiling
from decompile_tracr.sampling import map_primitives, rasp_utils


class SamplingError(Exception):
    """Raised when the sampler fails to sample a program
    satisfying the given constraints.
    This error is raises stochastically, so it's (usually) 
    not indicative of a bug.
    """
    pass


TEST_INPUTS = [rasp_utils.sample_test_input(np.random.default_rng(0)) 
               for _ in range(50)]
TEST_INPUTS += [[0], [0,0,0,0,0], [1,2,3,4]]


def get_recency_bias_weights(n: int, alpha: float = 0.3):
    """Unnormalized."""
    weights = np.arange(n) + 1
    return weights**alpha


def sample_from_scope(
        rng: np.random.Generator,
        sops: list[rasp.SOp],
        type_constraint=None,
        other_constraints: list[callable] = [],
        constraints_name: Optional[str] = None,
        size=None,
        replace=False,
        prefer_recent=False,
    ):
    """Sample a SOp from a given list of SOps, according to constraints."""
    sops = rasp_utils.filter_by_type(sops, type=type_constraint)
    sops = rasp_utils.filter_by_constraints(sops, other_constraints, constraints_name)

    if size is not None and len(sops) < size:
        raise rasp_utils.EmptyScopeError(
            f"Filter failed. Not enough SOps in scope; found {len(sops)}, need {size}")

    if prefer_recent:
        weights = get_recency_bias_weights(len(sops), 0.5)
    else:
        weights = [1] * len(sops)

    if type_constraint in {"categorical", None}:
        weights[0] = 3
    weights = np.array(weights) / np.sum(weights)

    return rng.choice(
        sops, 
        size=size, 
        replace=replace, 
        p=weights)


def sample_map(rng, variable_scope: list[rasp.SOp]):
    """Sample a map. A map applies a function elementwise to a SOp.
    The input SOps can be categorical, float, or bool."""
    sop_in = sample_from_scope(rng, variable_scope)
    fn, output_type = map_primitives.get_map_fn(rng, sop_in.annotations["type"])
    sop_out = rasp.Map(fn, sop_in, simplify=False)
    return rasp_utils.annotate_type(sop_out, type=output_type)


def sample_sequence_map(rng, variable_scope: list[rasp.SOp]):
    """Sample a sequence map. A SM applies a function elementwise to
    two categorical SOps. The output is always categorical."""
    args = sample_from_scope(
        rng, 
        variable_scope, 
        type_constraint="categorical", 
        size=2, 
        replace=False,
    )
    fn = rng.choice(map_primitives.NONLINEAR_SEQMAP_FNS)
    sop_out = rasp.SequenceMap(fn, *args)
    return rasp_utils.annotate_type(sop_out, type="categorical")


def sample_linear_sequence_map(rng, variable_scope: list):
    """Sample a linear sequence map. A LNS linearly combines two
    numerical SOps. The output is always numerical."""
    args = sample_from_scope(
        rng, 
        variable_scope, 
        type_constraint="float", 
        size=2, 
        replace=False,
    )
    weights = rng.choice(
        map_primitives.LINEAR_SEQUENCE_MAP_WEIGHTS, size=2, replace=True)
    weights = [int(w) for w in weights]
    sop_out = rasp.LinearSequenceMap(*args, *weights)
    return rasp_utils.annotate_type(sop_out, type="float")


def sample_select(rng, variable_scope: list):
    """Sample a rasp.Select. A select takes two categorical SOps and
    returns a selector (matrix) of booleans."""
    # TODO: allow Selectors with bools (numerical & 0-1) as input?
    sop_in1, sop_in2 = sample_from_scope(
        rng,
        variable_scope,
        type_constraint="categorical",
        other_constraints=[lambda sop: rasp_utils.no_none_in_values(sop, TEST_INPUTS)],
        constraints_name="no None values",
        size=2,
        replace=True,
    )
    comparison = rng.choice(map_primitives.COMPARISONS)
    selector = rasp.Select(sop_in1, sop_in2, comparison)
    return selector


def sample_numerical_aggregate(rng, variable_scope: list):
    """
    Tracr puts constraints on select-aggregate operations. This aggregate is 
    sampled to satisfy the constraint that the sop argument is 'boolean',
    that is numerical and only takes values 0 or 1.
    This constraint is necessary for all aggregates with numerical SOps.
    """
    selector = sample_select(rng, variable_scope)
    sop_in = sample_from_scope(
        rng,
        variable_scope,
        type_constraint="bool",
        other_constraints=[lambda sop: rasp_utils.no_none_in_values(sop, TEST_INPUTS)],
        constraints_name="no None values",
    )
    sop_out = rasp.Aggregate(selector, sop_in, default=0)
    # TODO: sometimes output can be bool here?
    return rasp_utils.annotate_type(sop_out, type="float")  # has to be numerical (tracr constraint)


def sample_categorical_aggregate(rng, variable_scope: list, max_retries=10):
    """
    Sample an aggregate operation with a categorical SOp. Categorical aggregate
    operations must satisfy the constraint that there is no aggregation (eg averaging).
    This usually means that the selector has width 1, but it could also mean that a width > 1
    selector is used but the output domain of the aggregate is still equal to the input domain.
    """
    selector = sample_select(rng, variable_scope)
    sop_in = sample_from_scope(
        rng,
        variable_scope,
        type_constraint="categorical",
        other_constraints=[lambda sop: rasp_utils.no_none_in_values(sop, TEST_INPUTS)],
        constraints_name="no None values",
    )
    sop_out = rasp.Aggregate(selector, sop_in, default=None)

    # validate:
    if not all(
        set(sop_out(x)).issubset(set(sop_in(x))) for x in TEST_INPUTS
    ):
        if max_retries > 0:
            sop_out = sample_categorical_aggregate(rng, variable_scope, max_retries=max_retries-1)
        else:
            raise SamplingError(
                "Could not sample categorical Aggregate with valid output domain "
                "(Maximum retries reached). "
                "This because the sampler couldn't find a selector with width 1, and other sampled selectors "
                "don't result in an output domain that is a subset of the input domain."
            )

    return rasp_utils.annotate_type(sop_out, type="categorical")


def sample_selector_width(rng, variable_scope: list):
    selector = sample_select(rng, variable_scope)
    sop_out = rasp.SelectorWidth(selector)
    return rasp_utils.annotate_type(sop_out, type="categorical")


SAMPLE_FUNCTIONS = {
    "map": sample_map,
    "sequence_map": sample_sequence_map,
    "linear_sequence_map": sample_linear_sequence_map,
    "numerical_aggregate": sample_numerical_aggregate,
    "categorical_aggregate": sample_categorical_aggregate,
    "selector_width": sample_selector_width,
}


def try_to_sample_sop(rng, variable_scope: list, avoid_types: set[str] = []):
    """Sample a SOp.
    If the sampler fails, it will return a SamplingError and the class of the SOp.
    This allows us to try again with a different class."""
    sample_from = set(SAMPLE_FUNCTIONS.keys()) - set(avoid_types)
    sop_class = rng.choice(list(sample_from))
    err = None
    try:
        sop = SAMPLE_FUNCTIONS[sop_class](rng, variable_scope)
        if any(rasp_utils.fraction_none(sop(x)) > 0.5 for x in TEST_INPUTS):
            raise SamplingError(f"Sampled SOp {sop} has too many None values.")
    except (rasp_utils.EmptyScopeError, SamplingError) as error:
        err = error
        sop = None
    return sop, err, sop_class


# TODO: move this fn to a different module, maybe validate.py?
def validate_custom_types(expr: rasp.SOp, test_input):
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


class ProgramSampler:
    def __init__(
            self, 
            rng: Optional[np.random.RandomState] = None,
        ):
        self.rng = np.random.default_rng(rng)
    
    def reset(self):
        self.sops = [
            rasp_utils.annotate_type(rasp.tokens, "categorical"),
            rasp_utils.annotate_type(rasp.indices, "categorical"),
        ]

    def sample_sops(self, n_sops=15):
        """Sample a list of SOps.
        Returns a list of errors that occurred during sampling.
        """
        self.reset()
        errs = []
        for _ in range(n_sops):
            avoid = set()
            sop, err, sop_class = try_to_sample_sop(
                self.rng, self.sops, avoid)

            if sop is not None:
                self.sops.append(sop)
            else:
                assert err is not None
                errs.append(
                    f"{repr(err)} (tried to sample {sop_class})"
                )

            if sop is None and isinstance(err, rasp_utils.EmptyScopeError):
                avoid.add(sop_class)

        if len(self.sops) <= 3:
            errs = "\n".join([str(err) for err in errs])
            raise SamplingError(f"Failed to sample program. Received errors:\n"
                                f"{errs}")

        return errs
    

    def sample(self, n_sops=15, min_length=None, max_length=None):
        """Sample a program. Always choose longest valid program.
        Args:
            n_sops: number of SOps to sample
            min_length: minimum length of the program in SOps
            max_length: maximum length of the program in SOps
        """
        errs = self.sample_sops(n_sops=n_sops)
        first_candidate = 3 if min_length is None else min_length
        candidates = np.array(self.sops[first_candidate:])
        lengths = np.array(
            [rasp_utils.count_sops(sop) for sop in candidates])

        valid = np.ones(len(candidates), dtype=bool)
        if min_length is not None:
            valid = lengths >= min_length
        if max_length is not None:
            valid = (lengths <= max_length) & valid
        
        candidates = candidates[valid]

        if len(candidates) == 0:
            raise SamplingError(
                f"Could not find program of length between "
                f"{min_length} and {max_length}.")
        
        program = candidates[lengths[valid].argmax()]

        length = rasp_utils.count_sops(program)
        if length < 4:
            raise SamplingError(
                f"Sampled program too short: length {length}."
            )
        program = rasp.annotate(program, length=length)
        self.program = program
        return program, errs


    def validate(self, sop: rasp.SOp, val_compile=False):
        """Validate the program. This is fairly slow when 
        val_compile is enabled."""
        for x in TEST_INPUTS:
            validate_custom_types(sop, x)
            errs = validate(sop, x)
            if len(errs) > 0:
                raise ValueError(f"Invalid program: {errs}")
            elif val_compile:
                validate_compilation(sop, x)
        return