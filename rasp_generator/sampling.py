
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
from rasp_generator import map_primitives, utils


class SamplingError(Exception):
    pass


TEST_INPUTS = [utils.sample_test_input(np.random.default_rng(0)) 
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
    sops = utils.filter_by_type(sops, type=type_constraint)
    sops = utils.filter_by_constraints(sops, other_constraints, constraints_name)

    if size is not None and len(sops) < size:
        raise utils.EmptyScopeError(
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
    return utils.annotate_type(sop_out, type=output_type)


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
    return utils.annotate_type(sop_out, type="categorical")


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
    return utils.annotate_type(sop_out, type="float")


def sample_select(rng, variable_scope: list):
    """Sample a rasp.Select. A select takes two categorical SOps and
    returns a selector (matrix) of booleans."""
    # TODO: allow Selectors with bools (numerical & 0-1) as input?
    sop_in1, sop_in2 = sample_from_scope(
        rng,
        variable_scope,
        type_constraint="categorical",
        other_constraints=[lambda sop: utils.no_none_in_values(sop, TEST_INPUTS)],
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
        other_constraints=[lambda sop: utils.no_none_in_values(sop, TEST_INPUTS)],
        constraints_name="no None values",
    )
    sop_out = rasp.Aggregate(selector, sop_in, default=0)
    # TODO: sometimes output can be bool here?
    return utils.annotate_type(sop_out, type="float")  # has to be numerical (tracr constraint)


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
        other_constraints=[lambda sop: utils.no_none_in_values(sop, TEST_INPUTS)],
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

    return utils.annotate_type(sop_out, type="categorical")


def sample_selector_width(rng, variable_scope: list):
    selector = sample_select(rng, variable_scope)
    sop_out = rasp.SelectorWidth(selector)
    return utils.annotate_type(sop_out, type="categorical")


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
        if any(utils.fraction_none(sop(x)) > 0.5 for x in TEST_INPUTS):
            raise SamplingError(f"Sampled SOp {sop} has too many None values.")
    except (utils.EmptyScopeError, SamplingError) as error:
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
            validate_compilation: bool = False,
            rng: Optional[np.random.RandomState] = None,
        ):
        self.sops = [
            utils.annotate_type(rasp.tokens, "categorical"),
            utils.annotate_type(rasp.indices, "categorical"),
        ]
        self.validate_compilation = validate_compilation
        self.rng = np.random.default_rng(rng)

    def sample_sops(self, n_sops=15):
        """Sample a list of SOps.
        Returns a list of errors that occurred during sampling.
        """
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

            if sop is None and isinstance(err, utils.EmptyScopeError):
                avoid.add(sop_class)

        if len(self.sops) <= 3:
            errs = "\n".join([str(err) for err in errs])
            raise SamplingError(f"Failed to sample program. Received errors:\n"
                                f"{errs}")

        return errs
    

    def sample(self, target_length=15):
        """Sample a program."""
        self.sample_sops(n_sops=target_length*2)
        candidates = self.sops[-10:]
        lengths = np.array(
            [utils.program_length(sop) for sop in candidates])

        # return longest program
        self.program = candidates[np.argmax(lengths)]

        if lengths.max() < 4:
            raise SamplingError(
                f"Sampled program too short: length {lengths.max()}."
            )

#        self.validate()
        return self.program


    def validate(self):
        """Validate the program. This is fairly slow when 
        self.validate_compilation is enabled."""
        sop = self.program
        for x in TEST_INPUTS:
            validate_custom_types(sop, x)
            errs = validate(sop, x)
            if len(errs) > 0:
                raise ValueError(f"Invalid program: {errs}")
            elif self.validate_compilation:
                validate_compilation(sop, x)
        return