
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


from time import time
from typing import Optional
from dataclasses import dataclass
import numpy as np
from tracr.rasp import rasp
from tracr.compiler import validating
from rasp_gen.sample import map_primitives
from rasp_gen.sample import rasp_utils
from rasp_gen.sample.rasp_utils import SamplingError
from rasp_gen.sample.validate import perform_checks
from rasp_gen.dataset.logger_config import setup_logger

logger = setup_logger(__name__)


const_rng = np.random.default_rng(42)
TEST_INPUTS = [rasp_utils.sample_test_input(const_rng) 
               for _ in range(15)]
TEST_INPUTS += [[0], [0,0,0,0,0], [1,2,3,4]]
TEST_INPUTS = set([tuple(x) for x in TEST_INPUTS])

EXTRA_TEST_INPUTS = [rasp_utils.sample_test_input(const_rng)
                        for _ in range(50)]
EXTRA_TEST_INPUTS += [
    rasp_utils.sample_test_input(const_rng, max_seq_len=5, min_seq_len=5)
    for _ in range(50)]
EXTRA_TEST_INPUTS = set([tuple(x) for x in EXTRA_TEST_INPUTS])


def get_recency_bias_weights(n: int, alpha: float = 0.3) -> np.ndarray:
    """Unnormalized."""
    weights = np.arange(n) + 1
    return weights**alpha


class Sampler:
    """Sampler for RASP programs. 
    - Sampled SOps are added to self.scope.
    - self.past is a list of sets that keeps track of the indices of the SOps
        that make up the past of each sampled SOp.
    """
    def __init__(
        self, 
        rng: np.random.Generator,
        only_categorical: bool = False,
    ):
        self.rng = rng
        self.scope = [
            rasp_utils.annotate_type(rasp.tokens, "categorical"),
            rasp_utils.annotate_type(rasp.indices, "categorical"),
        ]
        self.past = [{0}, {1}]
        self.only_categorical = only_categorical
#        self.value_set = []  # dynamically infer value set TODO
    
    def sample_from_scope(
        self,
        type: Optional[str] = None,
        allow_none_values: bool = True,
        size=None,
        replace=False,
        prefer_recent=False,
    ) -> int | list[int]:
        """Sample a SOp that satisfies constraints.
        Returns a index (or list of indices) of the sampled SOps in the scope.
        """
        mask = rasp_utils.filter_by_type(self.scope, type=type)
        if not allow_none_values:
            mask *= rasp_utils.filter_nones(self.scope, TEST_INPUTS)
            
        if (size is not None and mask.sum() < size) or (mask.sum() < 1):
            raise rasp_utils.EmptyScopeError(
                f"Filter failed. Not enough SOps in scope; "
                "found {mask.sum()}, need {size}")

        if prefer_recent:
            weights = get_recency_bias_weights(len(self.scope), 0.5)
        else:
            weights = [1] * len(self.scope)
        
        weights[0] = 3

        weights = np.array(weights)
        weights = weights * mask
        weights = weights / weights.sum()

        assert len(self.scope) == len(weights)
        assert len(mask) == len(weights)

        idx = self.rng.choice(
            len(mask),
            size=size, 
            replace=replace, 
            p=weights,
        )

        if size is None:
            return self.scope[idx]
        else:
            return [self.scope[i] for i in idx]

    def add_map(self):
        """Sample a map. A map applies a function elementwise to a SOp.
        The input SOps can be categorical, float, or bool."""
        sop_in = self.sample_from_scope()
        input_type = sop_in.annotations["type"]
        if input_type == "float" and isinstance(sop_in, rasp.Aggregate):
            input_type = "freq"  # hack to accomodate outputs from numerical Aggregates, which are always between 0 and 1
        output_types = (["categorical"] if self.only_categorical 
                        else map_primitives.TYPES)
        fn, output_type = map_primitives.get_map_fn(
            self.rng, input_type, output_types)
        sop_out = rasp.Map(fn, sop_in, simplify=False)
        self.scope.append(rasp_utils.annotate_type(sop_out, type=output_type))

    def add_sequence_map(self):
        """Sample a sequence map. A SM applies a function elementwise to
        two categorical SOps. The output is always categorical."""
        sops_in = self.sample_from_scope(
            type="categorical", 
            size=2, 
            replace=False,
        )
        fn = self.rng.choice(map_primitives.NONLINEAR_SEQMAP_FNS)
        sop_out = rasp.SequenceMap(fn, *sops_in)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="categorical"))

    def add_linear_sequence_map(self):
        """Sample a linear sequence map. A LNS linearly combines two
        numerical SOps. The output is always numerical."""
        sops_in = self.sample_from_scope(
            type="float", 
            size=2, 
            replace=False,
        )
        weights = self.rng.choice(
            map_primitives.LINEAR_SEQUENCE_MAP_WEIGHTS, size=2, replace=True)
        weights = [int(w) for w in weights]
        sop_out = rasp.LinearSequenceMap(*sops_in, *weights)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="float"))

    def add_numerical_aggregate(self):
        """
        Tracr puts constraints on select-aggregate operations. Numerical aggregates are 
        sampled to satisfy the constraint that the sop argument is 'boolean', i.e. it is
        numerical and only takes values 0 or 1. This constraint is necessary for all 
        aggregates with numerical SOps.
        Note: outputs are always in the closed interval [0, 1] ('frequencies').
        """
        selector = self.get_selector()
        sop_in = self.sample_from_scope(
            type="bool",
            allow_none_values=False,
        )
        sop_out = rasp.Aggregate(selector, sop_in, default=0)
        # TODO: sometimes output can be bool here?
        self.scope.append(rasp_utils.annotate_type(sop_out, type="float"))

    def add_categorical_aggregate(self, max_retries=10):
        """
        Sample an aggregate operation with a categorical SOp. Categorical aggregate
        operations must satisfy the constraint that there is no aggregation (eg averaging).
        This usually means that the selector has width 1, but it could also mean that a width > 1
        selector is used but the output domain of the aggregate is still equal to the input domain.
        """
        sop_in = self.sample_from_scope(
            type="categorical",
            allow_none_values=False,
        )
        selector = self.get_selector()
        sop_out = rasp.Aggregate(selector, sop_in, default=None)
        sop_out = rasp_utils.annotate_type(sop_out, type="categorical")

        # validate:
        if not all(
            set(sop_out(x)).issubset(set(sop_in(x)) | {None}) for x in TEST_INPUTS
        ):
            if max_retries > 0:
                self.add_categorical_aggregate(max_retries=max_retries-1)
            else:
                raise SamplingError(
                    "Could not sample categorical Aggregate with valid output domain "
                    "(Maximum retries reached). "
                    "This because the sampler couldn't find a selector with width 1, and other sampled selectors "
                    "don't result in an output domain that is a subset of the input domain."
                )
        else:
            for x in TEST_INPUTS:
                errs = validating.validate(sop_out, x)
                if len(errs) > 0:
                    print()
                    logger.warning(f"Sampled categorical Aggregate failed validation: {errs}")
                    logger.info(f"test input: {x}")
                    logger.info(f"aggregate sop: {sop_out.label}")
                    print()
            self.scope.append(sop_out)

    def add_selector_width(self):
        selector = self.get_selector()
        sop_out = rasp.SelectorWidth(selector)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="categorical"))

    def get_selector(self):
        """Sample a rasp.Select. A select takes two categorical SOps and
        returns a selector (matrix) of booleans. Note this is not an SOp.
        """
        # TODO: allow Selectors with bools (numerical & 0-1) as input?
        sops_in = self.sample_from_scope(
            type="categorical",
            allow_none_values=False,
            size=2,
            replace=True,
        )
        comparison = self.rng.choice(map_primitives.COMPARISONS)
        selector = rasp.Select(*sops_in, comparison)
        return selector

    def try_to_add_sop(self, avoid_types: set[str]) -> tuple[list, set[str]]:
        """Sample a single SOp.
        If the sampler fails, it will return a SamplingError and the class of the SOp.
        This allows us to try again with a different class.
        """
        add_functions = {
            "map": self.add_map,
            "sequence_map": self.add_sequence_map,
            "linear_sequence_map": self.add_linear_sequence_map,
            "numerical_aggregate": self.add_numerical_aggregate,
            "categorical_aggregate": self.add_categorical_aggregate,
            "selector_width": self.add_selector_width,
        }

        weights = {
            "map": 1,
            "sequence_map": 0.8,
            "linear_sequence_map": 1,
            "numerical_aggregate": 1,
            "categorical_aggregate": 1,
            "selector_width": 0.05,
        }
        if self.only_categorical:
            weights["linear_sequence_map"] = 0
            weights["numerical_aggregate"] = 0

        sop_classes = list(set(add_functions.keys()) - set(avoid_types))
        weights = np.array([weights[c] for c in sop_classes]); weights /= weights.sum()
        sop_class = self.rng.choice(sop_classes, p=weights)
        add = add_functions[sop_class]

        try:
            add()
            if any(rasp_utils.fraction_none(self.run(x)) > 0.5 for x in TEST_INPUTS):
                del self.scope[-1]
                raise SamplingError(f"Sampled SOp has too many None values.")
            logger.debug(f"Sampled: {sop_class}")
            avoid_types.clear()
        except (rasp_utils.EmptyScopeError, SamplingError, ValueError) as e:
            if isinstance(e, ValueError) and not e.args[0] in ["key is None!", "query is None!"]:
                raise # reraise other ValueErrors

            logger.debug(f"Failed to sample {sop_class}, retrying. {e}")
            # TODO: maybe I should return the error type and message to
            # collect stats on most common errors
            avoid_types.add(sop_class)
        return avoid_types
    
    def run(self, x):
        """Run the RASP program on a single input."""
        return self.scope[-1](x)
    
    def current_length(self):
        return rasp_utils.count_sops(self.scope[-1])


def sample(
    rng: np.random.Generator,
    program_length: int,
    **sampler_kwargs,
) -> rasp.SOp:
    """Sample a RASP program.
    Args:
        rng: numpy random number generator
        program_length: length of the program in SOps
    """
    start = time()
    sampler = Sampler(rng, **sampler_kwargs)
    avoid = set()
    while sampler.current_length() != program_length:
        avoid = sampler.try_to_add_sop(avoid)
        if time() - start > 30:
            logger.info("Sampling took too long, resampling.")
            return sample(rng, program_length=program_length, **sampler_kwargs)
    program = sampler.scope[-1]
    program = rasp.annotate(program, length=sampler.current_length())

    try:
        perform_checks(program, EXTRA_TEST_INPUTS)
    except SamplingError as e:
        logger.info(f"Failed checks, resampling. {e}")
        return sample(rng, program_length=program_length, **sampler_kwargs)

    logger.debug(f"(sample) Size of scope: {len(sampler.scope)}")
    return program
