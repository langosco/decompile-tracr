
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
from dataclasses import dataclass
import numpy as np
from tracr.rasp import rasp
from tracr.compiler import validating
from decompile_tracr.sampling import map_primitives
from decompile_tracr.sampling import rasp_utils
from decompile_tracr.sampling.validate import is_valid
from decompile_tracr.dataset.logger_config import setup_logger

logger = setup_logger(__name__)


rng = np.random.default_rng(42)
TEST_INPUTS = [rasp_utils.sample_test_input(rng) 
               for _ in range(5)]
TEST_INPUTS += [[0], [0,0,0,0,0], [1,2,3,4]]
TEST_INPUTS = set([tuple(x) for x in TEST_INPUTS])

EXTRA_TEST_INPUTS = [rasp_utils.sample_test_input(rng)
                        for _ in range(50)]


class SamplingError(Exception):
    """Raised when the sampler fails to sample a program
    satisfying the given constraints.
    This error is raises stochastically, so it's (usually) 
    not indicative of a bug.
    """
    pass


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
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.scope = [
            rasp_utils.annotate_type(rasp.tokens, "categorical"),
            rasp_utils.annotate_type(rasp.indices, "categorical"),
        ]
        self.past = [{0}, {1}]
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
            return idx, self.scope[idx]
        else:
            return idx.tolist(), [self.scope[i] for i in idx]

    def add_map(self):
        """Sample a map. A map applies a function elementwise to a SOp.
        The input SOps can be categorical, float, or bool."""
        idx, sop_in = self.sample_from_scope()
        fn, output_type = map_primitives.get_map_fn(self.rng, sop_in.annotations["type"])
        sop_out = rasp.Map(fn, sop_in, simplify=False)
        self.scope.append(rasp_utils.annotate_type(sop_out, type=output_type))
        self.past.append({len(self.past)} | self.past[idx])

    def add_sequence_map(self):
        """Sample a sequence map. A SM applies a function elementwise to
        two categorical SOps. The output is always categorical."""
        idxs, sops_in = self.sample_from_scope(
            type="categorical", 
            size=2, 
            replace=False,
        )
        fn = self.rng.choice(map_primitives.NONLINEAR_SEQMAP_FNS)
        sop_out = rasp.SequenceMap(fn, *sops_in)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="categorical"))
        self.past.append({len(self.past)} | self.past[idxs[0]] | self.past[idxs[1]])

    def add_linear_sequence_map(self):
        """Sample a linear sequence map. A LNS linearly combines two
        numerical SOps. The output is always numerical."""
        idxs, sops_in = self.sample_from_scope(
            type="float", 
            size=2, 
            replace=False,
        )
        weights = self.rng.choice(
            map_primitives.LINEAR_SEQUENCE_MAP_WEIGHTS, size=2, replace=True)
        weights = [int(w) for w in weights]
        sop_out = rasp.LinearSequenceMap(*sops_in, *weights)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="float"))
        self.past.append({len(self.past)} | self.past[idxs[0]] | self.past[idxs[1]])

    def add_numerical_aggregate(self):
        """
        Tracr puts constraints on select-aggregate operations. This aggregate is 
        sampled to satisfy the constraint that the sop argument is 'boolean',
        that is numerical and only takes values 0 or 1.
        This constraint is necessary for all aggregates with numerical SOps.
        """
        selector, parents = self.get_selector()
        idx, sop_in = self.sample_from_scope(
            type="bool",
            allow_none_values=False,
        )
        parents += [idx]
        sop_out = rasp.Aggregate(selector, sop_in, default=0)
        # TODO: sometimes output can be bool here?
        self.scope.append(rasp_utils.annotate_type(sop_out, type="float"))
        self.past.append({len(self.past)}.union(*[self.past[p] for p in parents]))

    def add_categorical_aggregate(self, max_retries=10):
        """
        Sample an aggregate operation with a categorical SOp. Categorical aggregate
        operations must satisfy the constraint that there is no aggregation (eg averaging).
        This usually means that the selector has width 1, but it could also mean that a width > 1
        selector is used but the output domain of the aggregate is still equal to the input domain.
        """
        idx, sop_in = self.sample_from_scope(
            type="categorical",
            allow_none_values=False,
        )
        selector, parents = self.get_selector()
        parents += [idx]
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
            self.past.append({len(self.past)}.union(*[self.past[p] for p in parents]))

    def add_selector_width(self):
        selector, parents = self.get_selector()
        sop_out = rasp.SelectorWidth(selector)
        self.scope.append(rasp_utils.annotate_type(sop_out, type="categorical"))
        self.past.append({len(self.past)}.union(*[self.past[p] for p in parents]))

    def get_selector(self):
        """Sample a rasp.Select. A select takes two categorical SOps and
        returns a selector (matrix) of booleans. Note this is not an SOp.
        """
        # TODO: allow Selectors with bools (numerical & 0-1) as input?
        idxs, sops_in = self.sample_from_scope(
            type="categorical",
            allow_none_values=False,
            size=2,
            replace=True,
        )
        comparison = self.rng.choice(map_primitives.COMPARISONS)
        selector = rasp.Select(*sops_in, comparison)
        return selector, idxs

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

        sop_class = self.rng.choice(list(set(add_functions) - set(avoid_types)))
        add = add_functions[sop_class]
        try:
            add()
            if any(rasp_utils.fraction_none(self.run(x)) > 0.5 for x in TEST_INPUTS):
                raise SamplingError(f"Sampled SOp {self.scope[-1]} has too many None values.")
            logger.debug(f"Sampled: {sop_class}")
            avoid_types.clear()
        except (rasp_utils.EmptyScopeError, SamplingError):
            logger.debug(f"Failed to sample: {sop_class}")
            # TODO: maybe I should return the error type and message to
            # collect stats on most common errors
            avoid_types.add(sop_class)
        return avoid_types
    
    def run(self, x):
        """Run the RASP program on a single input."""
        return self.scope[-1](x)
    
    def current_length(self):
#        return rasp_utils.count_sops(self.scope[-1])
        return len(self.past[-1])


def sample(
    rng: np.random.Generator,
    program_length: int,
) -> rasp.SOp:
    """Sample a RASP program.
    Args:
        rng: numpy random number generator
        program_length: length of the program in SOps
    """
    sampler = Sampler(rng)
    avoid = set()
    while sampler.current_length() != program_length:
        avoid = sampler.try_to_add_sop(avoid)
    program = sampler.scope[-1]
    program = rasp.annotate(program, length=sampler.current_length())

    valid = True
    for x in EXTRA_TEST_INPUTS:
        if not is_valid(program, x):
            valid = False
            break
    if not valid:
        # resample
        return sample(rng, program_length=program_length)

    logger.debug(f"(sample) Size of scope: {len(sampler.scope)}")
    return program
