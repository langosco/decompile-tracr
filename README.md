# Program Generator for RASP
## Installation
Clone this repo, then
```
pip install -e .
```

## Generating Programs

```python
from rasp_generator import sampling, utils

sampler = sampling.ProgramSampler(validate_compilation=True)
sampler.sample()

# run the program
print(sampler.program([1,2,3,4]))

# print the program
utils.print_program(sampler.program)

# print the program, tracking SOp values at every step
utils.print_program(sampler.program, test_input=[1,2,3,4])
```


## Status
### Remaining problems
- sometimes categorical Aggregate is hard to sample (reaches max retries)
- sometimes a sampled program doesn't depend on rasp.tokens
- sometimes programs are trivial (e.g. output is all Nones)
- I suspect that sometimes the output is constant.


### TODOS
- [x] remove SOps that are all (or mostly) None
- [ ] use multiple test inputs; maybe remove / downweight constant SOps if frequent
- [ ] add tests
- [x] send PR for tracr allowing floats in rasp.Aggregate
- [x] allow for floats in categorical SOps (after PR is accepted)
- [ ] collect statistics on generated SOps
- [ ] set up profiling for sampler
- [ ] upweight rasp.tokens to avoid sampling programs that don't depend on rasp.tokens
- [ ] figure out design for setting weights for sampling
- [ ] sanity check rasp.Map simplifications (and maybe fix repr)
- [ ] it's kind of unprincipled to just pick the last sampled SOp as the program


### Biasing the sampler
How should we set weights for the sampler? Some possible criteria that maybe should
influence the sampling likelihood of a SOp:
- operation type (e.g. Map vs SequenceMap)
- SOp type (float vs bool vs categorical)
- values (eg disprefer SOps constant across inputs and/or across sequence)
- input SOps (e.g. prefer categorical input SOps over bool for Aggregate)
- operation args (e.g. function for Map, predicate for Selector)
- program depth (e.g. prefer to depend on rasp.tokens early on but less so later)
- relative program depth (e.g. prefer to sample input args from 'close by' SOps)