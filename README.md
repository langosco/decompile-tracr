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
print(sampler.output([1,2,3,4]))

# print the program
utils.print_program(sampler.output)

# print the program, tracking SOp values at every step
utils.print_program(sampler.output, test_input=[1,2,3,4])
```


## Status
### Remaining problems
- sometimes categorical Aggregate is hard to sample (reaches max retries)
- sometimes a sampled program doesn't depend on rasp.tokens
- sometimes programs are trivial (e.g. output is all Nones)
- I suspect that sometimes the output is constant. TODO: check multiple inputs


### TODOS
- [ ] remove SOps that are all (or mostly) None
- [ ] maybe remove / downweight constant SOps?
- [ ] add tests
- [x] send PR for tracr allowing floats in rasp.Aggregate
- [x] allow for floats in categorical SOps (after PR is accepted)
