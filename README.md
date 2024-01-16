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


## TODOs