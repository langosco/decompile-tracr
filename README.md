# Program Generator for RASP
## Installation

Clone this repo, install [tracr](https://github.com/google-deepmind/tracr)
and dependancies, then pip install:

```
pip install -e .
```

To install GPU support for Jax, first update [CUDA toolkit](https://developer.nvidia.com/cuda-downloads),
then install Jax with CUDA support:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Generating Programs

```python
import numpy as np
from rasp_generator import sampling, utils

rng = np.random.default_rng(0)

sampler = sampling.ProgramSampler(rng=rng)
program, errs = sampler.sample(min_length=3, max_length=7)

# run the program
print(program([1,2,3,4]))

# print the program
utils.print_program(program)

# print the program, tracking SOp values at every step
utils.print_program(program, test_input=[1,2,3,4])

# validate correctness
sampler.validate(program, val_compile=True)
```

## Compiling and Tokenizing
```python
from rasp_generator import sampling, utils
from tokenizer import compile_and_tokenize

sampler = sampling.ProgramSampler()
program, _ = sampler.sample()
model, tokens = compile_and_tokenize.compile_and_tokenize(program)
```
The tokens are returned as a dictionary indexed by layer.

Under the hood, a RASP program is tokenized by first compiling it, then representing it as a dictionary with one RASP sub-program per layer of the compiled transformer.

A program is represented as a sequence of vocabulary elements (see `tokenizer/vocab.py`), for example
```python
sop_1 = rasp.Map("lambda x: x + 1", rasp.tokens)
sel_2 = rasp.Select(sop_1, rasp.tokens, rasp.Comparison.EQ)
sop_3 = rasp.Aggregate(sel_2, sop_1)
```
becomes the dictionary
```
{0: [],
 1: ['SOp_1', 'categorical', 'Map', 'lambda x: x + 1', 'tokens', 'SEP'],
 2: ['Selector_0',
  'Select',
  'EQ',
  'SOp_1',
  'tokens',
  'SEP',
  'SOp_0',
  'categorical',
  'Aggregate',
  'Selector_0',
  'SOp_1',
  'SEP'],
 3: []}
```


## Status
### Remaining problems
- sometimes categorical Aggregate is hard to sample (reaches max retries)
- sometimes a sampled program doesn't depend on rasp.tokens


### TODOS
- [ ] add tests for compiling
- [x] remove SOps that are all (or mostly) None
- [x] use multiple test inputs; 
- [ ] decide whether to remove / downweight constant SOps if frequent
- [x] send PR for tracr allowing floats in rasp.Aggregate
- [x] allow for floats in categorical SOps (after PR is accepted)
- [x] collect statistics on generated SOps
- [ ] set up profiling for sampler
- [x] upweight rasp.tokens to avoid sampling programs that don't depend on rasp.tokens
- [x] figure out design for setting weights for sampling
- [x] sanity check rasp.Map simplifications (and maybe fix repr)
- [x] it's kind of unprincipled to just pick the last sampled SOp as the program
- [x] generate 'draft' dataset of pairs (weights, tokenized program)
- [ ] for short programs, consider enumerating all possible programs exhaustively
- [ ] Ensure compatibility with 'base' tracr library
- [x] Accellerate saving/loading by using json for tokens/weights and
        only using pickle/dill for rasp programs and compiled models.
- [ ] Do linear sequencemaps compile to the same model if we switch order of
        expressions and weights are equal? what about symmetric sequencemaps?
- [ ] Test determinism under fixed rng seed


Tests for sampled programs
- [ ] compiled model valid on test inputs
- [ ] outputs are not constant in input
- [ ] outputs are within a reasonable range, eg [-1e6, 1e6]
- [ ] good distribution between SOp types and classes (see section 'Biasing the sampler')
- [ ] variable names in tokenized program have same order as sop names
- [ ] order in residual stream compiled model is the same as variable names

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
