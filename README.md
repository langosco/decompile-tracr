# Program Generator for RASP
## Installation
First install 
[Jax](https://github.com/google/jax) and
[Tracr](https://github.com/google-deepmind/tracr).

Then pip install this repository:

```
pip install -e .
```
# Overview
* `decompile_tracr/sampling`: code for generating random RASP programs.
* `decompile_tracr/tokenizing`: code for mapping to and from a token representation for RASP programs.
* `decompile_tracr/dataset`: code for generating a full dataset to train a meta-model.

To build a dataset of sampled programs + compiled weights, run
```bash
N_DATAPOINTS=1000
python -m decompile_tracr.dataset.make_dataset --ndata $N_DATAPOINTS --config range
```

The `--config` argument refers to one of the DatasetConfig presets
in `decompile_tracr/dataset/config.py`.



# Usage
## Sample RASP Program

```python
import numpy as np
from decompile_tracr.sampling import sampling, rasp_utils

rng = np.random.default_rng()
program = sampling.sample(rng=rng, program_length=5)

# Run the program
print(program([1,2,3,4]))

# Print the program:
rasp_utils.print_program(program)

# Trace values given a test input:
rasp_utils.print_program(program, test_input=[1,2,3,4])
```

## Tokenize RASP Program
```python
from decompile_tracr.tokenizing import tokenizer

tokens: list[int] = tokenizer.tokenize(program)
decoded: list[str] = tokenizer.decode(tokens)

print("Tokens:", tokens)
print("Named tokens:", decoded)

# Recover program from tokens
detokenized_program = tokenizer.detokenize(tokens)
```

## Compile RASP Program
To compile RASP programs to transformer weights, we use [Tracr](https://github.com/google-deepmind/tracr).

```python
from tracr.compiler import compiling
assembled_model = compiling.compile_rasp_to_model(
    program,
    vocab=set(range(5)),
    max_seq_len=5,
)

print(assembled_model.apply(['compiler_bos', 1, 2, 3, 4]).decoded)
```
Note that not all RASP programs are not supported by Tracr.


## Status
### Enhancements
- [x] add tests for compiling
- [x] remove SOps that are all (or mostly) None
- [x] use multiple test inputs; 
- [x] decide whether to remove / downweight constant SOps if frequent
- [x] send PR for tracr allowing floats in rasp.Aggregate
- [x] allow for floats in categorical SOps (after PR is accepted)
- [x] collect statistics on generated SOps
- [x] set up profiling for sampler
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
- [ ] Investigate why different InvalidValueSet errors happen at tokenize time vs compile time


Tests for sampled programs
- [x] compiled model valid on test inputs
- [x] outputs are not constant in input
- [x] program is not the identity
- [x] outputs are within a reasonable range, eg [-1e6, 1e6]
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
