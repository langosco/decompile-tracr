from collections import defaultdict
from tracr.rasp import rasp
from tracr.compiler import compiling
from rasp_generator import rasp_utils, sampling
import numpy as np
import jax

# script currently broken after refactoring
# TODO: delete this

rng = np.random.default_rng()

def compile_rasp_to_model(sop: rasp.SOp, vocab={0,1,2,3,4}, max_seq_len=5, compiler_bos="BOS"):
    return compiling.compile_rasp_to_model(
        sop,
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=compiler_bos
    )


test_inputs = [rasp_utils.sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]
n_samples = 200
errs = defaultdict(list)
results = []


def sample_program():
    sampler = sampling.ProgramSampler(rng=rng)
    retries = sampler.sample(n_sops=30)
    return sampler.program, retries



def validate_compiled(program: rasp.SOp, model):
    for test_input in test_inputs:
        rasp_out = r['program'](test_input)
        rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
        assert isinstance(test_input, list)
        model_out = model.apply(["BOS"] + test_input).decoded[1:]
        if not np.allclose(model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3):
            raise ValueError(f"Compiled program {r['program'].label} does not match RASP output.")


def print_info_and_clear_caches():
    print("Step", i)
    total_compiled = sum([r['compiled'] for r in results])
    print("Total programs compiled:", total_compiled)
    print("Total compilation errors:", sum([not r['compiled'] for r in results[:i]]))
    print()
    print("Total programs compiled validly (relative to test inputs):", 
            sum([r['validated'] for r in results]))
    compiled = [r for r in results if r['compiled']]
    print("Validation errors:", sum([not r['validated'] for r in compiled]))
    print()

    jax.clear_caches()
    jax.clear_backends()



print(f"Sampling {n_samples} programs...")
for _ in range(n_samples):
    try:
        program, retries = sample_program()
        errs['retries'] += retries
        results.append(dict(
            program=program,
            compiled=False,
            validated=False,
        ))
    except Exception as err:
        errs['sampling'].append(err)
    

print(f"Done sampling. Total programs sampled (minus sampling failures): "
      f"{len(results)}")
print("Total sampling retries:", len(errs['retries']))
print("Total sampling failures:", len(errs['sampling']))
print("Now compiling and validating...")
print()


for i, r in enumerate(results):
    try:
        model = compile_rasp_to_model(r['program'])
        r['compiled'] = True

        validate_compiled(r['program'], model)
        r['validated'] = True
    except Exception as err:
        pass


    if i % 50 == 0:
        print_info_and_clear_caches()

print("Done compiling.")
print_info_and_clear_caches()