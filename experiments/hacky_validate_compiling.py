from collections import defaultdict
from tracr.rasp import rasp
from tracr.compiler import compiling
from rasp_generator import sampling
import numpy as np


rng = np.random.default_rng()

def compile_rasp_to_model(sop: rasp.SOp, vocab={0,1,2,3,4}, max_seq_len=5, compiler_bos="BOS"):
    return compiling.compile_rasp_to_model(
        sop,
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=compiler_bos
    )


def sample_test_input(rng, vocab={0,1,2,3,4}, max_seq_len=5):
    seq_len = rng.choice(range(1, max_seq_len+1))
    return rng.choice(list(vocab), size=seq_len).tolist()


test_inputs = [sample_test_input(rng) for _ in range(100)]
test_inputs += [[0], [0,0,0,0,0], [4,4,4,4], [0,1,2,3]]



n_samples = 10
errs = defaultdict(list)
results = []


print(f"Sampling {n_samples} programs...")
for _ in range(n_samples):
    try:
        sampler = sampling.ProgramSampler(rng=rng)
        retries = sampler.sample(n_sops=30)
        errs['retries'] += retries
        results.append(dict(program=sampler.program))
    except Exception as err:
        errs['sampling'].append(err)
    

print(f"Done sampling. Total programs sampled (minus sampling failures): "
      f"{len(results)}")
print("Total sampling retries:", len(errs['retries']))
print("Total sampling failures:", len(errs['sampling']))
print("Now compiling and validating...")
print()


for r in results:
    if 'program' not in r:
        continue

    # compile
    try:
        model = compile_rasp_to_model(r['program'])
        r['compiled'] = True
    except Exception as err:
        errs['compilation'].append(err)
        r['compilation_error'] = err
        continue

    # validate
    try:
        for test_input in test_inputs:
            rasp_out = r['program'](test_input)
            rasp_out_sanitized = [0 if x is None else x for x in rasp_out]
            assert isinstance(test_input, list)
            model_out = model.apply(["BOS"] + test_input).decoded[1:]
            if not np.allclose(model_out, rasp_out_sanitized, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Compiled program {r['program'].label} does not match RASP output.\n"
                                    f"Compiled output: {model_out}\n"
                                    f"RASP output: {rasp_out}\n"
                                    f"Test input: {test_input}.")
    except Exception as err:
            errs['validation'].append(err)
            r['validation_error'] = err


total_compiled = len([r for r in results if 'compiled' in r])
print("Done compiling.")
print("Total programs compiled:", total_compiled)
print("Total compilation errors:", len(errs['compilation']))
print("Now validating...")
print()


print("Total programs compiled validly (relative to test inputs):", 
      total_compiled - len(errs['validation']))
print("Validation errors:", len(errs['validation']))
print()