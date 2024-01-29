# %%

# Assume that script/generate_data.py and script/dedupe.py has already been run
# and that the data has been saved to ../data/deduped/test_progs/data.pkl

# import numpy as np
# from rasp_generator import sampling, utils


import pickle
from tqdm import tqdm
import numpy as np
import numpy as np
import random
file_path = "../data/deduped/test_progs/data.pkl"
# import os

# # Set JAX to use CPU
# os.environ["JAX_PLATFORM_NAME"] = "cpu"


# def test_functionality_rasp_and_compiled(num_inputs=10, 
#                                          data_path="../data/deduped/test_progs/data.pkl", 
#                                          size_input=8, 
#                                          seed=None,
#                                          atol = 0.1,
#                                          rtol = 0.1):
#     random.seed(seed)
    
num_inputs = 10
size_input = 8
atol = 0.1
rtol = 0.1

    
with open(file_path, "rb") as file:
    data = pickle.load(file)

for i, datapoint in tqdm(enumerate(data), total=len(data)):
    vocab_size = datapoint['model'].input_encoder.vocab_size
    for _ in range(num_inputs):
        
        model = datapoint['model']
        rasp = datapoint['rasp']
        
        size_input = model.input_encoder._max_seq_len - 1 #account for BOS token
        bound_input = model.input_encoder.vocab_size - 2 #account for compiler pad token and BOS token
        
        
        input = list(np.random.randint(0, bound_input, size_input)) 
        
        print(input, vocab_size, size_input)
        output_model = np.array(model.apply(["compiler_bos"] + input).decoded[1:])
        output_rasp = np.array(rasp(input))
        
        error = False
        if np.issubdtype(output_rasp.dtype, np.bool_) and np.issubdtype(output_model.dtype, np.bool_):
            if not np.all(output_rasp == output_model):
                error = True
        #check if floats
        elif np.issubdtype(output_rasp.dtype, np.floating) and np.issubdtype(output_model.dtype, np.floating):
            if not np.allclose(output_rasp, output_model, atol=atol, rtol=rtol):
                error = True
        
        if error:
            raise ValueError(f"Outputs are not close for model '{i}' and input '{input}':\n"
                                f"Output RASP: {output_rasp}\n"
                                f"Output Model: {output_model}")


#test_functionality_rasp_and_compiled(seed=42)
# %%
