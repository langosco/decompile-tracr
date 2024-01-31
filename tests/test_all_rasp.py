# %%

# Assume that /script/rebuild_dataset.sh has been run to generate the data
# and that the data has been saved to ../data/deduped/pytest/data.pkl

import pickle
from tqdm import tqdm
import numpy as np
import numpy as np
import random
import rasp_generator
from rasp_generator.utils import count_sops
import matplotlib.pyplot as plt
from tracr.compiler import rasp_to_graph
import matplotlib.pyplot as plt


# import os

# # Set JAX to use CPU
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

__MAIN__ = "__main__"

# %%
from rasp_tokenizer import data_utils
from rasp_generator.utils import print_program
    
def generate_random_input(model):
    size_input = model.input_encoder._max_seq_len - 1  # account for BOS token
    bound_input = model.input_encoder.vocab_size - 2  # account for compiler pad token and BOS token

    return np.random.randint(0, bound_input, size_input).tolist()

    
def get_model_output(input, rasp, model, skip_model=False):
    raw_output_rasp = rasp(input)
    output_rasp = np.array(raw_output_rasp)
    
    output_model = None
    if not skip_model:
        raw_output_model = model.apply(["compiler_bos"] + input).decoded[1:]
        output_model = np.array(raw_output_model)

    return output_rasp, output_model

def generate_batch_outputs(batch_size, rasp, model, skip_model=False):
    batch_outputs_rasp = []
    batch_outputs_model = []

    for _ in range(batch_size):
        random_input = generate_random_input(model)
        output_rasp, output_model = get_model_output(random_input, rasp, model, skip_model=skip_model)
        batch_outputs_rasp.append(output_rasp)

        if model is not None:
            batch_outputs_model.append(output_model)

    batch_outputs_model = np.array(batch_outputs_model)
    batch_outputs_rasp = np.array(batch_outputs_rasp)
    return batch_outputs_rasp, batch_outputs_model

def test_func_equiv(data,num_inputs=100,seed=None,atol = 0.1,rtol = 0.1,verbose = False):
    """
    Tests for functional equivalence between RASP and compiled model
    on randomly sampled inputs.
    """
    random.seed(seed)
    np.random.seed(seed)
        
    # if data is None:
    #     with open(data_path, "rb") as file:
    #         data = pickle.load(file)
        
    for i, datapoint in tqdm(enumerate(data), total=len(data)):
        
        for _ in range(num_inputs):
            
            model = datapoint['model']
            rasp = datapoint['rasp']
            
            input = generate_random_input(model)
            output_rasp, output_model = get_model_output(input, rasp, model)
            
            error = False
      
            # check if any of the inputs are None
            if any([x is None for x in output_rasp]):
                raise ValueError(f"Output RASP {output_rasp}\n",
                                 f"Output model {output_model}\n",
                                 f"RASP contains None: for model '{i}' and input '{input}':\n")
            
            # if model is not floating, convert
            if not np.issubdtype(output_model.dtype, np.floating):
                output_model = output_model.astype(np.float32)
                
            if not np.issubdtype(output_rasp.dtype, np.floating):
                output_rasp = output_rasp.astype(np.float32)
    
            if not np.allclose(output_rasp, output_model, atol=atol, rtol=rtol):
                if verbose:
                    print_program(rasp, full=True)
                raise ValueError(f"Outputs are not close for model '{i}' and input '{input}':\n",
                                    f"Output RASP: {output_rasp}\n",
                                    f"Output Model: {output_model}\n")

# %%
def test_non_constant_program(data,
                            num_inputs=10, 
                                seed=None,
                                epsilon = 0.01,
                                verbose = False):
    """
    Tests that programs do not produce constant outputs.
    """
    random.seed(seed)
    np.random.seed(seed)
        
    # if data is None:
    #     with open(data_path, "rb") as file:
    #         data = pickle.load(file)
    count_constant = 0
        
    for i, datapoint in tqdm(enumerate(data), total=len(data)):
        rasp = datapoint['rasp']
        model = datapoint['model']
        rasp_outputs, _ = generate_batch_outputs(num_inputs, rasp, model, skip_model=True)
        variance = np.var(rasp_outputs, axis = 0)
        #print(f"Variance for model '{i}':\n", variance.max())
        if variance.mean() < epsilon:
            print(f"Model {i} is constant:")
            if verbose:
                print("=====================================")
                print_program(rasp, full=True)
                print("")
            count_constant += 1
            
    print(f"Number of constant models: {count_constant}/{len(data)}")
            #raise ValueError(f"Model is constant '{i}':\n")


def test_weights_range(data,lower_bound=-100, upper_bound=100, verbose = False):
    """
    Tests that weights are within a certain range.
    """
    for i, datapoint in tqdm(enumerate(data), total=len(data)):
        model = datapoint['model']
        for name, param in model.params.items():
            param_min, param_max = np.inf, -np.inf
            for key, value in param.items():
                if value is None:
                    continue
                param_min = min(param_min, value.min())
                param_max = max(param_max, value.max())
            if param_min < lower_bound or param_max > upper_bound:
                print(f"Model {i}, param {name}, is out of bounds: min {param_min} max {param_max}\n")

def test_outputs_range(data,lower_bound=-100, upper_bound=100, num_inputs = 100, seed=None, verbose = False):
    """
    Tests that outputs are within a certain range.
    """
    random.seed(seed)
    np.random.seed(seed)
    for i, datapoint in tqdm(enumerate(data), total=len(data)):
        model = datapoint['model']
        rasp  = datapoint['rasp']
        out_rasp, out_model = generate_batch_outputs(num_inputs, rasp, model, skip_model=False)
        if verbose:
            print(f"model {i} min {out_model.min()} max{out_model.max()}")
        if out_model.min() < lower_bound or out_model.max() > upper_bound:
            print(f"Model {i} is out of bounds: min {out_model.min()} max {out_model.max()}\n")


def program_length_distribution(data):
    """
    Plots a histogram of program lengths.
    No explicit test, best to just look at the plot.
    """
    prog_lens = []
    for i, datapoint in tqdm(enumerate(data), total=len(data)):
        rasp = datapoint['rasp']
        prog_lens.append(count_sops(rasp))

    # Calculate histogram
    counts, bin_edges = np.histogram(prog_lens, bins=range(min(prog_lens), max(prog_lens) + 2))

    # Plot the histogram
    plt.bar(bin_edges[:-1], counts, width=1, edgecolor='black')
    plt.xlabel('Program Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Program Lengths')
    plt.show()

    return counts.tolist()

def plot_operation_histogram(data):
    """
    Plots a histogram of the frequency of each operation across a list of programs.
    No explicit test, best to just look at the plot.
    """
    rasp_programs = [x['rasp'] for x in data]
    print(len(rasp_programs))
    prefix_count = {}
    for program in rasp_programs:
        graph = rasp_to_graph.extract_rasp_graph(program)
        
        for op in list(graph.graph.nodes):
            prefix = op.split("_")[0]
            if prefix in ["tokens", "indices"]:
                continue
            if prefix in prefix_count:
                prefix_count[prefix] += 1
            else:
                prefix_count[prefix] = 1
    
    # Plotting the histogram
    operations = list(prefix_count.keys())
    counts = list(prefix_count.values())
    plt.bar(operations, counts, width=1, edgecolor='black')
    plt.xlabel('Operation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Operations')
    plt.show()
    
    return prefix_count


# %%

if __name__ == __MAIN__:
    data = data_utils.load_batches(keep_aux=True)  # loads data generated by generate_data.py, including model & rasp
    deduped = data_utils.load_deduped(name = "pytest", flatten=False, keep_aux=True)  # loads data post-deduplication
    test_non_constant_program(deduped, num_inputs = 100, seed=42, verbose = True)
    test_func_equiv(deduped, seed=42, verbose = True)
    test_weights_range(deduped, verbose = True)
    test_outputs_range(deduped, num_inputs = 100, seed = 43)
    program_length_distribution(deduped)
    plot_operation_histogram(deduped)

# %%
