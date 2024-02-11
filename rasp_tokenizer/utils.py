import fcntl
import jax.flatten_util


def get_params(params: dict, layer_name: str) -> jax.Array:
    """
    params: hk parameters as returned by tracr compiler in model.params.
    layer_name: name of the layer to extract parameters for.
    
    Assume layer_name is in format `layer_n/attn` or `layer_n/mlp`.
    Return parameters for that layer as a 1-d array.
    """
    prefix = f'transformer/{layer_name}'

    if layer_name.endswith('attn'):
        layer_params = [
            params[f"{prefix}/{k}"] for k in ['key', 'query', 'value', 'linear']
        ]
    elif layer_name.endswith('mlp'):
        layer_params = [
            params[f'{prefix}/{k}'] for k in ['linear_1', 'linear_2']
        ]
    else:
        raise ValueError(f'Unknown layer name {layer_name}.')
    
    return jax.flatten_util.ravel_pytree(layer_params)[0]


# usage:
#     params_by_layer = {
#         layername: utils.get_params(model.params, layername) 
#             for layername in by_layer.keys()
#     }




def sequential_count_via_lockfile(countfile="/tmp/counter.txt"):
    """Increment a counter in a file. 
    Use a lockfile to ensure atomicity. If the file doesn't exist, 
    create it and start the counter at 1."""
    try:
        with open(countfile, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            f.seek(0)
            counter_str = f.read().strip()
            counter = 1 if not counter_str else int(counter_str) + 1

            f.seek(0)
            f.truncate()  # Clear the file content
            f.write(str(counter))
            f.flush()

            fcntl.flock(f, fcntl.LOCK_UN)

        return counter
    except ValueError as e:
        raise ValueError(f"Invalid counter value: {counter_str}. {e}")
