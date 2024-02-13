# Description: Convert a string representation to a RASP program,
# i.e. the reverse of rasp_to_str

from tracr.rasp import rasp

from decompile_tracr.sampling.map_primitives import FunctionWithRepr
from decompile_tracr.tokenizing import vocab


def str_to_rasp(
        rasp_str: list[list[str]],
    ) -> rasp.SOp:
    """Convert a string representation to a RASP program.
    """
    rasp_str = [l[1:-1] for l in rasp_str]
    ops = [op for layer in rasp_str
              for op in split_list(layer, vocab.SEP)]
    ops = [op for op in ops if len(op) > 2]
    sops = dict()
    for op in ops:
        name, expr = str_to_rasp_op(op, sops)
        sops[name] = expr
    return expr
    

def split_list(l: list, sep: str):
    """Split a list by a separator.
    """
    out = []
    current = []
    for x in l:
        if x == sep:
            out.append(current)
            current = []
        else:
            current.append(x)
    out.append(current)
    out = [x for x in out if len(x) > 0]
    return out


def str_to_rasp_op(op: list[str], sops: list[rasp.SOp]):
    """Recover a single rasp op from a string representation.
    """
    varname, enc, op_name, *args = op

    if op_name in ["SelectAggregate", "SelectorWidth"]:
        k, q, pred = args[:3]
        k, q = [get_sop_from_name(x, sops) for x in [k, q]]
        pred = rasp.Comparison.__getattr__(pred)
        selector = rasp.Select(k, q, pred)
        if op_name == "SelectAggregate":
            sop = get_sop_from_name(args[3], sops)
            default = None if enc == "categorical" else 0
            out = rasp.Aggregate(selector, sop, default=default)
        elif op_name == "SelectorWidth":
            out = rasp.SelectorWidth(selector)
    elif op_name == "Map":
        f, sop = args
        sop = get_sop_from_name(sop, sops)
        out = rasp.Map(FunctionWithRepr(f), sop, simplify=False)
    elif op_name == "SequenceMap":
        f, x, y = args
        x, y = [get_sop_from_name(s, sops) for s in [x, y]]
        out = rasp.SequenceMap(FunctionWithRepr(f), x, y)
    elif op_name == "LinearSequenceMap":
        x, y, a, b = args
        x, y = [get_sop_from_name(s, sops) for s in [x, y]]
        a, b = int(a), int(b)
        out = rasp.LinearSequenceMap(x, y, a, b)
    else:
        raise ValueError(f"Unknown op: {op_name}")
    
    return varname, rasp.__dict__[enc](out)


def get_sop_from_name(name: str, sops: dict[str, rasp.SOp]):
    if name == "tokens":
        return rasp.tokens
    elif name == "indices":
        return rasp.indices
    elif name in sops:
        return sops[name]
    else:
        raise ValueError(f"Unknown sop: {name}.")




