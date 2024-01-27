from tracr.rasp import rasp
from rasp_generator.utils import FunctionWithRepr


def make_length():
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    return rasp.SelectorWidth(all_true_selector)


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    length = make_length()
    opp_idx = (length - rasp.indices)
    opp_idx = (opp_idx - 1)
    reverse_selector = rasp.Select(
        rasp.indices, opp_idx, rasp.Comparison.EQ)
    return rasp.Aggregate(reverse_selector, sop)


def _frac_prevs(bools: rasp.SOp) -> rasp.SOp:
    """Helper function.
    Count the fraction of previous tokens where bools is True.
    Must be numerical.
    """
    assert rasp.is_numerical(bools)
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    return rasp.numerical(rasp.Aggregate(prevs, bools, default=0))



def _pair_balance(sop: rasp.SOp, open_token: str,
                      close_token: str) -> rasp.SOp:
    """Return fraction of previous open tokens minus the fraction of close tokens.

    (As implemented in the RASP paper.)

    If the outputs are always non-negative and end in 0, that implies the input
    has balanced parentheses.

    Example usage:
    num_l = make_pair_balance(rasp.tokens, "(", ")")
    num_l("a()b(c))")
    >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]

    Args:
    sop: Input SOp.
    open_token: Token that counts positive.
    close_token: Token that counts negative.

    Returns:
    pair_balance: SOp mapping an input to a sequence, where every element
        is the fraction of previous open tokens minus previous close tokens.
    """
    bools_open = rasp.numerical(sop == open_token)
    opens = rasp.numerical(_frac_prevs(bools_open))

    bools_close = rasp.numerical(sop == close_token)
    closes = rasp.numerical(_frac_prevs(bools_close))

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance


def make_pair_balance():
    return _pair_balance(rasp.tokens, 0, 1)

