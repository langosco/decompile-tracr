from tracr.rasp import rasp
from rasp_generator.utils import FunctionWithRepr


def make_length():
    all_true_selector = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    return rasp.SelectorWidth(all_true_selector)


length = make_length()


def make_reverse(sop: rasp.SOp) -> rasp.SOp:
    opp_idx = rasp.SequenceMap(
       FunctionWithRepr("lambda x, y: x - y"), length, rasp.indices)
    opp_idx = rasp.Map(
       FunctionWithRepr("lambda x: x - 1"), opp_idx)
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
    """
    bools_open = rasp.numerical(
       rasp.Map(FunctionWithRepr(f"lambda x: x == {open_token}"), sop))
    opens = _frac_prevs(bools_open)

    bools_close = rasp.numerical(
       rasp.Map(FunctionWithRepr(f"lambda x: x == {close_token}"), sop))
    closes = rasp.numerical(_frac_prevs(bools_close))

    pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
    return pair_balance


def make_pair_balance():
    return _pair_balance(rasp.tokens, 0, 1)


def make_shuffle_dyck(pairs: list = [(0,1), (2,3)]) -> rasp.SOp:
    """Returns 1 if a set of parentheses are balanced, 0 else.
    """
    assert len(pairs) >= 1

    # Compute running balance of each type of parenthesis
    balances = []
    for pair in pairs:
        assert len(pair) == 2
        open_token, close_token = pair
        balance = _pair_balance(
            rasp.tokens, open_token=open_token,
            close_token=close_token)
        balances.append(balance)

    # Check if balances where negative anywhere -> parentheses not balanced
    any_negative = rasp.Map(
        FunctionWithRepr("lambda x: x < 0"),
        balances[0],
    )

    for balance in balances[1:]:
        bal_neg = rasp.Map(
            FunctionWithRepr("lambda x: x < 0"),
            balance,
        )

        any_negative = rasp.SequenceMap(
            FunctionWithRepr("lambda x, y: x or y"),
            any_negative, 
            bal_neg,
        )

    # Convert to numerical SOp
    any_negative = rasp.numerical(
        rasp.Map(FunctionWithRepr("lambda x: x"), any_negative)
    )

    select_all = rasp.Select(rasp.indices, rasp.indices,
                            rasp.Comparison.TRUE)
    has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,
                                            default=0))

    # Check if all balances are 0 at the end -> closed all parentheses
    all_zero = rasp.Map(
        FunctionWithRepr("lambda x: x == 0"),
        balances[0],
    )

    for balance in balances[1:]:
        balance_is_zero = rasp.Map(
            FunctionWithRepr("lambda x: x == 0"),
            balance,
        )

        all_zero = rasp.SequenceMap(
            FunctionWithRepr("lambda x, y: x and y"),
            all_zero, 
            balance_is_zero,
        )

    select_last = rasp.Select(rasp.indices, length - 1,
                            rasp.Comparison.EQ)
    last_zero = rasp.Aggregate(select_last, all_zero)

    not_has_neg = rasp.Map(
        FunctionWithRepr("lambda x: not x"),
        has_neg,
    )
    
    return rasp.SequenceMap(
        FunctionWithRepr("lambda x, y: x and y"),
        last_zero, 
        not_has_neg,
    )
