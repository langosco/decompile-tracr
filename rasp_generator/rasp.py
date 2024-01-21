from tracr.rasp.rasp import Value, RASPExpr, SOp 
from typing import (Callable, Sequence)


# We reimplement rasp.Map to change one line: when simplify=True and `inner`
# is a Map, we use the compose method on utils.FunctionWithRepr instead of 
# defining a new function. This allows us to keep the repr of the function
# when two Maps are simplified into a single Map.

# TODO: doesn't work because class SOp has a method _eval_by_fn_type that
# doesn't know about our new Map class. Need to reimplement that too.

class Map(SOp):
  """SOp that evaluates the function elementwise on the input SOp.

  Map(lambda x: x + 1, tokens).eval([1, 2, 3]) == [2, 3, 4]
  """

  def __init__(
      self,
      f: Callable[[Value], Value],
      inner: SOp,
      simplify: bool = True,
  ):
    """Initialises.

    Args:
      f: the function to apply elementwise.
      inner: the SOp to which to apply `f`.
      simplify: if True and if `inner` is also a Map, will combine the new map
        and `inner` into a single Map object.
    """
    super().__init__()
    self.f = f
    self.inner = inner

    assert isinstance(self.inner, SOp)
    assert callable(self.f) and not isinstance(self.f, RASPExpr)

    if simplify and isinstance(self.inner, Map):
      # Combine the functions into just one.
      inner_f = self.inner.f
      self.f = self.f.compose(inner_f)  # Use the compose method instead of defining a new function
      self.inner = self.inner.inner

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.inner]
