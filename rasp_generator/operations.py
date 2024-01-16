from tracr.rasp import rasp


# Operations:
# - Map
# - SequenceMap
# - LinearSequenceMap (compiled exactly into MLP weights)
# - Select
# - Aggregate
# - SelectorWidth


# Goal: inherit from base rasp classes but add a __repr__ method.
# TODO: also add a type attribute?


class Map(rasp.Map):
    def __init__(self, 
                 function_rep: str, 
                 inner: rasp.SOp, 
                 simplify: bool = True):
        """Represent the function as a string, e.g. 'x+1'."""
        self.function_rep = f"lambda x: {function_rep}"
        fn = eval(self.function_rep)
        super().__init__(fn, inner, simplify=simplify)

    def __repr__(self):
        return f"Map({self.function_rep}, {self.sop})"


class SequenceMap(rasp.SequenceMap):
    def __init__(
            self,
            function_rep: str,
            fst: rasp.SOp,
            snd: rasp.SOp,
    ):
        """Represent the function as a string, e.g. 'x+y'."""
        self.function_rep = f"lambda x, y: {function_rep}"
        fn = eval(self.function_rep)
        super().__init__(fn, fst, snd)

