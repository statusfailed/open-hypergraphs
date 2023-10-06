from dataclasses import dataclass
from open_hypergraphs.finite_function import *
from open_hypergraphs.hypergraph import *

@dataclass
class OpenHypergraph:
    """ An OpenHypergraph is a ... """
    s: AbstractFiniteFunction
    t: AbstractFiniteFunction
    H: Hypergraph

    def type(self):
        return (
            self.s >> self.H.w,
            self.t >> self.H.w
        )

    @classmethod
    def identity(cls, w, x):
        s = t = cls._Fun.identity(w.source)
        H = cls._Graph.discrete(w, x)
        return cls(s, t, H)
