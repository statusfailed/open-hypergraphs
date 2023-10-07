from dataclasses import dataclass
from open_hypergraphs.finite_function import *
from open_hypergraphs.hypergraph import *

@dataclass
class OpenHypergraph(HasHypergraph):
    """ An OpenHypergraph is a ... """
    s: FiniteFunction
    t: FiniteFunction
    H: Hypergraph

    def type(self):
        return (
            self.s >> self.H.w,
            self.t >> self.H.w
        )

    @classmethod
    def identity(cls, w, x):
        s = t = cls.FiniteFunction().identity(w.source)
        H = cls.Hypergraph().discrete(w, x)
        return cls(s, t, H)
