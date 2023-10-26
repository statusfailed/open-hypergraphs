from dataclasses import dataclass
from open_hypergraphs.finite_function import *
from open_hypergraphs.hypergraph import *

@dataclass
class OpenHypergraph(HasHypergraph):
    """ An OpenHypergraph is a cospan in Hypergraph whose feet are discrete. """
    s: FiniteFunction
    t: FiniteFunction
    H: Hypergraph

    @property
    def source(self):
        return self.s >> self.H.w

    @property
    def target(self):
        return self.t >> self.H.w

    @classmethod
    def identity(cls, w, x):
        if x.source != 0:
            raise ValueError("x.source must be 0, but was {x.source}")
        Fun = cls.FiniteFunction()
        s = t = Fun.identity(w.source)
        H = cls.Hypergraph().discrete(w, x)
        return cls(s, t, H)

    def twist(a: FiniteFunction, b: FiniteFunction):
        raise NotImplementedError("TODO")

    def __rshift__(f, g):
        raise NotImplementedError("TODO")
