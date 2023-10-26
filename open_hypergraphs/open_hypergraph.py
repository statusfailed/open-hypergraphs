from dataclasses import dataclass
from open_hypergraphs.finite_function import FiniteFunction, DTYPE
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
    def identity(cls, w, x, dtype=DTYPE):
        if x.source != 0:
            raise ValueError("x.source must be 0, but was {x.source}")
        s = t = cls.FiniteFunction().identity(w.source, dtype=dtype)
        H = cls.Hypergraph().discrete(w, x, dtype=dtype)
        return cls(s, t, H)

    def twist(a: FiniteFunction, b: FiniteFunction):
        raise NotImplementedError("TODO")

    ##############################
    # Tensor product
    def tensor(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return type(f)(
            s = f.s @ g.s,
            t = f.t @ g.t,
            H = f.H + g.H)

    def __matmul__(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return f.tensor(g)

    def compose(f: 'OpenHypergraph', g: 'OpenHypergraph'):
        assert f.target == g.source
        h = f @ g
        q = f.t.inject0(g.H.W).coequalizer(g.s.inject1(f.H.W))
        return type(f)(
            s = f.s.inject0(g.H.W) >> q,
            t = g.t.inject1(f.H.W) >> q,
            H = h.H.coequalize_vertices(q))

    def __rshift__(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return f.compose(g)
