from typing import Any
from abc import abstractmethod
from dataclasses import dataclass
from open_hypergraphs.finite_function import AbstractFiniteFunction, AbstractIndexedCoproduct

@dataclass
class Hypergraph:
    s: AbstractIndexedCoproduct # sources : Σ_{x ∈ X} arity(e) → W
    t: AbstractIndexedCoproduct # targets : Σ_{x ∈ X} coarity(e) → W
    w: AbstractFiniteFunction   # hypernode labels w : W → Σ₀
    x: AbstractFiniteFunction   # hyperedge labels x : X → Σ₁

    # The type of hypergraphs is parametrised over the type of finite functions.
    @classmethod
    @property
    @abstractmethod
    def Fun(cls):
        ...

    # number of vertices
    @property
    def W(self):
        return self.w.source

    # number of edges
    @property
    def X(self):
        return self.x.source

    def __post_init__(self):
        assert self.s.target == self.W
        assert self.t.target == self.W

        assert type(self.s) == self.Fun.IndexedCoproduct
        assert type(self.t) == self.Fun.IndexedCoproduct

        assert len(self.s.sources) == self.X
        assert len(self.t.sources) == self.X

    @classmethod
    def empty(cls, w, x):
        """ Construct the empty hypergraph with no hypernodes or hyperedges """
        e = cls.Fun.initial(0)
        return cls(e, e, w, x)

    @classmethod
    def discrete(cls, w, x):
        """ The discrete hypergraph, consisting of only hypernodes """
        if len(x) > 0:
            raise ValueError(f"Hypergraph.discrete(w, x) must be called with len(x) == 0, but x : {x.source} → {x.target}")

        return cls(
            s = cls.Fun.IndexedCoproduct.initial(len(w)),
            t = cls.Fun.IndexedCoproduct.initial(len(w)),
            w = w,
            x = x)

    def coproduct(f, g):
        """ A coproduct of hypergraphs is pointwise on the components """
        assert f.w.target == g.w.target
        assert f.x.target == g.x.target
        return type(f)(f.s @ g.s, f.t @ g.t, f.w + g.w, f.x + g.x)

    def __matmul__(f, g):
        return f.coproduct(g)

    def coequalize_vertices(self, q: AbstractFiniteFunction):
        assert q.source == self.W
        u = universal(q, self.w)
        if not (q >> u) == self.w:
            raise ValueError(f"Universal morphism doesn't make {q};{u}, {self.w} commute. Is q really a coequalizer?")

        return type(self)(self.s >> q, self.t >> q, u, self.x)

def universal(q: AbstractFiniteFunction, f: AbstractFiniteFunction):
    """
    Given a coequalizer q : B → Q of morphisms a, b : A → B
    and some f : B → B' such that f(a) = f(b),
    Compute the universal map u : Q → B'
    such that q ; u = f.
    """
    target = f.target
    u = q._Array.zeros(q.target, dtype=f.table.dtype)
    # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
    # same memory location in the 'u' array can happen in parallel, with an
    # arbitrary write succeeding.
    # Note that this doesn't affect correctness because q and f are co-forks,
    # and q is a coequalizer.
    # However, this won't perform well on e.g., GPU hardware. FIXME!
    u[q.table] = f.table
    return type(f)(target, u)
