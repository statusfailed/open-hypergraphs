from typing import Any, Type
from typing_extensions import Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from open_hypergraphs.finite_function import FiniteFunction, IndexedCoproduct, HasIndexedCoproduct

@dataclass
class Hypergraph(HasIndexedCoproduct):
    s: IndexedCoproduct # sources : Σ_{x ∈ X} arity(e) → W
    t: IndexedCoproduct # targets : Σ_{x ∈ X} coarity(e) → W
    w: FiniteFunction   # hypernode labels w : W → Σ₀
    x: FiniteFunction   # hyperedge labels x : X → Σ₁

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

        assert type(self.s) == type(self).IndexedCoproduct()
        assert type(self.t) == type(self).IndexedCoproduct()

        assert len(self.s.sources) == self.X
        assert len(self.t.sources) == self.X

    @classmethod
    def empty(cls, w: FiniteFunction, x: FiniteFunction) -> 'Hypergraph':
        """ Construct the empty hypergraph with no hypernodes or hyperedges """
        e = cls.FiniteFunction().initial(0)
        return cls(e, e, w, x)

    @classmethod
    def discrete(cls, w: FiniteFunction, x: FiniteFunction) -> 'Hypergraph':
        """ The discrete hypergraph, consisting of only hypernodes """
        if len(x) > 0:
            raise ValueError(f"Hypergraph.discrete(w, x) must be called with len(x) == 0, but x : {x.source} → {x.target}")

        return cls(
            s = cls.IndexedCoproduct().initial(len(w)),
            t = cls.IndexedCoproduct().initial(len(w)),
            w = w,
            x = x)

    def coproduct(G, H):
        """ A coproduct of hypergraphs is pointwise on the components """
        assert G.w.target == H.w.target
        assert G.x.target == H.x.target
        return type(G)(G.s @ H.s, G.t @ H.t, G.w + H.w, G.x + H.x)

    def __matmul__(G, H):
        return G.coproduct(H)

    def coequalize_vertices(self: 'Hypergraph', q: FiniteFunction) -> 'Hypergraph':
        assert q.source == self.W
        u = universal(q, self.w)
        if not (q >> u) == self.w:
            raise ValueError(f"Universal morphism doesn't make {q};{u}, {self.w} commute. Is q really a coequalizer?")

        # TODO: FIX BUG!!!
        # We need to map self.s.values and self.t.values!
        return type(self)(self.s >> q, self.t >> q, u, self.x) # type: ignore

def universal(q: FiniteFunction, f: FiniteFunction):
    """
    Given a coequalizer q : B → Q of morphisms a, b : A → B
    and some f : B → B' such that f(a) = f(b),
    Compute the universal map u : Q → B'
    such that q ; u = f.
    """
    target = f.target
    u = q.Array.zeros(q.target, dtype=f.table.dtype)
    # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
    # same memory location in the 'u' array can happen in parallel, with an
    # arbitrary write succeeding.
    # Note that this doesn't affect correctness because q and f are co-forks,
    # and q is a coequalizer.
    # However, this won't perform well on e.g., GPU hardware. FIXME!
    u[q.table] = f.table
    return type(f)(target, u)

class HasHypergraph(HasIndexedCoproduct):
    @classmethod
    @abstractmethod
    def Hypergraph(cls) -> Type[Hypergraph]:
        ...
