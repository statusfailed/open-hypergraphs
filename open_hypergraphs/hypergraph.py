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
        return len(self.x)

    def __post_init__(self):
        assert self.s.target == self.W
        assert self.t.target == self.W

        assert type(self.s) == type(self).IndexedCoproduct()
        assert type(self.t) == type(self).IndexedCoproduct()

        # Number of edges
        assert len(self.s) == self.X
        assert len(self.t) == self.X

    @classmethod
    def empty(cls, w: FiniteFunction, x: FiniteFunction) -> 'Hypergraph':
        """ Construct the empty hypergraph with no hypernodes or hyperedges """
        e = cls.IndexedCoproduct().initial(0)
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

    def coproduct(G: 'Hypergraph', H: 'Hypergraph') -> 'Hypergraph':
        """ A coproduct of hypergraphs is pointwise on the components """
        assert G.w.target == H.w.target
        assert G.x.target == H.x.target
        return type(G)(G.s @ H.s, G.t @ H.t, G.w + H.w, G.x + H.x)

    def __add__(G: 'Hypergraph', H: 'Hypergraph') -> 'Hypergraph':
        return G.coproduct(H)

    def coequalize_vertices(self: 'Hypergraph', q: FiniteFunction) -> 'Hypergraph':
        assert self.W == q.source
        u = q.coequalizer_universal(self.w)
        s = self.s.map_values(q)
        t = self.t.map_values(q)
        return type(self)(s, t, u, self.x) # type: ignore

class HasHypergraph(HasIndexedCoproduct):
    @classmethod
    @abstractmethod
    def Hypergraph(cls) -> Type[Hypergraph]:
        ...
