from typing import Self, List, Protocol, Any, Type
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

    def is_discrete(self) -> bool:
        """ Check if a hypergraph is discrete """
        return len(self.s) == 0 and len(self.t) == 0 and len(self.x) == 0

    def coproduct(G: 'Hypergraph', H: 'Hypergraph') -> 'Hypergraph':
        """ A coproduct of hypergraphs is pointwise on the components """
        assert G.w.target == H.w.target
        assert G.x.target == H.x.target
        return type(G)(G.s @ H.s, G.t @ H.t, G.w + H.w, G.x + H.x)

    @classmethod
    def coproduct_list(cls, Hs: List['Hypergraph']) -> Self:
        """ Compute the n-fold coproduct of hypergraphs for n > 0 """
        if len(Hs) == 0:
            raise ValueError("Hs must be a nonempty list")
        # note: s, t are tensored since we need to renumber hypernodes
        IndexedCoproduct = cls.IndexedCoproduct()
        FiniteFunction   = IndexedCoproduct.FiniteFunction()
        return cls(
            s=IndexedCoproduct.tensor_list([H.s for H in Hs]),
            t=IndexedCoproduct.tensor_list([H.t for H in Hs]),
            w=FiniteFunction.coproduct_list([H.w for H in Hs]),
            x=FiniteFunction.coproduct_list([H.x for H in Hs]))


    def __add__(G: 'Hypergraph', H: 'Hypergraph') -> 'Hypergraph':
        return G.coproduct(H)

    def coequalize_vertices(self: 'Hypergraph', q: FiniteFunction) -> 'Hypergraph':
        assert self.W == q.source
        u = q.coequalizer_universal(self.w)
        s = self.s.map_values(q)
        t = self.t.map_values(q)
        return type(self)(s, t, u, self.x) # type: ignore

    def permute(self, w: FiniteFunction, x: FiniteFunction) -> 'Hypergraph':
        """ Permute the nodes and edges of a Hypergraph """
        if w.source != self.W or w.target != self.W:
            raise ValueError(f"w must be a permutation of type {self.W} → {self.W}")
        if x.source != self.X or x.target != self.X:
            raise ValueError(f"x must be a permutation of type {self.X} → {self.X}")
        # (w, x) is an ACSet/hypergraph morphism G ⇒ H
        # So we must have that both w and x preserve labels.
        # i.e.,
        #   G.w == w >> H.w
        # whence:
        #   H.w == w⁻¹ ; G.w
        # similarly
        #   H.x == x⁻¹ ; G.x
        #
        # source/target maps
        #   Σ_{i ∈ X} f(i)
        # become
        #   Σ_{i ∈ X} g(x(i)) = Σ_{i ∈ X} f(i)
        x = x.argsort()
        return type(self)(
            s = self.s.map_indexes(x).map_values(w),
            t = self.t.map_indexes(x).map_values(w),
            w = w.argsort() >> self.w,
            x = x >> self.x)

class HasHypergraph(HasIndexedCoproduct):
    @classmethod
    @abstractmethod
    def Hypergraph(cls) -> Type[Hypergraph]:
        ...

    @classmethod
    def IndexedCoproduct(cls) -> Type[IndexedCoproduct]:
        return cls.Hypergraph().IndexedCoproduct()

    @classmethod
    def FiniteFunction(cls) -> Type[FiniteFunction]:
        return cls.IndexedCoproduct().FiniteFunction()

@dataclass
class HypergraphArrow:
    # source and target hypergraphs
    source: Hypergraph
    target: Hypergraph

    # components of a natural transformation on w and x
    w: FiniteFunction
    x: FiniteFunction

    def __post_init__(self):
        # TODO: have we checked everything that needs to commute?
        f = self
        G = self.source
        H = self.target

        assert f.w.source == G.W
        assert f.w.target == H.W

        assert f.x.source == G.X
        assert f.x.target == H.X

        # wire labels and operation labels preserved
        assert G.w == f.w >> H.w
        assert G.x == f.x >> H.x

        # The types of operations should be preserved under the mapping
        assert G.s.values >> G.w == H.s.indexed_values(f.x) >> H.w
        assert G.t.values >> G.w == H.t.indexed_values(f.x) >> H.w

    # def inj0(G₀: Hypergraph, G₁: Hypergraph):
        # Fun = type(G₀).FiniteFunction()
        # return HypergraphMorphism(
                # source=G₀,
                # target=G₀ + G₁,
                # w = Fun.inj0(len(G₀.W), len(G₁.W)),
                # x = Fun.inj0(len(G₀.X), len(G₁.X)))

    # def inj1(G₀: Hypergraph, G₁: Hypergraph):
        # Fun = type(G₀).FiniteFunction()
        # return HypergraphMorphism(
                # source=G₀,
                # target=G₀ + G₁,
                # w = Fun.inj1(len(G₀.W), len(G₁.W)),
                # x = Fun.inj1(len(G₀.X), len(G₁.X)))
