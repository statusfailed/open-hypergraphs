from abc import ABC, abstractmethod

from open_hypergraphs.finite_function import FiniteFunction, IndexedCoproduct
from open_hypergraphs.hypergraph import Hypergraph
from open_hypergraphs.open_hypergraph import OpenHypergraph, HasOpenHypergraph

class Functor(HasOpenHypergraph, ABC):
    """ Strict symmetric monoidal hypergraph functors """
    @abstractmethod
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        ...

    @abstractmethod
    def map_arrow(self, f: OpenHypergraph) -> OpenHypergraph:
        ...

    # This shorthand for map_arrow is a bit nicer sometimes
    def __call__(self, f: OpenHypergraph) -> OpenHypergraph:
        return self.map_arrow(f)

# This is not *quite* what we want!
# It only maps the values; but we also need the new sources!
def map_half_spider(Fw: IndexedCoproduct, f: FiniteFunction) -> FiniteFunction:
    return Fw.sources.injections(f)

class FrobeniusFunctor(Functor, ABC):
    # Map a *tensoring of operations*, into an OpenHypergraph
    @abstractmethod
    def map_operations(self, x: FiniteFunction, sources: IndexedCoproduct, targets: IndexedCoproduct) -> OpenHypergraph:
        """ Compute ``F(x₁) ● F(x₂) ● ... ● F(xn)``, where each ``x ∈ Σ₁`` is an operation,
        and ``sources`` and ``targets`` are the *types* of each operation. """
        ...

    # Implement map_arrow based on map_operations!
    def map_arrow(self, f: OpenHypergraph) -> OpenHypergraph:
        # Ff: the tensoring of operations F(x₀) ● F(x₁) ● ... ● F(xn)
        sources = f.H.s.map_values(f.H.w) # source types
        targets = f.H.t.map_values(f.H.w) # target types
        Fx = self.map_operations(f.H.x, sources, targets)

        # Fw is the tensoring of objects F(w₀) ● F(w₁) ● ... ● F(wn)
        Fw = self.map_objects(f.H.w)

        # Signature
        w, x = f.signature()
        # Identity map on wires of F(w)
        i = self.OpenHypergraph().identity(Fw.values, x)

        Fs = map_half_spider(Fw, f.s)
        Fe_s = map_half_spider(Fw, f.H.s.values)
        sx = self.OpenHypergraph().spider(Fs, i.t + Fe_s, i.H.w, x)

        Ft = map_half_spider(Fw, f.t)
        Fe_t = map_half_spider(Fw, f.H.t.values)
        yt = self.OpenHypergraph().spider(i.s + Fe_t, Ft, i.H.w, x)

        return (sx >> (i @ Fx) >> yt)
