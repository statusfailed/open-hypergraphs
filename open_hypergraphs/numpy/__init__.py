from typing import Type
import open_hypergraphs.finite_function as f
import open_hypergraphs.hypergraph as h
import open_hypergraphs.open_hypergraph as o

from open_hypergraphs.array.numpy import NumpyBackend

class FiniteFunction(f.FiniteFunction):
    Array = NumpyBackend

class IndexedCoproduct(f.IndexedCoproduct):
    @classmethod
    def FiniteFunction(cls) -> Type[f.FiniteFunction]:
        return FiniteFunction

class Hypergraph(h.Hypergraph):
    @classmethod
    def IndexedCoproduct(cls) -> Type[f.IndexedCoproduct]:
        return IndexedCoproduct

class OpenHypergraph(o.OpenHypergraph):
    @classmethod
    def Hypergraph(cls) -> Type[h.Hypergraph]:
        return Hypergraph
