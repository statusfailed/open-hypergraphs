import open_hypergraphs.finite_function as f
import open_hypergraphs.hypergraph as h
import open_hypergraphs.open_hypergraph as o

import open_hypergraphs.array.numpy as numpy_backend

class FiniteFunction(f.AbstractFiniteFunction):
    Array = numpy_backend

class IndexedCoproduct(f.AbstractIndexedCoproduct):
    Fun = FiniteFunction

class Hypergraph(h.Hypergraph):
    @classmethod
    @property
    def Fun(cls):
        return FiniteFunction

class OpenHypergraph(o.OpenHypergraph):
    Fun        = FiniteFunction
    Hypergraph = Hypergraph

FiniteFunction.IndexedCoproduct = IndexedCoproduct
FiniteFunction.Hypergraph = Hypergraph
FiniteFunction.OpenHypergraph = OpenHypergraph
