from open_hypergraphs.finite_function import *
from open_hypergraphs.hypergraph import *
from open_hypergraphs.open_hypergraph import *

import open_hypergraphs.array.numpy as numpy_backend

class FiniteFunction(AbstractFiniteFunction):
    _Array = numpy_backend

class IndexedCoproduct(AbstractIndexedCoproduct):
    _Fun = FiniteFunction

class Hypergraph(AbstractHypergraph):
    _Fun = FiniteFunction

class OpenHypergraph(AbstractOpenHypergraph):
    _Fun        = FiniteFunction
    _Hypergraph = Hypergraph
