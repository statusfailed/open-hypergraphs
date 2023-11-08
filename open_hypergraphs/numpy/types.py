from typing import Type
from abc import ABC

import numpy as np

import open_hypergraphs.finite_function as f
import open_hypergraphs.hypergraph as h
import open_hypergraphs.open_hypergraph as o
import open_hypergraphs.functor as functor
import open_hypergraphs.functor.optic as optic

from open_hypergraphs.array.numpy import NumpyBackend

class FiniteFunction(f.FiniteFunction):
    Dtype = np.uint32
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

class Functor(functor.Functor):
    @classmethod
    def OpenHypergraph(cls) -> Type[o.OpenHypergraph]:
        return OpenHypergraph

# NOTE: have to inherit from FrobeniusFunctor to make method resolution work,
# but that means supplying OpenHypergraph method!
class FrobeniusFunctor(functor.FrobeniusFunctor):
    @classmethod
    def OpenHypergraph(cls) -> Type[o.OpenHypergraph]:
        return OpenHypergraph

class Optic(optic.Optic):
    @classmethod
    def OpenHypergraph(cls) -> Type[o.OpenHypergraph]:
        return OpenHypergraph
