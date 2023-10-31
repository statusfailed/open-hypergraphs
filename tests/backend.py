# Backends: a choice of concrete implementation for Array, FiniteFunction, etc.
# This exists so test code can be parametrised over this choice, giving a bunch
# of tests "for free" for different array backends.

from typing import Protocol

from open_hypergraphs.finite_function import FiniteFunction, IndexedCoproduct
from open_hypergraphs.hypergraph import Hypergraph
from open_hypergraphs.open_hypergraph import OpenHypergraph

import open_hypergraphs.numpy as numpy

class Backend(Protocol):
    FiniteFunction: FiniteFunction
    IndexedCoproduct: IndexedCoproduct
    Hypergraph: Hypergraph
    OpenHypergraph: OpenHypergraph

################################################################################
# Numpy backend

from tests.strategy.numpy import DEFAULT_DTYPE, arrays, permutations
from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.strategy.hypergraph import HypergraphStrategies
from tests.strategy.open_hypergraph import OpenHypergraphStrategies

# Run tests using the numpy backend
class NumpyBackend(Backend):
    FiniteFunction = numpy.FiniteFunction
    IndexedCoproduct = numpy.IndexedCoproduct
    Hypergraph = numpy.Hypergraph
    OpenHypergraph = numpy.OpenHypergraph

    @classmethod
    def setUpClass(cls):
        # NOTE: we set *global* variables here by overwriting member variables
        # of FiniteFunctionStrategies.
        # We have to use globals because the @given decorators refer to the
        # strategies classes directly; there's no way to parametrise them that
        # I'm aware of.
        # This kinda sucks, but it works reliably because hypothesis is
        # guaranteed to run tests single-threaded.
        # 
        # "arrays" and "permutation_arrays" are Hypothesis strategies for arrays
        # of the chosen array backend.
        FiniteFunctionStrategies.DEFAULT_DTYPE = DEFAULT_DTYPE
        FiniteFunctionStrategies.arrays = arrays
        FiniteFunctionStrategies.permutation_arrays = permutations

        FiniteFunctionStrategies.FiniteFunction = cls.FiniteFunction
        FiniteFunctionStrategies.IndexedCoproduct = cls.IndexedCoproduct
        FiniteFunctionStrategies.Array = cls.FiniteFunction.Array

        HypergraphStrategies.Hypergraph = cls.Hypergraph

        OpenHypergraphStrategies.OpenHypergraph = cls.OpenHypergraph
        OpenHypergraphStrategies.FiniteFunction = cls.FiniteFunction
