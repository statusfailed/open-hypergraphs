import unittest
from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.strategy.hypergraph import HypergraphStrategies

from tests.spec.hypergraph import HypergraphSpec

from open_hypergraphs import Hypergraph, FiniteFunction, IndexedCoproduct
from tests.test_finite_function import numpy_arrays

class TestHypergraph(unittest.TestCase, HypergraphSpec):
    @classmethod
    def setUpClass(cls):
        FiniteFunctionStrategies.arrays = numpy_arrays
        FiniteFunctionStrategies.Fun = FiniteFunction
        FiniteFunctionStrategies.IndexedCoproduct = IndexedCoproduct
        FiniteFunctionStrategies.Array = FiniteFunction.Array

        HypergraphStrategies.Hypergraph = Hypergraph
