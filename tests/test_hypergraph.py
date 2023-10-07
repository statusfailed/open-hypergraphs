import unittest
from tests.strategy.numpy import arrays, permutations

from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.strategy.hypergraph import HypergraphStrategies

from tests.spec.hypergraph import HypergraphSpec

from open_hypergraphs import Hypergraph, FiniteFunction, IndexedCoproduct

class TestHypergraph(unittest.TestCase, HypergraphSpec):
    @classmethod
    def setUpClass(cls):
        FiniteFunctionStrategies.arrays = arrays
        FiniteFunctionStrategies.permutation_arrays = permutations

        FiniteFunctionStrategies.Fun = FiniteFunction
        FiniteFunctionStrategies.IndexedCoproduct = IndexedCoproduct
        FiniteFunctionStrategies.Array = FiniteFunction.Array

        HypergraphStrategies.Hypergraph = Hypergraph
