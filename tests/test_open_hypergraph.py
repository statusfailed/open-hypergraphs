import unittest
from tests.strategy.numpy import DEFAULT_DTYPE, arrays, permutations

from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.strategy.hypergraph import HypergraphStrategies
from tests.strategy.open_hypergraph import OpenHypergraphStrategies

from tests.spec.open_hypergraph import OpenHypergraphSpec

from open_hypergraphs import OpenHypergraph, Hypergraph, FiniteFunction, IndexedCoproduct

class TestOpenHypergraph(unittest.TestCase, OpenHypergraphSpec):
    @classmethod
    def setUpClass(cls):
        FiniteFunctionStrategies.DEFAULT_DTYPE = DEFAULT_DTYPE
        FiniteFunctionStrategies.arrays = arrays
        FiniteFunctionStrategies.permutation_arrays = permutations

        FiniteFunctionStrategies.Fun = FiniteFunction
        FiniteFunctionStrategies.IndexedCoproduct = IndexedCoproduct
        FiniteFunctionStrategies.Array = FiniteFunction.Array

        HypergraphStrategies.Hypergraph = Hypergraph

        OpenHypergraphStrategies.OpenHypergraph = OpenHypergraph
        OpenHypergraphStrategies.FiniteFunction = FiniteFunction
