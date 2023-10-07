import unittest

from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.spec.finite_function import FiniteFunctionSpec

from open_hypergraphs import FiniteFunction, IndexedCoproduct

from tests.strategy.numpy import arrays, permutations

class TestFiniteFunction(unittest.TestCase, FiniteFunctionSpec):
    @classmethod
    def setUpClass(cls):
        # NOTE: we set *global* variables here by overwriting member variables
        # of FiniteFunctionStrategies.
        # This kinda sucks, but it works reliably because hypothesis is
        # guaranteed to run tests single-threaded.
        # "arrays" is a generator for arrays of the backend
        FiniteFunctionStrategies.arrays = arrays
        FiniteFunctionStrategies.permutation_arrays = permutations

        # Fun is the FiniteFunction implementation
        FiniteFunctionStrategies.Fun = FiniteFunction
        FiniteFunctionStrategies.IndexedCoproduct = IndexedCoproduct

        # Array is the array backend
        FiniteFunctionStrategies.Array = FiniteFunction.Array
