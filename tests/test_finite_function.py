import unittest

from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.spec.finite_function import FiniteFunctionSpec

from open_hypergraphs import FiniteFunction

class TestFiniteFunction(unittest.TestCase, FiniteFunctionSpec):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        # NOTE: we set a *global* variable here by overwriting member variables
        # of FiniteFunctionStrategies.
        # This kinda sucks, but it works reliably because hypothesis is
        # guaranteed to run tests single-threaded.
        FiniteFunctionStrategies.np = np
        FiniteFunctionStrategies.Fun = FiniteFunction
        FiniteFunctionStrategies.Array = FiniteFunction._Array
