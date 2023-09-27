import unittest

import numpy as np
import hypothesis.extra.numpy as numpygen
from hypothesis import strategies as st

from tests.strategy.finite_function import FiniteFunctionStrategies
from tests.spec.finite_function import FiniteFunctionSpec

from open_hypergraphs import FiniteFunction

# NOTE: Hypothesis seems to have a bug where zero-length arrays trigger an
# error, so we'll just use numpy's random module instead.
@st.composite
def numpy_arrays(draw, n, high, dtype=np.uint32):
    # dummy call to draw to silence hypothesis warning
    _ = draw(st.integers(min_value=0, max_value=0))
    return np.random.randint(0, high=high, size=(n,), dtype=dtype)

class TestFiniteFunction(unittest.TestCase, FiniteFunctionSpec):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        # NOTE: we set *global* variables here by overwriting member variables
        # of FiniteFunctionStrategies.
        # This kinda sucks, but it works reliably because hypothesis is
        # guaranteed to run tests single-threaded.
        # "arrays" is a generator for arrays of the backend
        FiniteFunctionStrategies.arrays = numpy_arrays
        # Fun is the FiniteFunction implementation
        FiniteFunctionStrategies.Fun = FiniteFunction
        # Array is the array backend
        FiniteFunctionStrategies.Array = FiniteFunction._Array
