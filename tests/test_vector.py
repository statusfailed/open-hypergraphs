""" Test the default vectorised routines in open_hypergraphs/array/backend.py
are correct.  We only test against the numpy backend; other backends should be
compatible. """
import unittest
from hypothesis import given
import hypothesis.strategies as st

import numpy as np
from open_hypergraphs.array.numpy import NumpyBackend as NumpyArrayBackend
from tests.backend import NumpyBackend

from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

# A "run" of length N being e.g. 0 1 2 3 4 ... N
_MAX_RUN_LENGTH = 128
_MAX_RUNS = 128

# A non-vectorised implementation of segmented_arange
def _slow_segmented_arange(x):
    x = np.array(x)

    N = np.sum(x) # how many values to make?
    r = np.zeros(N, dtype=x.dtype) # allocate

    k = 0
    # for each size s,
    for i in range(0, len(x)):
        size = x[i]
        # fill result with a run 0, 1, ..., s
        for j in range(0, size):
            r[k] = j
            k += 1

    return r

@given(
    x=st.lists(st.integers(min_value=0, max_value=_MAX_RUN_LENGTH), min_size=0, max_size=_MAX_RUNS)
)
def test_segmented_arange(x):
    """ Ensure the 'segmented_arange' vector program outputs runs like 0, 1, 2, 0, 1, 2, 3, 4, ... """
    # We're returning an array of size MAX_VALUE * MAX_SIZE, so keep it smallish!
    x = np.array(x, dtype=int)
    N = np.sum(x)
    a = NumpyArrayBackend.segmented_arange(x)

    # Check we got the expected number of elements
    assert len(a) == N
    assert np.all(_slow_segmented_arange(x) == a)


class NumpyArrayBackendTests(unittest.TestCase, NumpyBackend):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
    
    # This is really an IndexedCoproduct
    @given(FinFun.indexed_coproducts())
    def test_segmented_sum(self, x):
        # this will actually error in the IndexedCoproduct constructor, but check anyway!
        assert x.sources.table.sum() == len(x.values)

        # Segmented sum is the same as summing each segment individuall
        actual = NumpyArrayBackend.segmented_sum(x.sources.table, x.values.table)
        expected = np.array([ s.table.sum() for s in x ])
        assert np.all(actual == expected)
