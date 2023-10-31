################################################################################
# Spec
from hypothesis import given
import hypothesis.strategies as st

from open_hypergraphs import *

from tests.strategy.numpy import DEFAULT_DTYPE
from tests.strategy.open_hypergraph import OpenHypergraphStrategies as OpenHyp
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

class Identity(Functor):
    """ The identity functor, implemented using Functor """
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_arrow(self, f: OpenHypergraph) -> OpenHypergraph:
        return f

class FrobeniusIdentity(FrobeniusFunctor):
    """ The identity functor, implemented via FrobeniusFunctor """
    def map_objects(self, objects: FiniteFunction, dtype=DEFAULT_DTYPE) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects, dtype)

    def map_operations(self, x: FiniteFunction, a: FiniteFunction, b: FiniteFunction) -> OpenHypergraph:
        return self.OpenHypergraph().tensor_operations(x, a, b)

class FunctorSpec():

    @given(OpenHyp.arrows())
    def test_frobenius_identity_functor(self, f: OpenHypergraph):
        Id = Identity()
        FrobId = FrobeniusIdentity()

        g1 = Id(f)
        g2 = FrobId(f)
        # Equality is actually on-the-nose here.
        assert g1 == g2

################################################################################
# Actual test class

import unittest
from tests.backend import NumpyBackend

# from tests.backend import NumpyBackend
# import unittest
class NumpyFunctorTests(unittest.TestCase, NumpyBackend, FunctorSpec):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
