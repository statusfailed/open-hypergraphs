################################################################################
# Spec
from hypothesis import given
import hypothesis.strategies as st

from open_hypergraphs import *

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
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, a: IndexedCoproduct, b: IndexedCoproduct) -> OpenHypergraph:
        return self.OpenHypergraph().tensor_operations(x, a, b)

class FrobeniusDagger(FrobeniusFunctor):
    """ The dagger functor, implemented via FrobeniusFunctor """
    def map_objects(self, objects: FiniteFunction) -> IndexedCoproduct:
        # identity-on-objects
        return self.IndexedCoproduct().elements(objects)

    def map_operations(self, x: FiniteFunction, a: IndexedCoproduct, b: IndexedCoproduct) -> OpenHypergraph:
        # swap source/target of each operation
        return self.OpenHypergraph().tensor_operations(x, b, a)

class DaggerOptic(Optic):
    F: FrobeniusFunctor = FrobeniusIdentity()
    R: FrobeniusFunctor = FrobeniusDagger()

    def residual(self, x: FiniteFunction, A: IndexedCoproduct, B: IndexedCoproduct) -> IndexedCoproduct:
        # NOTE: we use map_values to get an IndexedCoproduct whose values are
        # objects, not ints.
        sources = FiniteFunction.constant(0, x.source, None)
        values = FiniteFunction.initial(A.target, dtype=A.values.dtype)
        return IndexedCoproduct(sources, values)

class OpticSpec():
    # A basic test case for the identity on objects A, B
    def test_dagger_optic_identity2(self):
        O = DaggerOptic()

        X = FiniteFunction(None, FiniteFunction.Array.array(["A", "B"], 'O'))
        id2 = OpenHypergraph.identity(X, X.to_initial())

        OX = O.map_objects(X)
        # Check that for each generating object in X, there is a sublist in OX.
        assert len(OX) == len(X)
        assert FiniteFunction.transpose(len(X), 2) >> OX.values == X + X

        f = O(id2)
        assert f.source == OX.values
        assert f.target == OX.values

    # Test case for singleton op  f : A ● B → C, which we assume has a dagger
    # f† : C → B ● A
    def test_dagger_optic_singleton(self):
        F = FrobeniusIdentity()
        R = FrobeniusDagger()
        O = DaggerOptic(F, R)

        X = FiniteFunction(None, FiniteFunction.Array.array(["A", "B"], dtype='O'))
        Y = FiniteFunction(None, FiniteFunction.Array.array(["Y"], dtype='O'))
        x = FiniteFunction.singleton(0, 1)
        f = OpenHypergraph.singleton(x, X, Y)

        # Check that for each generating object in X, there is a sublist in OX.
        OX = O.map_objects(X)
        assert len(OX) == len(X)
        assert FiniteFunction.transpose(len(X), 2) >> OX.values == X + X

        OY = O.map_objects(Y)
        assert len(OY) == len(Y)
        assert FiniteFunction.transpose(len(Y), 2) >> OY.values == Y + Y

        Of = O(f)
        assert Of.source == OX.values
        assert Of.target == OY.values

    @given(OpenHyp.arrows())
    def test_dagger_optic_type(self, f):
        F = FrobeniusIdentity()
        R = FrobeniusDagger()
        O = DaggerOptic(F, R)

        Of = O(f)
        FA = F.map_objects(f.source)
        FB = F.map_objects(f.target)
        RA = R.map_objects(f.source)
        RB = R.map_objects(f.target)

        assert len(FA) == len(RA)
        assert len(FB) == len(RB)

        pA = (FA + RA).sources.injections(self.FiniteFunction.transpose(len(FA), 2))
        pB = (FB + RB).sources.injections(self.FiniteFunction.transpose(len(FB), 2))

        assert pA >> Of.source == FA.values + RA.values
        assert pB >> Of.target == FB.values + RB.values

################################################################################
# Actual test class

import unittest
from tests.backend import NumpyBackend

class NumpyOpticTests(unittest.TestCase, NumpyBackend, OpticSpec):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
