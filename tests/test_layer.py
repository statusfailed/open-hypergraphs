################################################################################
# Spec
import unittest
from hypothesis import given
import hypothesis.strategies as st

from tests.backend import NumpyBackend
from tests.strategy.open_hypergraph import OpenHypergraphStrategies as OpenHyp
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

from open_hypergraphs import *
from open_hypergraphs.numpy.layer import layer, operation_adjacency

class TestLayer(unittest.TestCase, NumpyBackend):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()

    @given(OpenHyp.arrows())
    def test_operation_adjacency_dimension(self, f: OpenHypergraph):
        A = operation_adjacency(f)
        n = len(f.H.x)
        assert A.shape == (n, n)
    
    @given(OpenHyp.arrows())
    def test_layer_dimension(self, f: OpenHypergraph):
        # Check that the output of "layer" is a finite function with as many
        # entries as there are operations.
        l, c = layer(f)
        assert l.source == f.H.x.source
        assert l.target == f.H.x.source
        assert len(c) == len(f.H.x)
