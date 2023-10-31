import unittest
from tests.backend import NumpyBackend
from tests.spec.hypergraph import HypergraphSpec

class TestHypergraph(unittest.TestCase, NumpyBackend, HypergraphSpec):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
