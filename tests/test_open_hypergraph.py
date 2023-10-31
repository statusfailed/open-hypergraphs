import unittest
from tests.backend import NumpyBackend
from tests.spec.open_hypergraph import OpenHypergraphSpec

class TestOpenHypergraph(unittest.TestCase, NumpyBackend, OpenHypergraphSpec):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
