import unittest
from tests.backend import NumpyBackend
from tests.spec.finite_function import FiniteFunctionSpec

class TestFiniteFunction(unittest.TestCase, NumpyBackend, FiniteFunctionSpec):
    @classmethod
    def setUpClass(cls):
        NumpyBackend.setUpClass()
