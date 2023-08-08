from hypothesis import given

from tests.strategy.finite_function import FiniteFunctionStrategies as FFS

class FiniteFunctionSpec:

    @given(FFS.finite_functions())
    def test_equality_reflexive(self, f):
        assert f == f
