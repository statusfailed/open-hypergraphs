from hypothesis import given

from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

class FiniteFunctionSpec:

    @given(FinFun.arrows())
    def test_equality_reflexive(self, f):
        assert f == f

    # If two functions have unequal tables or dtypes, they are unequal.
    @given(FinFun.parallel_arrows(n=2))
    def test_inequality(self, fg):
        f, g = fg

        if FinFun.np.any(f.table != g.table):
            assert f != g
        elif (f.table.dtype != g.table.dtype):
            assert f != g
        else:
            assert f == g

    # Two finite functions with unequal types are always unequal
    @given(FinFun.arrows(), FinFun.arrows())
    def test_unequal_types_unequal(self, f, g):
        if f.type != g.type:
            assert f != g

    ############################################################################
    # Category Laws

    @given(FinFun.arrows())
    def test_category_identity_left(self, f):
        """ ``id ; f = f`` """
        FinFun.Fun.identity(f.source) >> f == f

    @given(FinFun.arrows())
    def test_category_identity_right(self, f):
        """ ``f ; id = f`` """
        f >> FinFun.Fun.identity(f.target) == f

    # from hypothesis import settings, reproduce_failure
    # @settings(print_blob=True)
    @given(FinFun.composite_arrows(n=3))
    def test_category_composition_associative(self, fgh):
        """ ``(f ; g) ; h = f ; (g ; h)`` """
        f, g, h = fgh
        assert (f >> g) >> h == f >> (g >> h)

    ############################################################################
    # Coproducts
    # TODO!
