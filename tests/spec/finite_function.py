from hypothesis import given

from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

from open_hypergraphs.finite_function import AbstractFiniteFunction, AbstractIndexedCoproduct

class FiniteFunctionSpec:

    @given(FinFun.arrows())
    def test_equality_reflexive(self, f):
        assert f == f

    # If two functions have unequal tables or dtypes, they are unequal.
    @given(FinFun.parallel_arrows(n=2))
    def test_inequality(self, fg):
        f, g = fg

        if FinFun.Array.any(f.table != g.table):
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

    @given(FinFun.composite_arrows(n=3))
    def test_category_composition_associative(self, fgh):
        """ ``(f ; g) ; h = f ; (g ; h)`` """
        f, g, h = fgh
        assert (f >> g) >> h == f >> (g >> h)

    ############################################################################
    # Coproducts

    # Uniqueness of the initial map
    # any map f : 0 → B is equal to the initial map ? : 0 → B
    @given(FinFun.arrows(source=0))
    def test_initial_map_unique(self, f):
        assert f == FinFun.Fun.initial(f.target)

    @given(FinFun.arrows())
    def test_to_initial(self, f):
        assert f.to_initial() == FinFun.Fun.initial(f.target)

    @given(FinFun.indexed_coproducts(n=2))
    def test_coproduct_diagram_commutes(self, c: AbstractIndexedCoproduct):
        f, g = c # note: this uses the IndexedCoproduct's __iter__ to unpack
        i0 = FinFun.Fun.inj0(f.source, g.source)
        i1 = FinFun.Fun.inj1(f.source, g.source)

        assert (i0 >> (f + g)) == f
        assert (i1 >> (f + g)) == g

    @given(f=FinFun.arrows(), b=FinFun.objects())
    def test_f_cp_inj0_equals_inject0(self, f, b):
        assert f >> FinFun.Fun.inj0(f.target, b) == f.inject0(b)

    @given(f=FinFun.arrows(), a=FinFun.objects())
    def test_f_cp_inj1_equals_inject1(self, f, a):
        assert f >> FinFun.Fun.inj1(a, f.target) == f.inject1(a)

    ############################################################################
    # Strict symmetric monoidal properties

    @given(f=FinFun.arrows(), g=FinFun.arrows())
    def test_tensor_vs_injections(self, f, g):
        """ Verify that the tensor product of arrows corresponds to its
        definition in terms of coproducts and injections """
        i0 = FinFun.Fun.inj0(f.target, g.target)
        i1 = FinFun.Fun.inj1(f.target, g.target)

        assert f @ g == (f >> i0) + (g >> i1)

    @given(a=FinFun.objects(), b=FinFun.objects())
    def test_twist_inverse(self, a, b):
        """ Verify that σ ; σ = id """
        f = FinFun.Fun.twist(a, b)
        g = FinFun.Fun.twist(b, a)

        identity = FinFun.Fun.identity(a+b)
        assert f >> g == identity
        assert g >> f == identity

    @given(f=FinFun.arrows(), g=FinFun.arrows())
    def test_twist_naturality(self, f, g):
        """ Check naturality of σ, so that (f @ g) ; σ = σ ; (f @ g) """
        post_twist = FinFun.Fun.twist(f.target, g.target)
        pre_twist = FinFun.Fun.twist(f.source, g.source)

        assert ((f @ g) >> post_twist) == (pre_twist >> (g @ f))

    ############################################################################
    # Coequalizers

    @given(fg=FinFun.parallel_arrows(n=2))
    def test_coequalizer_commutes(self, fg):
        f, g = fg
        c = f.coequalizer(g)
        assert (f >> c) == (g >> c)

    ############################################################################
    # Finite (indexed) coproducts

    @given(s=FinFun.arrows())
    def test_injection_coproduct_identity(self, s: AbstractFiniteFunction):
        """ Test that the (finite) coproduct of injections is the identity
            ι_0 + ι_1 + ... + ι_N = identity(sum_{i ∈ N} s(i))
        """
        i = FinFun.Fun.identity(s.source)
        n = FinFun.Fun.Array.sum(s.table)
        assert s.injections(i) == FinFun.Fun.identity(n)

    @given(FinFun.indexed_coproducts())
    def test_indexed_coproduct_roundtrip(self, c):
        d = type(c).from_list(c.target, list(c))
        # Test that IndexedCoproduct roundtrips via a list of functions
        assert c == d
        # Also test that a list of functions roundtrips through IndexedCoproduct
        assert list(d) == list(c)

    @given(FinFun.map_with_indexed_coproducts())
    def test_indexed_coproduct_map(self, fsx):
        # Test that we can pre-compose an arrow with the indexing function
        c, x = fsx
        target = c.values.target
        fs = list(c)

        expected_sources = FinFun.Fun(None, [len(fs[x(i)]) for i in range(0, x.source)] )
        expected_values   = FinFun.Fun.coproduct_list([ fs[x(i)] for i in range(0, x.source) ], target=target)

        assert len(c) == len(fs)

        d = c.map(x)
        assert len(d) == len(x)
        assert d.sources == expected_sources
        assert d.values == expected_values
