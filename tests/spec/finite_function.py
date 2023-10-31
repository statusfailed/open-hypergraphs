from hypothesis import given
import hypothesis.strategies as st

from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

from open_hypergraphs.finite_function import FiniteFunction, IndexedCoproduct
from open_hypergraphs.open_hypergraph import OpenHypergraph

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
        FinFun.FiniteFunction.identity(f.source) >> f == f

    @given(FinFun.arrows())
    def test_category_identity_right(self, f):
        """ ``f ; id = f`` """
        f >> FinFun.FiniteFunction.identity(f.target) == f

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
        assert f == FinFun.FiniteFunction.initial(f.target)

    @given(FinFun.arrows())
    def test_to_initial(self, f):
        assert f.to_initial() == FinFun.FiniteFunction.initial(f.target)

    @given(FinFun.indexed_coproducts(n=2))
    def test_coproduct_diagram_commutes(self, c: IndexedCoproduct):
        f, g = c # note: this uses the IndexedCoproduct's __iter__ to unpack
        i0 = FinFun.FiniteFunction.inj0(f.source, g.source)
        i1 = FinFun.FiniteFunction.inj1(f.source, g.source)

        assert (i0 >> (f + g)) == f
        assert (i1 >> (f + g)) == g

    @given(f=FinFun.arrows(), b=FinFun.objects())
    def test_f_cp_inj0_equals_inject0(self, f, b):
        assert f >> FinFun.FiniteFunction.inj0(f.target, b) == f.inject0(b)

    @given(f=FinFun.arrows(), a=FinFun.objects())
    def test_f_cp_inj1_equals_inject1(self, f, a):
        assert f >> FinFun.FiniteFunction.inj1(a, f.target) == f.inject1(a)

    ############################################################################
    # Strict symmetric monoidal properties

    @given(f=FinFun.arrows(), g=FinFun.arrows())
    def test_tensor_vs_injections(self, f, g):
        """ Verify that the tensor product of arrows corresponds to its
        definition in terms of coproducts and injections """
        i0 = FinFun.FiniteFunction.inj0(f.target, g.target)
        i1 = FinFun.FiniteFunction.inj1(f.target, g.target)

        assert f @ g == (f >> i0) + (g >> i1)

    @given(a=FinFun.objects(), b=FinFun.objects())
    def test_twist_inverse(self, a, b):
        """ Verify that σ ; σ = id """
        f = FinFun.FiniteFunction.twist(a, b)
        g = FinFun.FiniteFunction.twist(b, a)

        identity = FinFun.FiniteFunction.identity(a+b)
        assert f >> g == identity
        assert g >> f == identity

    @given(f=FinFun.arrows(), g=FinFun.arrows())
    def test_twist_naturality(self, f, g):
        """ Check naturality of σ, so that (f @ g) ; σ = σ ; (g @ f) """
        post_twist = FinFun.FiniteFunction.twist(f.target, g.target)
        pre_twist = FinFun.FiniteFunction.twist(f.source, g.source)

        assert ((f @ g) >> post_twist) == (pre_twist >> (g @ f))

    ############################################################################
    # Coequalizers

    @given(fg=FinFun.parallel_arrows(n=2))
    def test_coequalizer_commutes(self, fg):
        f, g = fg
        c = f.coequalizer(g)
        assert (f >> c) == (g >> c)

    @given(fg=FinFun.parallel_arrows(n=2))
    def test_coequalizer_universal_identity(self, fg):
        f, g = fg
        q = f.coequalizer(g)
        u = q.coequalizer_universal(q)
        assert u == type(u).identity(q.target)


    # A custom strategy to generate a coequalizer
    #   q : A → Q
    # and a compatible permutation
    #   p : Q → Q
    @st.composite
    @staticmethod
    def coequalizer_and_permutation(draw):
        f, g = draw(FinFun.parallel_arrows(n=2))
        q = f.coequalizer(g)
        p = draw(FinFun.permutations(target=q.target))
        return f, g, q, p

    @given(coequalizer_and_permutation())
    def test_coequalizer_and_permutation(self, fgqp):
        """ Coequalizers are unique only up to permutation. This checks that a
        coequalizer postcomposed with a permutation commutes with the universal
        morphism """
        f, g, q0, p = fgqp

        assert f.target == g.target
        assert f.target == q0.source
        assert p.source == q0.target

        q1 = q0 >> p
        u  = q0.coequalizer_universal(q1)
        assert q0 >> u == q1

    ############################################################################
    # Finite (indexed) coproducts

    @given(FinFun.arrows())
    def test_indexed_coproduct_singleton(self, f):
        s = FinFun.IndexedCoproduct.singleton(f)
        assert len(s) == 1
        assert s.values == f

    @given(FinFun.arrows())
    def test_indexed_coproduct_elements(self, f: FiniteFunction):
        s = FinFun.IndexedCoproduct.elements(f)
        assert len(s) == len(f)
        assert s.values == f

    @given(s=FinFun.arrows())
    def test_injection_coproduct_identity(self, s: FiniteFunction):
        """ Test that the (finite) coproduct of injections is the identity
            ι_0 + ι_1 + ... + ι_N = identity(sum_{i ∈ N} s(i))
        """
        i = FinFun.FiniteFunction.identity(s.source, dtype=s.table.dtype)
        n = FinFun.FiniteFunction.Array.sum(s.table)
        assert s.injections(i) == FinFun.FiniteFunction.identity(n)

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

        expected_sources = FinFun.FiniteFunction(None, [len(fs[x(i)]) for i in range(0, x.source)] )
        expected_values  = FinFun.FiniteFunction.coproduct_list([ fs[x(i)] for i in range(0, x.source) ], target=target)

        assert len(c) == len(fs)

        d = c.map_indexes(x)
        assert len(d) == len(x)
        assert d.sources == expected_sources
        assert d.values == expected_values
