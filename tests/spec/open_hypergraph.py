from typing import List

import pytest
from hypothesis import given
import hypothesis.strategies as st

from tests.strategy.hypergraph import HypergraphStrategies as Hyp
from tests.strategy.open_hypergraph import OpenHypergraphStrategies as OpenHyp

from open_hypergraphs import Hypergraph, OpenHypergraph

def _assert_equality_invariants(f: OpenHypergraph, g: OpenHypergraph):
    """ Not-quite equality checking for OpenHypergraph.
    This only verifies that types are equal, and that there are the same number
    of internal wires, edges whose labels have the same cardinalities.
    Proper equality checking requires graph isomorphism: TODO!
    """
    # same type
    assert f.source == g.source
    assert f.target == g.target

    # same number of wires, wire labels
    assert f.H.W == g.H.W
    i = f.H.w.argsort()
    j = g.H.w.argsort()
    assert (i >> f.H.w) == (j >> g.H.w)

    # same number of edges, edge labels
    assert f.H.X == g.H.X
    i = f.H.x.argsort()
    j = g.H.x.argsort()
    assert (i >> f.H.x) == (j >> g.H.x)

arrow_pair = Hyp.labels().flatmap(lambda labels: st.tuples(OpenHyp.arrows(labels=labels), OpenHyp.arrows(labels=labels)))

class OpenHypergraphSpec:

    ########################################
    # Category laws
    ########################################

    @given(OpenHyp.identities())
    def test_identity_type(self, f):
        # Check the morphism is discrete
        assert len(f.H.s) == 0
        assert len(f.H.t) == 0
        assert len(f.H.x) == 0

        # Check identity map is of type id : A → A
        assert f.source == f.target

        # Check the apex of the cospan is discrete with labels A.
        assert f.H.w == f.source

    @given(OpenHyp.arrows())
    def test_identity_law(self, f):
        """ Check that ``f ; id = f`` and ``id ; f == f``. """
        dtype = f.source.table.dtype
        id_A = OpenHyp.OpenHypergraph.identity(f.source, f.H.x.to_initial(), dtype)
        id_B = OpenHyp.OpenHypergraph.identity(f.target, f.H.x.to_initial(), dtype)

        _assert_equality_invariants(f, id_A >> f)
        _assert_equality_invariants(f, f >> id_B)

    @given(OpenHyp.composite_arrows(n=2))
    def test_composition_wire_count(self, fg):
        f, g = fg
        h = f >> g

        # check types
        assert h.source == f.source
        assert h.target == g.target

        assert h.H.W <= f.H.W + g.H.W
        assert h.H.W >= f.H.W + g.H.W - len(f.t)

    @given(OpenHyp.composite_arrows(n=3))
    def test_composition_associative(self, fgh):
        f, g, h = fgh
        _assert_equality_invariants((f >> g) >> h, f >> (g >> h))

    @given(Hyp.labels().flatmap(lambda labels:
        st.tuples(OpenHyp.composite_arrows(n=2, labels=labels),
                  OpenHyp.composite_arrows(n=2, labels=labels))))
    def test_monoidal_interchange(self, fg):
        f, g = fg
        a = (f[0] >> f[1]) @ (g[0] >> g[1])
        b = (f[0] @ g[0]) >> (f[1] @ g[1])
        _assert_equality_invariants(a, b)

    @given(OpenHyp.arrows())
    def test_tensor_unit(self, f):
        u = f.unit_of()
        assert f == f @ u
        assert f == u @ f

    @given(arrow_pair)
    def test_tensor_type(self, fg):
        f, g = fg
        assert (f @ g).source == (f.source + g.source)
        assert (f @ g).target == (f.target + g.target)

    @given(arrow_pair)
    def test_twist_naturality(self, fg):
        f, g = fg
        dtype = f.s.table.dtype
        x = f.H.x.to_initial()

        #     s_Y           s_X
        #
        # --f--\ /--     --\ /--g--
        #       x     =     x
        # --g--/ \--     --/ \--f--

        s_Y = type(f).twist(f.target, g.target, x, dtype)
        a = (f @ g) >> s_Y

        s_X = type(f).twist(f.source, g.source, x, dtype)
        b = s_X >> (g @ f)

        _assert_equality_invariants(a, b)

    @given(OpenHyp.arrows())
    def test_dagger(self, f):
        """ Test the dagger swaps source and target """
        g = f.dagger()
        assert f.source == g.target
        assert f.target == g.source
        assert f.H == g.H
        assert f.s == g.t
        assert f.t == g.s

    @given(OpenHyp.spiders())
    def test_spider_discrete(self, s):
        assert s.H.is_discrete()

    @given(OpenHyp.spiders())
    def test_spider_composed_dagger_node_count(self, s):
        """ Test that a composition of spiders ``s ; s†`` has the same number of hypernodes as s. """
        f = s >> s.dagger()
        assert f.source == s.source
        assert f.target == s.source
        assert f.H.W <= s.H.W * 2
        assert f.H.W >= min(1, s.H.W)

    @given(OpenHyp.singletons())
    def test_singleton(self, f):
        assert f.H.X == 1

        # check the only sources/targets are those in image of cospan legs
        assert f.H.s.values == f.s
        assert f.H.t.values == f.t

        # check number of wires equal to arity + coarity
        assert f.H.W == len(f.source + f.target)
