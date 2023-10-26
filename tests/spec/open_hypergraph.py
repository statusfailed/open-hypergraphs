from typing import List

import pytest
from hypothesis import given
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
        assert h.H.W <= f.H.W + g.H.W
        assert h.H.W >= f.H.W + g.H.W - len(f.t)

    @given(OpenHyp.composite_arrows(n=3))
    def test_composition_associative(self, fgh):
        f, g, h = fgh
        _assert_equality_invariants((f >> g) >> h, f >> (g >> h))
