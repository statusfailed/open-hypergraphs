from typing import List

import pytest
from hypothesis import given
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

from open_hypergraphs import Hypergraph

def _assert_hypergraph_equality_invariants(H: Hypergraph, G: Hypergraph):
    """ Not-quite-equality checking for Hypergraphs.
    This only verifies that types are equal, and that there are the same number
    of internal wires, edges whose labels have the same cardinalities.
    Proper equality checking requires graph isomorphism: TODO!
    """

    # same number of wires, wire labels
    assert H.W == G.W
    i = H.w.argsort()
    j = G.w.argsort()
    assert (i >> H.w) == (j >> G.w)

    # same number of edges, edge labels
    assert H.X == G.X
    i = H.x.argsort()
    j = G.x.argsort()
    assert (i >> H.x) == (j >> G.x)

class HypergraphSpec:

    @given(Hyp.objects(n=1))
    def test_sources_and_targets_bounded(self, H: Hypergraph):
        """ Ensure the source and target arrays of a hypergraph all go to the set of hypernodes """
        H = H[0]
        assert H.s.target == H.w.source
        assert H.t.target == H.w.source

    @given(Hyp.labels())
    def test_empty(self, labels):
        w, x = labels
        H = Hypergraph.empty(w.to_initial(), x.to_initial())
        assert H.W == 0
        assert H.X == 0

    @given(Hyp.labels())
    def test_discrete(self, labels):
        w, x = labels

        # A discrete hypergraph can have no hyperedges
        if len(x) > 0:
            with pytest.raises(ValueError):
                H = Hypergraph.discrete(w, x)

        x = x.to_initial()
        H = Hypergraph.discrete(w, x)

        assert len(H.s) == 0
        assert len(H.t) == 0
        assert len(H.x) == 0

    @given(Hyp.discrete())
    def test_discrete_is_discrete(self, H):
        assert H.is_discrete()

    @given(Hyp.objects(n=2))
    def test_hypergraph_coproduct(self, G: List[Hypergraph]):
        H = G[0] + G[1]
        assert len(H.s) == len(G[0].s) + len(G[1].s)
        assert len(H.t) == len(G[0].t) + len(G[1].t)
        assert H.s.values == G[0].s.values @ G[1].s.values
        assert H.t.values == G[0].t.values @ G[1].t.values
        assert H.w == G[0].w + G[1].w
        assert H.x == G[0].x + G[1].x

    @given(Hyp.objects())
    def test_hypergraph_coproduct_list(self, Hs: List[Hypergraph]):
        if len(Hs) == 0:
            return # not allowed
        actual = Hypergraph.coproduct_list(Hs)
        expected = sum(Hs[1:], Hs[0])
        assert actual == expected

    @given(Hyp.discrete_span())
    def test_hypergraph_coequalize_vertices(self, discrete_span):
        l, K, r = discrete_span
        L = l.target
        R = r.target

        # Get the coproduct of L and R
        G = L + R

        # Coequalize parallel maps and identify the "shared" nodes K.
        q = l.w.inject0(R.W).coequalizer(r.w.inject1(L.W))
        H = G.coequalize_vertices(q)

        # H should now have K.W fewer nodes, since we've "removed" one copy of
        # the shared nodes K.
        assert H.W == L.W + R.W - K.W
        assert H.X == L.X + R.X

    @given(Hyp.object_and_permutation())
    def test_hypergraph_permutation(self, Gwx):
        G, w, x = Gwx
        H = G.permute(w, x)
        _assert_hypergraph_equality_invariants(H, G)
