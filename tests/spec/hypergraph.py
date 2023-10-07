from typing import List

import pytest
from hypothesis import given
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

from open_hypergraphs import Hypergraph

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

    @given(Hyp.objects(n=2))
    def test_hypergraph_coproduct(self, G: List[Hypergraph]):
        H = G[0] + G[1]
        assert len(H.s) == len(G[0].s) + len(G[1].s)
        assert len(H.t) == len(G[0].t) + len(G[1].t)
        assert H.s.values == G[0].s.values @ G[1].s.values
        assert H.t.values == G[0].t.values @ G[1].t.values
        assert H.w == G[0].w + G[1].w
        assert H.x == G[0].x + G[1].x
