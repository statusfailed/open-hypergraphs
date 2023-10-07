import pytest
from hypothesis import given
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

from open_hypergraphs import Hypergraph

class HypergraphSpec:

    @given(Hyp.objects())
    def test_sources_and_targets_bounded(self, H: Hypergraph):
        """ Ensure the source and target arrays of a hypergraph all go to the set of hypernodes """
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
