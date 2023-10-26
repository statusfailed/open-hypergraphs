from typing import List

import pytest
from hypothesis import given
from tests.strategy.open_hypergraph import OpenHypergraphStrategies as OpenHyp

from open_hypergraphs import Hypergraph, OpenHypergraph

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
