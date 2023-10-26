import hypothesis.strategies as st

from tests.strategy.random import Random
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

class OpenHypergraphStrategies:
    OpenHypergraph = None
    FiniteFunction = None
    
    @classmethod
    @st.composite
    def arrows(draw, cls):
        [H] = draw(Hyp.objects(n=1))

        num_sources, _ = draw(FinFun.arrow_type(target=H.W))
        num_targets, _ = draw(FinFun.arrow_type(target=H.W))
        s = draw(FinFun.arrows(source=num_sources, target=H.W))
        t = draw(FinFun.arrows(source=num_targets, target=H.W))
        
        return cls.OpenHypergraph(s, t, H)

    @classmethod
    @st.composite
    def identities(draw, cls):
        H = draw(Hyp.discrete())
        s = t = cls.FiniteFunction.identity(len(H.w))
        return cls.OpenHypergraph(s, t, H)
