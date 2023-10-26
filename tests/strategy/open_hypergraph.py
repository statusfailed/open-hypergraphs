import hypothesis.strategies as st

from tests.strategy.random import Random
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

class OpenHypergraphStrategies:
    OpenHypergraph = None
    
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
        s = t = FinFun.Fun.identity(len(H.w))
        return cls.OpenHypergraph(s, t, H)

    @classmethod
    @st.composite
    def composite_arrows(draw, cls, n=2):
        if n == 0:
            return []

        arrows = [draw(cls.arrows())]

        for i in range(1, n):
            f = arrows[i-1]

            # draw the left leg and apex of the cospan
            # A is the source object of the previous arrow.
            A = Hyp.Hypergraph.discrete(f.target, f.H.x.to_initial(), dtype=f.target.table.dtype)
            α = draw(Hyp.inclusions(A))
            s, H = α.w, α.target

            # draw the right leg of the cospan
            t = draw(FinFun.arrows(target=H.W))

            arrows.append(cls.OpenHypergraph(s, t, H))

        return arrows
