import hypothesis.strategies as st

from tests.strategy.random import Random
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun
from tests.strategy.hypergraph import HypergraphStrategies as Hyp

class OpenHypergraphStrategies:
    OpenHypergraph = None
    
    @classmethod
    @st.composite
    def arrows(draw, cls, labels=Random):
        [H] = draw(Hyp.objects(n=1, labels=labels))

        num_sources, _ = draw(FinFun.arrow_type(target=H.W))
        num_targets, _ = draw(FinFun.arrow_type(target=H.W))
        s = draw(FinFun.arrows(source=num_sources, target=H.W))
        t = draw(FinFun.arrows(source=num_targets, target=H.W))
        
        return cls.OpenHypergraph(s, t, H)

    @classmethod
    @st.composite
    def many_arrows(draw, cls):
        labels = draw(Hyp.labels())
        x, w = labels
        return x, w, draw(st.lists(cls.arrows(labels=labels), min_size=0, max_size=5))

    @classmethod
    @st.composite
    def identities(draw, cls):
        H = draw(Hyp.discrete())
        s = t = FinFun.FiniteFunction.identity(len(H.w))
        return cls.OpenHypergraph(s, t, H)

    @classmethod
    @st.composite
    def composite_arrows(draw, cls, n=2, labels=Random):
        if n == 0:
            return []

        arrows = [draw(cls.arrows(labels=labels))]

        for i in range(1, n):
            f = arrows[i-1]

            # draw the left leg and apex of the cospan
            # A is the source object of the previous arrow.
            A = Hyp.Hypergraph.discrete(f.target, f.H.x.to_initial())
            α = draw(Hyp.inclusions(A))
            s, H = α.w, α.target

            # draw the right leg of the cospan
            t = draw(FinFun.arrows(target=H.W))

            arrows.append(cls.OpenHypergraph(s, t, H))

        return arrows

    @classmethod
    @st.composite
    def spiders(draw, cls, labels=Random):
        w, x = labels if labels is not Random else draw(Hyp.labels())
        s = draw(FinFun.arrows(target=w.source))
        t = draw(FinFun.arrows(target=w.source))
        return cls.OpenHypergraph.spider(s, t, w, x.to_initial())

    @classmethod
    @st.composite
    def singletons(draw, cls, labels=Random):
        w, x = labels if labels is not Random else draw(Hyp.labels())
        # Bit of a hack; we'll always need at least one label.
        if x.target == 0:
            x.target = 1
        x = draw(FinFun.arrows(source=1, target=x.target))
        a = draw(FinFun.arrows(target=w.source))
        b = draw(FinFun.arrows(target=w.source))
        return cls.OpenHypergraph.singleton(x, a, b)

    @classmethod
    @st.composite
    def isomorphism(draw, cls, labels=Random):
        f = draw(cls.arrows(labels=labels))
        w = draw(FinFun.permutations(target=f.H.W))
        x = draw(FinFun.permutations(target=f.H.X))
        return f, w, x
