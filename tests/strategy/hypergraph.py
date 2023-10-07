import hypothesis.strategies as st
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

class HypergraphStrategies:
    """ Hypothesis strategies for generating hypergraphs and arrows of hypergraphs """
    Hypergraph = None

    MAX_HYPERNODES = 32

    @classmethod
    @st.composite
    def num_hypernodes(draw, cls, num_hyperedges):
        min_value = 1 if num_hyperedges > 0 else 0
        return draw(st.integers(min_value=min_value, max_value=cls.MAX_HYPERNODES))

    @classmethod
    @st.composite
    def labels(draw, cls):
        x = draw(FinFun.arrows())
        X = x.source
        
        W = draw(cls.num_hypernodes(X))
        w = draw(FinFun.arrows(source=W))

        return x, w


    @classmethod
    @st.composite
    def objects(draw, cls, n=1):
        """ Generate a random (non-acyclic) hypergraph """
        assert n >= 1
        x, w = draw(cls.labels())
        X, W = len(x), len(w)

        result = [None]*n
        for i in range(0, n):
            s = draw(FinFun.indexed_coproducts(n=X, target=W))
            t = draw(FinFun.indexed_coproducts(n=X, target=W))
            result[i] = cls.Hypergraph(s, t, w, x)

        return result
