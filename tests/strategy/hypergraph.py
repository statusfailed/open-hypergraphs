import hypothesis.strategies as st

from tests.strategy.random import Random
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
    def labels(draw, cls, sigma_0=Random, sigma_1=Random):
        x = draw(FinFun.arrows(target=sigma_1))
        X = x.source

        W = draw(cls.num_hypernodes(X))
        w = draw(FinFun.arrows(source=W, target=sigma_0))

        return w, x


    @classmethod
    @st.composite
    def objects(draw, cls, n=1, labels=Random):
        """ Generate a random (possibly cyclic) hypergraph """
        assert n >= 1
        if labels is Random:
            w, x = draw(cls.labels())
        else:
            w, x = labels
        W, X = len(w), len(x)

        result = [None]*n
        for i in range(0, n):
            s = draw(FinFun.indexed_coproducts(n=X, target=W))
            t = draw(FinFun.indexed_coproducts(n=X, target=W))
            result[i] = cls.Hypergraph(s, t, w, x)

        return result

    # Draw a discrete hypergraph
    @classmethod
    @st.composite
    def discrete(draw, cls):
        w, x = draw(cls.labels())
        # Make a discrete hypergraph with labels w.
        [K] = draw(cls.objects(labels=(w, x.to_initial())))
        return K

    # Draw a span of hypergraphs
    #     l   r
    #   L ← K → R
    # whose apex K is discrete
    @classmethod
    @st.composite
    def discrete_span(draw, cls):
        w, x = draw(cls.labels())

        # Make a discrete hypergraph with labels w.
        [K] = draw(cls.objects(labels=(w, x.to_initial())))

        # L is K with a bunch of additional stuff included
        w_L_prime = draw(FinFun.arrows(target=w.target))
        w_L = w + w_L_prime # TODO!!! PICK UP FROM HERE
        x_L = x + draw(FinFun.arrows(target=x.target))
        [L] = draw(cls.objects(labels=(w_L, x_L)))

        # R is K with a bunch of additional stuff included
        w_R_prime = draw(FinFun.arrows(target=w.target))
        w_R = w + w_R_prime
        x_R = x + draw(FinFun.arrows(target=x.target))
        [R] = draw(cls.objects(labels=(w_R, x_R)))

        l = FinFun.Fun.inj0(len(w), len(w_L) - len(w))
        r = FinFun.Fun.inj0(len(w), len(w_R) - len(w))

        # Smoketest:
        # 1: Check that w/x maps have compatible targets and dtypes
        assert L.w.target == R.w.target
        assert L.w.table.dtype == R.w.table.dtype
        assert L.x.target == R.x.target
        assert L.x.table.dtype == R.x.table.dtype

        # 2: Check dtypes of arrays in sources/targets are equal.
        assert L.s.sources.table.dtype == R.s.sources.table.dtype
        assert L.s.values.table.dtype == R.s.values.table.dtype

        assert L.t.sources.table.dtype == R.t.sources.table.dtype
        assert L.t.values.table.dtype == R.t.values.table.dtype


        return L, l, K, r, R
