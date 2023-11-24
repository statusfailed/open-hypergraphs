import hypothesis.strategies as st

from tests.strategy.random import Random
from tests.strategy.finite_function import FiniteFunctionStrategies as FinFun

from open_hypergraphs.open_hypergraph import HypergraphArrow

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

    @classmethod
    @st.composite
    def object_and_permutation(draw, cls, labels=Random):
        [H] = draw(cls.objects(n=1, labels=labels))
        w = draw(FinFun.permutations(target=H.W))
        x = draw(FinFun.permutations(target=H.X))
        return H, w, x

    # Draw a discrete hypergraph
    @classmethod
    @st.composite
    def discrete(draw, cls):
        w, x = draw(cls.labels())
        # Make a discrete hypergraph with labels w.
        [K] = draw(cls.objects(labels=(w, x.to_initial())))
        return K

    @classmethod
    @st.composite
    def inclusions(draw, cls, G: Hypergraph = None):
        """ Given a hypergraph G, generate an inclusion ``i : G → H`` of ``G`` in ``H``
        i₀ : G → G + H
        """
        if G is None:
            [G] = draw(cls.objects(n=1))

        # old labels + new labels
        w = G.w + draw(FinFun.arrows(target=G.w.target, dtype=st.just(G.w.dtype)))
        x = G.x + draw(FinFun.arrows(target=G.x.target, dtype=st.just(G.x.dtype)))
        [H] = draw(cls.objects(labels=(w, x)))

        # A morphism of hypergraphs is a mapping on vertices and edges, respectively!
        return HypergraphArrow(
            source=G,
            target=H,
            w=FinFun.FiniteFunction.inj0(G.W, H.W - G.W),
            x=FinFun.FiniteFunction.inj0(G.X, H.X - G.X))

    # Draw a span of hypergraphs
    #     l   r
    #   L ← K → R
    # whose apex K is discrete
    @classmethod
    @st.composite
    def discrete_span(draw, cls):
        K = draw(cls.discrete())
        l = draw(cls.inclusions(K))
        r = draw(cls.inclusions(K))

        L = l.target
        R = r.target

        # Smoketest
        # 0: Check that targets of l and r have the correct number of wires, operations
        assert l.w.target == L.W
        assert r.w.target == R.W
        assert l.x.target == L.X
        assert r.x.target == R.X

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

        return l, K, r
