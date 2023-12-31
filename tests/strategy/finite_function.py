import hypothesis.strategies as st

from tests.strategy.random import Random

def _is_valid_arrow_type(A, B):
    if B == 0:
        return A == 0
    return True

class FiniteFunctionStrategies:
    """ Hypothesis strategies for generating finite functions.
    Parametrised by the array module (np), Array backend, and Finite Function class.
    """

    # Generators for numpy-compatible arrays
    arrays = None
    permutation_arrays = None

    # open-hypergraphs Array backend and FiniteFunction implementation(?)
    Array = None
    Fun = None
    IndexedCoproduct = None

    # Largest object to test with
    MAX_OBJECT = 32

    # Largest indexed of a coproduct (i.e., number of elements n in a
    # coproduct X_1 + X_2 ... X_n)
    MAX_COPRODUCT = 32

    @classmethod
    @st.composite
    def objects(draw, cls, allow_initial=True):
        min_value = 0 if allow_initial else 1
        return draw(st.integers(min_value=min_value, max_value=cls.MAX_OBJECT))

    @classmethod
    @st.composite
    def arrow_type(draw, cls, source=Random, target=Random, finite_target=False):
        """ Generate the type of a random finite function.
        Only one type is forbidden: ``N → 0`` for ``N > 0``.
        """
        # User specified both target and source
        if target is not Random and source is not Random:
            if target == 0 and source != 0:
                raise ValueError("No arrows exist of type n → 0 for n != 0.")
            return source, target

        elif source is Random:
            # any target
            target = draw(cls.objects()) if target is Random else target

            if target == 0:
                source = 0
            else:
                source = draw(cls.objects())

            return source, target

        # target can only be initial if source is also initial.
        target = draw(cls.objects(allow_initial=(source==0)))
        if finite_target == False and draw(st.booleans()):
            target = None

        assert _is_valid_arrow_type(source, target)
        return source, target


    @classmethod
    @st.composite
    def arrows(draw, cls, source=Random, target=Random, finite_target=False, dtype=None):
        if dtype is None:
            dtype = st.sampled_from([cls.FiniteFunction.Dtype, 'O'])

        if source is None:
            raise ValueError("Cannot generate an arrow with source = None")

        source, target = draw(cls.arrow_type(
            source=source,
            target=target,
            finite_target=finite_target))

        if target == 0:
            table = cls.Array.zeros(0, dtype=cls.FiniteFunction.Dtype)
        else:
            # NOTE: set high to MAX_OBJECT if non-finite.
            high = cls.MAX_OBJECT if target is None else target
            table = draw(cls.arrays(n=source, high=high, dtype=cls.FiniteFunction.Dtype))

        if target is None and finite_target == False:
            dtype = draw(dtype)
            table = cls.Array.array([x for x in table], dtype=dtype)

        return cls.FiniteFunction(target, table)

    @classmethod
    @st.composite
    def permutations(draw, cls, target=Random):
        if target is Random:
            target = draw(cls.objects())

        table = draw(cls.permutation_arrays(n=target, dtype=cls.FiniteFunction.Dtype))
        return cls.FiniteFunction(target, table)

    @classmethod
    @st.composite
    def parallel_arrows(draw, cls, source=Random, target=Random, n=2, finite_target=False):
        source, target = draw(cls.arrow_type(source=source, target=target))
        return [
            draw(cls.arrows(source=source, target=target, finite_target=finite_target))
            for _ in range(0, n) ]

    @classmethod
    @st.composite
    def composite_arrows(draw, cls, source=Random, target=Random, n=2, finite_target=False):
        """
        ``composite(A, B, n=k)`` draws a list of ``k`` arrows which can be composed in sequence.
        For example, when k = 3, we might have the following arrows:

        ```
            f₀ : A → X₀
            f₁ : X₀ → X₁
            f₂ : X₁ → B
        ```
        """

        if n == 0:
            return []
        arrows = []

        # if target is 0, all arrows have to have type 0 → 0
        if target == 0:
            return [ draw(cls.arrows(0, 0)) for _ in range(0, n) ]

        T = target if n == 1 else Random
        arrows.append(draw(cls.arrows(source, T)))

        for i in range(0, n-1):
            S = arrows[i].target
            T = Random if i < n - 2 else target
            arrows.append(draw(cls.arrows(
                source=S,
                target=T,
                finite_target=finite_target if i == n - 2 else True)))

        return arrows

    ######################################################################
    # Coproducts

    @classmethod
    @st.composite
    def coproduct_indexes(draw, cls):
        min_value = 0
        return draw(st.integers(min_value=min_value, max_value=cls.MAX_COPRODUCT))

    @classmethod
    @st.composite
    def indexed_coproducts(draw, cls, n=Random, target=Random, finite_target=False):
        if n is Random:
            n = draw(cls.coproduct_indexes())

        # IndexedCoproduct(s, v)
        #   s : N → Inf
        #   v : sum(s) → X
        # when X == 0, we must have sum(s) == 0, so s is an array of zeros.
        max_source = Random if target is Random or target > 0 else 1
        sources = draw(cls.arrows(source=n, target=max_source, finite_target=True))
        sources.target = None # NOTE: need sources to be non-finite codomain.

        # sum of sources is the source of the values array
        source = cls.FiniteFunction.Array.sum(sources.table)
        _, target = draw(cls.arrow_type(source=source, target=target, finite_target=finite_target))
        values = draw(cls.arrows(
            source=cls.FiniteFunction.Array.sum(sources.table),
            target=target,
            finite_target=finite_target))

        return cls.IndexedCoproduct(sources=sources, values=values)

    @classmethod
    @st.composite
    def indexed_coproduct_nonempty_lists(draw, cls, finite_target=False):
        return draw(st.lists(cls.indexed_coproducts(finite_target=finite_target), max_size=5, min_size=1))

    @classmethod
    @st.composite
    def composable_indexed_coproducts(draw, cls, n=Random, target=Random):
        x = draw(cls.indexed_coproducts(n=n, target=target))
        y = draw(cls.indexed_coproducts(n=len(x.values)))
        return x, y

    @classmethod
    @st.composite
    def map_with_indexed_coproducts(draw, cls, n=Random, target=Random):
        # c : Σ_{i ∈ N} f_i → s(f_i) → T
        c = draw(cls.indexed_coproducts(n=n, target=target))
        N = len(c)
        # x : X → N
        x = draw(cls.arrows(target=N))

        return c, x
