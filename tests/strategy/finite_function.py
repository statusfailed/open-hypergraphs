import hypothesis.strategies as st

def _is_valid_arrow_type(A, B):
    if B == 0:
        return A == 0
    return True

class FiniteFunctionStrategies:
    """ Hypothesis strategies for generating finite functions.
    Parametrised by the array module (np), Array backend, and Finite Function class.
    """
    np = None
    Array = None
    Fun = None

    MAX_OBJECT = 32

    @classmethod
    @st.composite
    def objects(draw, cls, allow_initial=True):
        min_value = 0 if allow_initial else 1
        return draw(st.integers(min_value=min_value, max_value=cls.MAX_OBJECT))

    @classmethod
    @st.composite
    def arrow_type(draw, cls, source=None, target=None):
        """ Generate the type of a random finite function.
        Only one type is forbidden: ``N → 0`` for ``N > 0``.
        """
        # User specified both target and source
        if target is not None and source is not None:
            if target == 0 and source != 0:
                raise ValueError("No arrows exist of type n → 0 for n != 0.")
            return source, target

        elif source is None:
            # any target
            target = draw(cls.objects()) if target is None else target

            if target == 0:
                source = 0
            else:
                source = draw(cls.objects())

            return source, target

        # target can only be initial if source is also initial.
        target = draw(cls.objects(allow_initial=(source==0)))

        assert _is_valid_arrow_type(source, target)
        return source, target


    @classmethod
    @st.composite
    def arrows(draw, cls, source=None, target=None):
        source, target = draw(cls.arrow_type(source=source, target=target))
        if target == 0:
            table = cls.np.zeros(0, dtype=int)
        else:
            table = cls.np.random.randint(0, high=target, size=source)
        return cls.Fun(target, table)

    @classmethod
    @st.composite
    def parallel_arrows(draw, cls, source=None, target=None, n=2):
        source, target = draw(cls.arrow_type(source=source, target=target))
        return [ draw(cls.arrows(source=source, target=target)) for _ in range(0, n) ]

    @classmethod
    @st.composite
    def composite_arrows(draw, cls, source=None, target=None, n=2):
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

        T = target if n == 1 else None
        arrows.append(draw(cls.arrows(source, T)))

        for i in range(0, n-1):
            S = arrows[i].target
            T = None if i < n - 1 else target
            arrows.append(draw(cls.arrows(source=S, target=T)))


        return arrows
