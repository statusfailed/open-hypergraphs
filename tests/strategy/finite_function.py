import hypothesis.strategies as st

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
    def objects(draw, cls):
        return draw(st.integers(min_value=0, max_value=cls.MAX_OBJECT))

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

        # target is None, but source is not
        target = draw(nonzero_objects) if source > 0 else draw(objects)

        assert _is_valid_arrow_type(source, target)
        return source, target


    @classmethod
    @st.composite
    def finite_functions(draw, cls, source=None, target=None):
        source, target = draw(cls.arrow_type(source=source, target=target))
        if target == 0:
            table = cls.np.zeros(0, dtype=int)
        else:
            table = cls.np.random.randint(0, high=target, size=source)
        return cls.Fun(target, table)
