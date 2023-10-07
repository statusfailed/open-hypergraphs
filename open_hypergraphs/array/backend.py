""" Loosely typed array backend protocol """
from typing import Generic, Type, TypeVar, Sequence, List, Tuple, Any
from typing_extensions import Protocol

from collections.abc import MutableMapping

# Arrays are (very) loosely typed
class ArrayType(MutableMapping):
    dtype: Any

    def __sub__(self, other):
        ...

A = TypeVar('A', bound=ArrayType)

# A generic array backend for arrays of type A.
class ArrayBackend(Protocol[A]):

    DEFAULT_DTYPE: Any
    Type: Type[A]

    # Construct an array from an iterable of elements.
    @classmethod
    def array(cls, elems, dtype=None) -> A:
        ...

    ########################################
    # Constructors
    @classmethod
    def arange(cls, start: int, stop: int, dtype=None) -> A:
        ...
    
    @classmethod
    def zeros(cls, n: int, dtype=None) -> A:
        ...

    @classmethod
    def ones(cls, n: int, dtype=None) -> A:
        ...

    ########################################
    # reductions
    # NOTE: return values are untyped, but represent scalars.

    @classmethod
    def max(cls, array: A) -> Any:
        ...

    @classmethod
    def all(cls, array: A) -> Any:
        ...

    @classmethod
    def any(cls, array: A) -> Any:
        ...

    @classmethod
    def sum(cls, array: A) -> Any:
        ...

    ########################################
    # transformations

    @classmethod
    def cumsum(cls, array: A) -> A:
        ...

    @classmethod
    def repeat(cls, x: A, repeats: A) -> A:
        ...

    @classmethod
    def concatenate(cls, x: List[A]) -> A:
        ...

    ########################################
    # connected components

    @classmethod
    def connected_components(cls, self: A, other: A, m: int) -> Tuple[int, A]:
        ...

    ########################################
    # utilities

    @classmethod
    def argsort(cls, x: A) -> A:
        ...

    ########################################
    # Non-primitive routines (i.e., vector routines built out of primitives)

    # Segmented Arange example run:
    #   x       = [ 2 3 0 5 ]
    #   output  = [ 0 1 | 0 1 2 | | 0 1 2 3 4 ]
    # compute ptrs
    #   p       = [ 0 2 5 5 ]
    #   r       = [ 0 0 | 2 2 2 | | 5 5 5 5 5 ]
    #   i       = [ 0 1   2 3 4     5 6 7 8 9 ]
    #   i - r   = [ 0 1 | 0 1 2 | | 0 1 2 3 4 ]
    # Note: r is computed as repeat(p, n)
    #
    # Complexity
    #   O(n)     sequential
    #   O(log n) PRAM CREW (cumsum is log n)
    @classmethod
    def segmented_arange(cls, x: A) -> A:
        """Given an array of *sizes*, ``[x₀, x₁, ...]``  output an array equal to the concatenation
        ``concatenate([arange(x₀), arange(x₁), ...])``

        >>> FiniteFunction._Array.segmented_arange([5, 2, 3, 1])
        array([0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0])

        Params:
            x: An array of the sizes of each "segment" of the output

        Returns:
            array:

            segmented array with segment ``i`` equal to ``arange(i)``.
        """
        x = cls.array(x, dtype=cls.DEFAULT_DTYPE)

        # create segment pointer array
        ptr = cls.zeros(len(x) + 1, dtype=x.dtype) # O(1) PRAM
        ptr[1:] = cls.cumsum(x)                    # O(log x) PRAM
        N = ptr[-1] # total size

        r = cls.repeat(ptr[:-1], x) # O(log x) PRAM
        return cls.arange(0, N) - r # O(1)     PRAM
