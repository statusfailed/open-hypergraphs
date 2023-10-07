"""numpy- and scipy-backed arrays and algorithms.
Almost all exposed functions are thin wrappers around numpy functions.
The only exceptions are:

* :func:`connected_components` -- wraps `scipy.sparse.csgraph.connected_components`
* :func:`segmented_arange` -- a subroutine implemented in terms of the other primitives

This module is the default array backend.
It's used by :py:class:`FiniteFunction`.
"""
import numpy as np
import scipy.sparse as sparse

from open_hypergraphs.array.backend import ArrayBackend


class NumpyBackend(ArrayBackend):
    DEFAULT_DTYPE = np.int64

    Type = np.ndarray
    """ The underlying array type used by functions in the backend. For numpy this is ``np.ndarray``.

       :meta hide-value:
    """
    # NOTE: we use :meta hide-value: above because numpy is mocked, so sphinx will
    # have the incorrect value in documentation.

    @classmethod
    def to_dtype(cls, dtype=None):
        return cls.DEFAULT_DTYPE

    @classmethod
    def array(cls, elems, dtype=None):
        dtype = cls.to_dtype(dtype)
        return np.fromiter(elems, dtype)

    @classmethod
    def arange(cls, start: int, stop: int, dtype=None) -> np.ndarray:
        dtype = cls.to_dtype(dtype)
        return np.arange(start=start, stop=stop, dtype=dtype)

    @classmethod
    def zeros(cls, n: int, dtype=None):
        dtype = cls.to_dtype(dtype)
        return np.zeros(n, dtype=dtype)

    @classmethod
    def ones(cls, n: int, dtype=None):
        dtype = cls.to_dtype(dtype)
        return np.ones(n, dtype=dtype)

    @classmethod
    def max(cls, x: np.ndarray):
        return np.max(x)

    @classmethod
    def all(cls, x: np.ndarray):
        return np.all(x)

    @classmethod
    def any(cls, x: np.ndarray):
        return np.any(x)

    @classmethod
    def cumsum(cls, x):
        return np.cumsum(x)

    @classmethod
    def sum(cls, *args, **kwargs):
        return np.sum(*args, **kwargs)

    @classmethod
    def repeat(cls, *args, **kwargs):
        return np.repeat(*args, **kwargs)

    @classmethod
    def concatenate(cls, *args, **kwargs):
        return np.concatenate(*args, **kwargs)

    # Compute the connected components of a graph.
    # connected components of a graph, encoded as a list of edges between points
    # so we have s, t arrays encoding edges (s[i], t[i]) of a square n×n matrix.
    # NOTE: we have to wrap libraries since we don't tend to get a consistent interface,
    # and don't want to expose e.g. sparse graphs in the main code.
    @classmethod
    def connected_components(cls, source, target, n, dtype=None):
        """Compute the connected components of a graph with ``N`` nodes,
        whose edges are encoded as a pair of arrays ``(source, target)``
        such that the edges of the graph are ``source[i] → target[i]``.

        Args:
            source(array): A length-N array with elements in the set ``{0 .. N - 1}``.
            target(array): A length-N array with elements in the set ``{0 .. N - 1}``.

        Returns:
            (int, array):

            A pair ``(c, cc_ix)`` of the number of connected components
            ``c`` and a mapping from nodes to connected components ``cc_ix``.
        """
        if len(source) != len(target):
            raise ValueError("Expected a graph encoded as a pair of arrays (source, target) of the same length")

        assert len(source) == len(target)

        # make an n×n sparse matrix representing the graph with edges
        # source[i] → target[i]
        ones = np.ones(len(source), dtype=cls.DEFAULT_DTYPE)
        M = sparse.csr_matrix((ones, (source, target)), shape=(n, n))

        # compute & return connected components
        c, cc_ix = sparse.csgraph.connected_components(M)
        return c, cc_ix

    # def bincount(x, *args, **kwargs):
        # return np.bincount(x, *args, **kwargs)

    # def full(n, x, *args, **kwargs):
        # kwargs.setdefault('dtype', DEFAULT_DTYPE)
        # return np.full(n, x, *args, **kwargs)

    # def argsort(x):
        # return np.argsort(x, kind='stable')
