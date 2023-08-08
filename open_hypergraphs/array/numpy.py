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

DEFAULT_DTYPE='int64'

Type = np.ndarray
""" The underlying array type used by functions in the backend. For numpy this is ``np.ndarray``.

   :meta hide-value:
"""
# NOTE: we use :meta hide-value: above because numpy is mocked, so sphinx will
# have the incorrect value in documentation.

def array(*args, **kwargs):
    kwargs.setdefault('dtype', DEFAULT_DTYPE)
    return np.fromiter(*args, **kwargs)

def max(*args, **kwargs):
    return np.max(*args, **kwargs)

def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)

def all(*args, **kwargs):
    return np.all(*args, **kwargs)

def zeros(*args, **kwargs):
    kwargs.setdefault('dtype', DEFAULT_DTYPE)
    return np.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    kwargs.setdefault('dtype', DEFAULT_DTYPE)
    return np.ones(*args, **kwargs)

def cumsum(*args, **kwargs):
    return np.cumsum(*args, **kwargs)

def sum(*args, **kwargs):
    return np.sum(*args, **kwargs)

def repeat(*args, **kwargs):
    return np.repeat(*args, **kwargs)

def concatenate(*args, **kwargs):
    return np.concatenate(*args, **kwargs)

# Compute the connected components of a graph.
# connected components of a graph, encoded as a list of edges between points
# so we have s, t arrays encoding edges (s[i], t[i]) of a square n×n matrix.
# NOTE: we have to wrap libraries since we don't tend to get a consistent interface,
# and don't want to expose e.g. sparse graphs in the main code.
def connected_components(source, target, n, dtype=DEFAULT_DTYPE):
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
    ones = np.ones(len(source), dtype=DEFAULT_DTYPE)
    M = sparse.csr_matrix((ones, (source, target)), shape=(n, n))

    # compute & return connected components
    c, cc_ix = sparse.csgraph.connected_components(M)
    return c, cc_ix

def argsort(x):
    return np.argsort(x, kind='stable')

################################################################################
# Non-primitive routines (i.e., vector routines built out of primitives)
# TODO: implement an "asbtract array library" class, inherit faster impls for numpy etc.

# e.g.,
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
def segmented_arange(x):
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
    x = np.array(x, dtype=DEFAULT_DTYPE)

    # create segment pointer array
    ptr = np.zeros(len(x) + 1, dtype=x.dtype) # O(1) PRAM
    ptr[1:] = np.cumsum(x)                    # O(log x) PRAM
    N = ptr[-1] # total size

    r = np.repeat(ptr[:-1], x) # O(log x) PRAM
    return np.arange(0, N) - r # O(1)     PRAM

def bincount(x, *args, **kwargs):
    return np.bincount(x, *args, **kwargs)

def full(n, x, *args, **kwargs):
    kwargs.setdefault('dtype', DEFAULT_DTYPE)
    return np.full(n, x, *args, **kwargs)
