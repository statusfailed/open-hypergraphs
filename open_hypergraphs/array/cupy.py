"""A CuPy array backend.

.. danger::
   **Experimental Module**

   This code is not thoroughly tested.
   It's included here as a proof-of-concept for GPU acceleration.
"""
import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse import csgraph

DEFAULT_DTYPE='int64'

Type = cp.ndarray

def array(*args, **kwargs):
    return cp.array(*args, **kwargs)

def max(*args, **kwargs):
    return cp.max(*args, **kwargs)

def arange(*args, **kwargs):
    return cp.arange(*args, **kwargs)

def all(*args, **kwargs):
    return cp.all(*args, **kwargs)

def zeros(*args, **kwargs):
    return cp.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    return cp.ones(*args, **kwargs)

def cumsum(*args, **kwargs):
    return cp.cumsum(*args, **kwargs)

def sum(*args, **kwargs):
    return cp.sum(*args, **kwargs)

def repeat(*args, **kwargs):
    return cp.repeat(*args, **kwargs)

def concatenate(*args, **kwargs):
    return cp.concatenate(*args, **kwargs)

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
    # TODO: FIXME:
    # Something seems broken with cupy's weakly connected components but I can't
    # figure out what.
    # The workaround is to make the graph symmetric and compute *strongly*
    # connected components.
    # This is obviously lame and needs to be fixed.

    # make an n×n sparse matrix representing the graph with edges
    # source[i] → target[i]
    # NOTE: dtype of the values is float32 since integers aren't supported.
    # This doesn't matter though, we actually don't care about this data.
    ones = cp.ones(2*len(source), dtype='float32')

    s = cp.concatenate([source, target])
    t = cp.concatenate([target, source])
    M = sparse.csr_matrix((ones, (s, t)), shape=(n, n))

    # compute & return connected components
    c, cc_ix = csgraph.connected_components(M, connection='strong')
    return c, cc_ix

def argsort(x):
    return cp.argsort(x, kind='stable')

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
    x = cp.array(x)

    # create segment pointer array
    ptr = cp.zeros(len(x) + 1, dtype=x.dtype) # O(1) PRAM
    ptr[1:] = cp.cumsum(x)                    # O(log x) PRAM
    N = ptr[-1] # total size

    r = cp.repeat(ptr[:-1], x) # O(log x) PRAM
    return cp.arange(0, N) - r # O(1)     PRAM

def bincount(x, *args, **kwargs):
    return cp.bincount(x, *args, **kwargs)

def full(n, x, *args, **kwargs):
    return cp.full(n, x, *args, **kwargs)
