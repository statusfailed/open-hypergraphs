""" Layered decomposition for numpy-backed :py:class:`OpenHypergraph`.
Note that this (currently) uses SciPy sparse arrays, so it can't be used for diagrams backed
by other array libraries (e.g., CuPy).

Use the ``layer`` function to assign a *layering* to operations in the diagram.
This is like a topological sort, except multiple operations can be assigned to
the same layering.
"""
from typing import Tuple
import numpy as np
import scipy.sparse as sp

# We use FiniteFunction, OpenHypergraph etc. from the Numpy backend.
import open_hypergraphs.finite_function as f
from open_hypergraphs.numpy.types import *

# NOTE: types are a bit of a hack; actually this function expects the numpy-backed FiniteFunction!
def make_sparse(s: f.FiniteFunction, t: f.FiniteFunction) -> sp.csr_array:
    """Given finite functions ``s : E → A`` and ``t : E → B``
    representing a bipartite graph ``G : A → B``,
    return the sparse ``B×A`` adjacency matrix representing ``G``.
    """
    assert s.source == t.source
    N = s.source
    # (data, (row, col))
    # rows are *outputs*, so row = t.table
    # cols are *inputs* so col = s.table
    return sp.csr_array((np.ones(N, dtype=bool), (t.table, s.table)), shape=(t.target, s.target))

def operation_adjacency(f: OpenHypergraph) -> sp.csr_array:
    """ Construct the underlying graph of operation adjacency from an :py:class:`OpenHypergraph`.
    An operation ``x`` is adjacent to an operation ``y`` if there is a directed
    path from ``x`` to ``y`` going through a single node.
    """
    Array = FiniteFunction.Array

    x = FiniteFunction.Array.arange(0, len(f.H.x), FiniteFunction.Dtype)
    x_s = FiniteFunction(len(f.H.x), FiniteFunction.Array.repeat(x, f.H.s.sources.table))
    x_t = FiniteFunction(len(f.H.x), FiniteFunction.Array.repeat(x, f.H.t.sources.table))

    # ● → □
    # edge source is a wire ●, target is an operation □
    Mi = make_sparse(f.H.s.values, x_s)
    # □ → ●
    # edge source is an op □, target is a wire ●
    Mo = make_sparse(x_t, f.H.t.values)
    return Mi @ Mo

# Kahn's Algorithm, but vectorised a bit.
# https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
# Attempts to 'parallelize' the layering where possible, e.g.:
#
#          ○--\
#              ○---○
#          ○--/
#
#       ------------
#
#  layer   0   1   2
#
def kahn(adjacency: sp.csr_array) -> Tuple[np.ndarray, sp.csr_array]:
    """ A version of Kahn's algorithm which assigns a *layering* to each node,
    but where multiple nodes can have the same layering.

    Returns a pair of arrays ``(order, visited)``.
    ``order[v]`` is a natural number indicating the computed ordering of node ``v``,
    and ``visited[v]`` is 1 if and only if ``v`` was visited while traversing the graph.

    If not all vertices were visited, the graph had a cycle.
    """
    n, m = adjacency.shape
    assert n == m
    adjacency = adjacency.astype(int)

    # NOTE: convert to numpy ndarray instead of matrix given by adjacency;
    # this makes indexing behaviour a bit nicer!
    # NOTE: we use reshape here because adjacency.sum() gives different dimensions when input is 0x0!
    indegree = np.asarray(adjacency.sum(axis=1, dtype=int)).reshape((n,))

    # return values
    visited  = np.zeros(n, dtype=bool)
    order    = np.zeros(n, dtype=FiniteFunction.Dtype)

    # start at nodes with no incoming edges
    start = (indegree == 0).nonzero()[0]

    # initialize the frontier at the requested start nodes
    k = len(start)
    frontier = sp.csr_array((np.ones(k, int), (start, np.zeros(k, int))), (n, 1))

    # as long as the frontier contains some nodes, we'll keep going.
    depth = 0
    while frontier.nnz > 0:
        # Mark nodes in the current frontier as visited,
        # and set their layering value ('order') to the current depth.
        frontier_ixs = frontier.nonzero()[0]
        visited[frontier_ixs] = True
        order[frontier_ixs] = depth

        # Find "reachable", which is the set of nodes adjacent to the current frontier.
        # Decrement the indegree of each adjacent node by the number of edges between it and the frontier.
        # NOTE: nodes only remain in the frontier for ONE iteration, so this
        # will only decrement once for each edge.
        reachable = adjacency @ frontier
        reachable_ix = reachable.nonzero()[0]
        indegree[reachable_ix] -= reachable.data

        # Compute the new frontier: the reachable nodes with indegree equal to zero (no more remaining edges!)
        # NOTE: indegree is an (N,1) matrix, so we select out the first row.
        new_frontier_ixs = reachable_ix[indegree[reachable_ix] == 0]
        k = len(new_frontier_ixs)
        frontier = sp.csr_array((np.ones(k, int), (new_frontier_ixs, np.zeros(k, int))), shape=(n, 1))

        # increment depth so the new frontier will be layered correctly
        depth += 1

    # Return the layering (order) and whether each node was visited.
    # Note that if not np.all(visited), then there must be a cycle.
    return order, visited

def layer(f: OpenHypergraph) -> Tuple[FiniteFunction, np.ndarray]:
    """ Assign a *layering* to an :py:class:`OpenHypergraph` `F`
        This computes a FiniteFunction ``layer(F) : F(X) → K``
        mapping operations of ``F.H`` to a natural number in the ``range(0, K)``
        where `K <= F(X)`.
        This mapping is 'dense' in the sense that for each ``i ∈ {0..K}``,
        there is always some ○-node v for which ``layer(F)(v) = i``.
    """
    # (Partially) topologically sort it using a kahn-ish algorithm
    # NOTE: if completed is not all True, then some generators were not ordered.
    # In this case, we leave their ordering as 0, since they contain cycles.
    M = operation_adjacency(f)
    ordering, completed = kahn(M)
    return FiniteFunction(len(f.H.x), ordering), completed
