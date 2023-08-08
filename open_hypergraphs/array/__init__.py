"""Array backends for :ref:`open_hypergraphs.finite_function`.

Each sub-module of :ref:`open_hypergraphs.array` is an "array backend".
Array backends provide a small number of *primitive functions*
like :func:`open_hypergraphs.array.numpy.zeros` and :func:`open_hypergraphs.array.numpy.arange` .
See :ref:`open_hypergraphs.array.numpy` (the default backend) for a list.

.. warning::
   This part of the API is likely to change significantly in future releases.

.. autosummary::
    :toctree: _autosummary
    :recursive:

    open_hypergraphs.array.numpy
    open_hypergraphs.array.cupy
"""
