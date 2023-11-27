"""An implementation of finite functions as arrays.
All datastructures are ultimately built from finite functions.
For an overview, see :cite:t:`dpafsd`, Section 2.2.2.

Finite functions can be thought of as a thin wrapper around integer arrays whose
elements are within a specified range.
Here's an example of contructing a finite function:

>>> print(FiniteFunction(3, [0, 1, 2, 0]))
[0 1 2 0] : 4 → 3

Mathematically, this represents a function from the set of 4 elements to the set
of 3 elements, and so its "type" is ``4 → 3``.

There are several constructors for finite functions corresponding to useful
morphisms in category theory.
For example, the ``identity`` map is like numpy's ``arange``:

>>> print(FiniteFunction.identity(5))
[0 1 2 3 4] : 5 → 5

and the ``terminal`` map is an array of zeroes:

>>> print(FiniteFunction.terminal(5))
[0 0 0 0 0] : 5 → 1

Finite functions form a *symmetric monoidal category*.
They can be composed sequentially:

>>> print(FiniteFunction.identity(5) >> FiniteFunction.terminal(5))
[0 0 0 0 0] : 5 → 5

And in parallel:

>>> FiniteFunction.identity(5) @ FiniteFunction.terminal(5)
FiniteFunction(6, [0 1 2 3 4 5 5 5 5 5])
"""

from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Protocol, Self, List, Type, Union, Any

from open_hypergraphs.array.backend import ArrayBackend, ArrayType

Target = Union[None, int]

class FiniteFunction(ABC):
    """
    Finite functions parametrised over the underlying array type (the "backend").
    This implementation assumes there is a cls.Array member implementing various primitives.
    For example, cls.Array.sum() should compute the sum of an array.
    """
    # These are class properties, set by concrete implementations (inheriting classes).
    Dtype: Any
    Array: Type[ArrayBackend]

    # the actual data of the FiniteFunction.
    target: Target

    def __init__(self, target, table):
        self.table = table
        self.target = target

        if type(table) != self.Array.Type:
            raise ValueError(f"table must be of type {self.Array.Type}")

        if target is not None and table.dtype != self.Dtype:
            raise ValueError(f"table.dtype must be {self.Dtype} for finite target")

        if len(self.table.shape) != 1:
            raise ValueError(f"table must be a 1D array, but had shape {self.table.shape}")

        if self.source > 0 and self.target is not None:
            m = self.Array.max(table)
            if self.target <= m:
                raise ValueError(f"table max value must be less than target {self.target} but was {m}")

    def _nonfinite_target(self):
        return ValueError("FiniteFunction must have finite domain, but had target = {self.target}")

    @property
    def dtype(self) -> Any:
        return self.table.dtype

    @property
    def source(self) -> int:
        """The source (aka "domain") of this finite function"""
        return len(self.table)

    def __len__(self):
        """Same as self.source.
        Sometimes this is clearer when thinking of a finite function as an array.
        """
        return len(self.table)

    def __str__(self):
        return f'{self.table} : {self.source} → {self.target}'

    def __repr__(self):
        return f'FiniteFunction({repr(self.target)}, {repr(self.table)})'

    def __call__(self, i: int) -> int:
        if i >= self.source:
            raise ValueError(f"Calling {self} with {i} >= source {self.source}")
        return self.table[i]

    def __iter__(self):
        """ Iterate the elements of this FiniteFunction """
        return iter(self.table)

    @property
    def type(f):
        """Get the source and target of this finite function.

        Returns:
            tuple: (f.source, f.target)
        """
        return f.source, f.target

    ################################################################################
    # FiniteFunction forms a category

    @classmethod
    def identity(cls, n: int) -> 'FiniteFunction':
        """Return the identity finite function of type n → n.
        Args:
            n(int): The object of which to return the identity map

        Returns:
            FiniteFunction: Identity map at n
        """
        assert n >= 0
        return cls(n, cls.Array.arange(0, n, dtype=cls.Dtype))

    # Compute (f ; g), i.e., the function x → g(f(x))
    def compose(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        """Compose this finite function with another

        Args:
            g: A FiniteFunction for which self.target == g.source

        Returns:
            The composition f ; g.

        Raises:
            ValueError: if self.target != g.source
        """
        if f.target != g.source:
            raise ValueError(f"Can't compose FiniteFunction {f} with {g}: f.target != g.source")

        source = f.source
        target = g.target
        # Use array indexing to compute composition in parallel (if applicable
        # cls.Array backend is used)
        table = g.table[f.table]

        return type(f)(target, table)

    def __rshift__(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        return f.compose(g)

    # We can compare functions for equality in a reasonable way: by just
    # comparing elements.
    # This is basically because FinFun is skeletal, so we don't need to check
    # "up to isomorphism".
    def __eq__(f, g):
        return f.source == g.source \
           and f.target == g.target \
           and f.Array.all(f.table == g.table)

    ################################################################################
    # FiniteFunction has initial objects and coproducts
    @classmethod
    def initial(cls, b: Target, dtype=None) -> 'FiniteFunction':
        """Compute the initial map ``? : 0 → b``"""
        return cls(b, cls.Array.zeros(0, dtype=dtype or cls.Dtype))

    def to_initial(self) -> 'FiniteFunction':
        """ Turn a finite function ``f : A → B`` into the initial map ``? : 0 → B``.

        >>> f.to_initial() == FiniteFunction.initial(f.target) >> f
        """
        return type(self).initial(self.target, dtype=self.table.dtype)

    @classmethod
    def inj0(cls, a: int, b: int) -> 'FiniteFunction':
        """Compute the injection ``ι₀ : a → a + b``"""
        table = cls.Array.arange(0, a, dtype=cls.Dtype)
        return cls(a + b, table)

    @classmethod
    def inj1(cls, a: int, b: int) -> 'FiniteFunction':
        """Compute the injection ``ι₁ : b → a + b``"""
        table = cls.Array.arange(a, a + b, dtype=cls.Dtype)
        return cls(a + b, table)

    def inject0(f: 'FiniteFunction', b: int) -> 'FiniteFunction':
        """
        Directly compute (f ; ι₀) instead of by composition.

        >>> f.inject0(b) == f >> ι₀
        """
        if f.target is None:
            raise f._nonfinite_target()
        return type(f)(f.target + b, f.table)

    def inject1(f: 'FiniteFunction', a: int) -> 'FiniteFunction':
        """
        Directly compute (f ; ι₁) instead of by composition.

        >>> f.inject1(a) == f >> ι₁
        """
        if f.target is None:
            raise f._nonfinite_target()
        return type(f)(a + f.target, a + f.table)

    def coproduct(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        """ Given maps ``f : A₀ → B`` and ``g : A₁ → B``
        compute the coproduct ``f.coproduct(g) : A₀ + A₁ → B``"""
        assert f.target == g.target
        assert f.table.dtype == g.table.dtype
        target = f.target
        table = type(f).Array.concatenate([f.table, g.table], f.dtype)
        return type(f)(target, table)

    def __add__(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        """ Inline coproduct """
        return f.coproduct(g)

    ################################################################################
    # FiniteFunction as a strict symmetric monoidal category
    @staticmethod
    def unit() -> int:
        """ return the unit object of the category """
        return 0

    def tensor(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        """ Given maps
        ``f : A₀ → B₀`` and
        ``g : A₁ → B₁``
        compute the *tensor* product
        ``f.tensor(g) : A₀ + A₁ → B₀ + B₁``"""
        # The tensor (f @ g) is the same as (f;ι₀) + (g;ι₁)
        # however, we compute it directly for the sake of efficiency
        if f.target is None:
            raise f._nonfinite_target()
        if g.target is None:
            raise f._nonfinite_target()
        T = type(f)
        table = T.Array.concatenate([f.table, g.table + f.target], f.Dtype)
        return T(f.target + g.target, table)

    def __matmul__(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        return f.tensor(g)

    @classmethod
    def twist(cls, a: int, b: int) -> 'FiniteFunction':
        # Read a permutation as the array whose ith position denotes "where to send" value i.
        # e.g., twist_{2, 3} = [3 4 0 1 2]
        #       twist_{2, 1} = [1 2 0]
        #       twist_{0, 2} = [0 1]
        table = cls.Array.concatenate([b + cls.Array.arange(0, a, cls.Dtype), cls.Array.arange(0, b, cls.Dtype)], dtype=cls.Dtype)
        return cls(a + b, table)

    ################################################################################
    # Coequalizers for FiniteFunction

    def coequalizer(f: 'FiniteFunction', g: 'FiniteFunction') -> 'FiniteFunction':
        """
        Given finite functions    ``f, g : A → B``,
        return the *coequalizer*  ``q    : B → Q``
        which is the unique arrow such that  ``f >> q = g >> q``
        having a unique arrow to any other such map.
        """

        if f.type != g.type:
            raise ValueError(
                f"cannot coequalize arrows {f} and {g} of different types: {f.type} != {g.type}")

        # connected_components returns:
        #   Q:        number of components
        #   q: B → Q  map assigning vertices to their component
        # For the latter we have that
        #   * if f.table[i] == g.table[i]
        #   * then q[f.table[i]] == q[g.table[i]]
        # NOTE: we pass f.target so the size of the sparse adjacency matrix
        # representing the graph can be computed efficiently; otherwise we'd
        # have to take a max() of each table.
        # Q: number of connected components
        if f.target is None:
            raise f._nonfinite_target()
        cls = type(f)
        Q, q = cls.Array.connected_components(f.table, g.table, f.target, f.Dtype)
        return cls(Q, q)

    def coequalizer_universal(q: 'FiniteFunction', f: 'FiniteFunction') -> 'FiniteFunction':
        """
        Given a coequalizer q : B → Q of morphisms a, b : A → B
        and some f : B → B' such that f(a) = f(b),
        Compute the universal map u : Q → B'
        such that q ; u = f.
        """
        # NOTE: we *don't* check that f.target is None: we can compute the
        # universal map when f has non-finite target!
        if q.target is None:
            raise q._nonfinite_target()

        target = f.target
        table = q.Array.zeros(q.target, dtype=f.table.dtype)
        # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
        # same memory location in the 'u' array can happen in parallel, with an
        # arbitrary write succeeding.
        # Note that this doesn't affect correctness because q and f are co-forks,
        # and q is a coequalizer.
        # However, this won't perform well on e.g., GPU hardware. FIXME!
        table[q.table] = f.table
        u = type(f)(target, table)
        if not (q >> u) == f:
            raise ValueError("Universal morphism doesn't make {q} ; {u} commute. Is q really a coequalizer?")
        return u


    ################################################################################
    # FiniteFunction also has cartesian structure which is useful
    @classmethod
    def terminal(cls, a: int) -> 'FiniteFunction':
        """ Compute the terminal map ``! : a → 1``. """
        return cls(1, cls.Array.zeros(a, dtype=cls.Dtype))

    # TODO: rename this "element"?
    @classmethod
    def singleton(cls, x: int, b: int | None, dtype=None) -> 'FiniteFunction':
        """ return the singleton array ``[x]`` whose domain is ``b``. """
        return cls.constant(x, 1, b, dtype or cls.Dtype)

    @classmethod
    def constant(cls, x: int, a: int, b: int  | None, dtype=None) -> 'FiniteFunction':
        """ ``constant(x, a, b)`` is the constant function of type ``a → b``
        mapping all inputs to the value ``x``. """
        if type(b) is int and x >= b:
            raise ValueError(f"{x} is not an element of the set {{0..{b}}}")
        return cls(b, cls.Array.full(a, x, dtype=dtype or cls.Dtype))

    ################################################################################
    # Sorting morphisms
    def argsort(f: Self) -> Self:
        """
        Given a finite function                     ``f : A → B``
        Return the *stable* sorting permutation     ``p : A → A``
        such that                                   ``p >> f``  is monotonic.
        """
        return type(f)(f.source, f.Array.argsort(f.table).astype(f.Dtype))

    ################################################################################
    # Useful permutations

    @classmethod
    def transpose(cls, a: int, b: int, dtype=None) -> 'FiniteFunction':
        """ ``transpose(a, b)`` is the "transposition permutation" for an ``a → b`` matrix.

        Given an ``b*a``-dimensional input thought of as a matrix in row-major
        order with ``b`` rows and ``a`` columns,
        ``transpose(a, b)`` computes the "target indices" in the transpose.
        So if we have a matrix ``M : A → B``
        and a matrix ``N : B → A``,
        then setting indexes ``N[transpose(a, b)] = M`` is the same as writing
        ``N = M.T``
        """
        table = cls.Array.zeros(b*a, dtype=dtype or cls.Dtype)
        i = cls.Array.arange(0, b*a, dtype=cls.Dtype)
        # TODO: this can be done without arithmetic operators; but is it faster?
        # A quick sketch of how:
        # repeating the vector (0, m, 2m, ... nm) and adding with (0 0 0 ... 1 1 1 ... 2 2 2 ... )
        #   something like this: repeat(m * range(n), n) + repeat(n, range(m))
        table[i] = (i % a) * b + i // a
        return cls(b*a, table)

    ################################################################################
    # Sequential-only methods

    @classmethod
    def coproduct_list(cls, fs: List['FiniteFunction'], target=None, dtype=None):
        """ Compute the coproduct of a list of finite functions. O(n) in size of the result.

        .. warning::
            Does not speed up to O(log n) in the parallel case.
        """
        if len(fs) == 0:
            return cls.initial(target, dtype)

        # all targets must be equal
        assert all(f.target == g.target for f, g in zip(fs, fs[:1]))
        return cls(fs[0].target, cls.Array.concatenate([f.table for f in fs], fs[0].dtype))

    @classmethod
    def tensor_list(cls, fs: List['FiniteFunction']):
        """ Compute the tensor product of a list of finite functions. O(n) in size of the result.

        .. warning::
            Does not speed up to O(log n) in the parallel case.
        """
        if len(fs) == 0:
            return cls.initial(0)

        targets = cls.Array.array([f.target for f in fs], cls.Dtype)
        offsets = cls.Array.zeros(len(targets) + 1, dtype=cls.Dtype)
        offsets[1:] = cls.Array.cumsum(targets) # exclusive scan
        table = cls.Array.concatenate([f.table + offset for f, offset in zip(fs, offsets[:-1])], cls.Dtype)
        return cls(offsets[-1], table)


    ################################################################################
    # Finite coproducts
    def injections(s: 'FiniteFunction', a: 'FiniteFunction'):
        """
        Given a finite function ``s : N → K``
        representing the objects of the coproduct
        ``Σ_{n ∈ N} s(n)``
        whose injections have the type
        ``ι_x : s(x) → Σ_{n ∈ N} s(n)``,
        and given a finite map
        ``a : A → N``,
        compute the coproduct of injections

        .. code-block:: text

            injections(s, a) : Σ_{x ∈ A} s(x) → Σ_{n ∈ N} s(n)
            injections(s, a) = Σ_{x ∈ A} ι_a(x)

        So that ``injections(s, id) == id``

        Note that when a is a permutation,
        injections(s, a) is a "blockwise" version of that permutation with block
        sizes equal to s.
        """
        # segment pointers
        if s.table.dtype != a.table.dtype:
            raise ValueError(f"s and a must have the same dtype, but got {s.table.dtype} and {a.table.dtype}")

        Array = a.Array
        dtype = a.table.dtype

        # cumsum is inclusive, we need exclusive so we just allocate 1 more space.
        p = Array.zeros(s.source + 1, dtype=dtype)
        p[1:] = Array.cumsum(s.table)

        k = a >> s # avoid recomputation
        r = Array.segmented_arange(k.table)
        # NOTE: p[-1] is sum(s).
        cls = type(s)
        return cls(p[-1], r + cls.Array.repeat(p[a.table], k.table))

class HasFiniteFunction(Protocol):
    """ Classes which have a chosen finite function implementation """
    @classmethod
    def FiniteFunction(cls) -> Type[FiniteFunction]:
        ...

@dataclass
class IndexedCoproduct(HasFiniteFunction):
    """ A finite coproduct of finite functions.
    You can think of it simply as a segmented array.
    Categorically, it represents a finite coproduct::

        Σ_{i ∈ N} f_i : s(f_i) → Y

    as a pair of maps::

        sources: N            → Nat     (target is natural numbers)
        values : sum(sources) → Σ₀
    """
    # sources: an array of segment sizes (note: not ptrs)
    sources: FiniteFunction

    # values: the values of the coproduct
    values: FiniteFunction

    def __post_init__(self):
        assert type(self.sources) == self.FiniteFunction()
        assert type(self.values) == self.FiniteFunction()
        # TODO FIXME: make Array type derivable from FiniteFunction
        self.Array = self.FiniteFunction().Array

        # we always ignore the target of sources; this ensures
        # roundtrippability.
        assert self.sources.target is None
        assert len(self.values) == self.Array.sum(self.sources.table)

    @property
    def dtype(self):
        """ Return the dtype of the underlying table """
        return self.sources.dtype

    @classmethod
    def initial(cls, target: Target, dtype=None) -> 'IndexedCoproduct':
        return cls(
            cls.FiniteFunction().initial(None, dtype=dtype),
            cls.FiniteFunction().initial(target, dtype=dtype))

    @classmethod
    def singleton(cls, values: FiniteFunction, dtype=None) -> 'IndexedCoproduct':
        """ Turn a :py:class:`FiniteFunction` ``f : A → B`` into an :py:class:`IndexedCoproduct`
        `` Σ_{x ∈ 1} f : A → B """
        sources = cls.FiniteFunction().singleton(len(values), None, dtype)
        return cls(sources, values)

    @classmethod
    def elements(cls, values: FiniteFunction, dtype=None) -> 'IndexedCoproduct':
        """ Turn a :py:class:`FiniteFunction` ``f : A → B`` into an :py:class:`IndexedCoproduct`
        `` Σ_{a ∈ A} f_a : A → B """
        sources = cls.FiniteFunction().constant(1, len(values), None, dtype=dtype)
        return cls(sources, values)

    @property
    def target(self) -> Target:
        return self.values.target

    def __len__(self):
        """ return the number of finite functions in the coproduct """
        return len(self.sources)

    @classmethod
    def from_list(cls, target, fs: List['FiniteFunction'], dtype=None) -> 'IndexedCoproduct':
        """ Create an `IndexedCoproduct` from a list of :py:class:`FiniteFunction` """
        assert all(target == f.target for f in fs)
        Dtype = cls.FiniteFunction().Dtype
        sources_table = cls.FiniteFunction().Array.array([len(f) for f in fs], dtype=Dtype)
        return cls(
            sources=cls.FiniteFunction()(None, sources_table),
            values=cls.FiniteFunction().coproduct_list(fs, target=target, dtype=dtype))

    def __iter__(self):
        """ Yield an iterator of the constituent finite functions

        >>> list(IndexedCoproduct.from_list(fs)) == fs
        True
        """
        N     = len(self.sources)

        # Compute source pointers
        s_ptr = self.Array.zeros(N+1, dtype=self.sources.table.dtype)
        s_ptr[1:] = self.Array.cumsum(self.sources.table)

        for i in range(0, N):
            yield self.FiniteFunction()(self.target, self.values.table[s_ptr[i]:s_ptr[i+1]])

    def coproduct(x: 'IndexedCoproduct', y: 'IndexedCoproduct') -> 'IndexedCoproduct':
        return type(x)(
            sources = x.sources + y.sources,
            values  = x.values + y.values)

    def __add__(x: 'IndexedCoproduct', y: 'IndexedCoproduct') -> 'IndexedCoproduct':
        return x.coproduct(y)

    def tensor(x: 'IndexedCoproduct', y: 'IndexedCoproduct') -> 'IndexedCoproduct':
        return type(x)(
            sources = x.sources + y.sources,
            values  = x.values @ y.values)

    def __matmul__(x: 'IndexedCoproduct', y: 'IndexedCoproduct') -> 'IndexedCoproduct':
        return x.tensor(y)

    @classmethod
    def tensor_list(cls, xs: List['IndexedCoproduct']):
        """ Concatenate a nonempty sequence of IndexedCoproduct """
        if len(xs) == 0:
            raise ValueError("xs must be a nonempty list")
        sources = cls.FiniteFunction().coproduct_list([x.sources for x in xs])
        values  = cls.FiniteFunction().tensor_list([x.values for x in xs])
        return cls(sources, values)

    def indexed_values(self, x: FiniteFunction) -> FiniteFunction:
        """Like ``map_indexes`` but only computes the ``values`` array of an IndexedCoproduct"""
        assert x.target == len(self.sources)
        return self.sources.injections(x) >> self.values

    def map_values(self, x: FiniteFunction) -> 'IndexedCoproduct':
        """ Given an :py:class:`IndexedCoproduct` of finite functions::

            Σ_{i ∈ X} f_i : Σ_{i ∈ X} A_i → B

        and a finite function::

            b : B → C

        return a new :py:class:`IndexedCoproduct` representing::

            Σ_{i ∈ X} f_i : Σ_{i ∈ W} A_i → C
        """
        return type(self)(
            sources = self.sources,
            values = self.values >> x)

    def map_indexes(self, x: FiniteFunction) -> 'IndexedCoproduct':
        """ Given an :py:class:`IndexedCoproduct` of finite functions::

            Σ_{i ∈ X} f_i : Σ_{i ∈ X} A_i → B

        and a finite function::

            x : W → X

        return a new :py:class:`IndexedCoproduct` representing::

            Σ_{i ∈ X} f_{x(i)} : Σ_{i ∈ W} A_{x(i)} → B
        """
        return type(self)(
            sources = x >> self.sources,
            values = self.indexed_values(x))

    def flatmap(x, y: 'IndexedCoproduct') -> 'IndexedCoproduct':
        """ Compose two IndexedCoproducts. """
        # TODO: docstring!
        #
        #   x : Σ_{a ∈ A} s(a) → B      #   aka A → B*
        #   y : Σ_{b ∈ B} s(b) → C      #   aka B → C*
        #   z : Σ_{a ∈ A} s'(a) → C      #   aka A → C*
        #
        # TODO: be explicit that s'(a) = Σ_{b ∈ B} s(b) ... ?
        assert len(x.values) == len(y)

        # segmented sum of x.sources
        return type(x)(
            sources = x.FiniteFunction()(None, x.FiniteFunction().Array.segmented_sum(x.sources.table, y.sources.table)),
            values  = y.values)

class HasIndexedCoproduct(HasFiniteFunction):
    """ Classes which have a chosen indexed coproduct implementation """
    @classmethod
    @abstractmethod
    def IndexedCoproduct(cls) -> Type[IndexedCoproduct]:
        ...
