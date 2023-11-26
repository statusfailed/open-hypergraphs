from typing import Self, List
from dataclasses import dataclass
from open_hypergraphs.finite_function import FiniteFunction
from open_hypergraphs.hypergraph import *

@dataclass
class OpenHypergraph(HasHypergraph):
    """ An OpenHypergraph is a cospan in Hypergraph whose feet are discrete. """
    s: FiniteFunction
    t: FiniteFunction
    H: Hypergraph

    def __post_init__(self):
        if self.s.target != self.H.w.source:
            raise ValueError(f"self.s.target must equal H.w.source, but {self.s.target} != {self.H.w.source}")
        if self.t.target != self.H.w.source:
            raise ValueError(f"self.t.target must equal H.w.source, but {self.t.target} != {self.H.w.source}")

    @property
    def source(self):
        return self.s >> self.H.w

    @property
    def target(self):
        return self.t >> self.H.w

    def signature(self):
        return self.H.w.to_initial(), self.H.x.to_initial()

    @classmethod
    def identity(cls, w, x):
        if x.source != 0:
            raise ValueError(f"x.source must be 0, but was {x.source}")
        s = t = cls.FiniteFunction().identity(w.source)
        H = cls.Hypergraph().discrete(w, x)
        return cls(s, t, H)

    def compose(f: 'OpenHypergraph', g: 'OpenHypergraph'):
        assert f.target == g.source
        h = f @ g
        q = f.t.inject0(g.H.W).coequalizer(g.s.inject1(f.H.W))
        return type(f)(
            s = f.s.inject0(g.H.W) >> q,
            t = g.t.inject1(f.H.W) >> q,
            H = h.H.coequalize_vertices(q))

    def __rshift__(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return f.compose(g)

    ##############################
    # Symmetric monoidal structure

    @classmethod
    def unit(cls, w, x):
        """ The empty open hypergraph; the monoidal unit ``OpenHypergraph.unit : I → I`` """
        assert len(w) == 0
        assert len(x) == 0
        e = cls.FiniteFunction().initial(0)
        return cls(e, e, cls.Hypergraph().empty(w, x))

    def unit_of(self):
        """ Given an OpenHypergraph, return the unit over the same signature """
        dtype = self.s.table.dtype
        return type(self).unit(self.H.w.to_initial(), self.H.x.to_initial())

    def tensor(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return type(f)(
            s = f.s @ g.s,
            t = f.t @ g.t,
            H = f.H + g.H)

    def __matmul__(f: 'OpenHypergraph', g: 'OpenHypergraph') -> 'OpenHypergraph':
        return f.tensor(g)

    @classmethod
    def twist(cls, a: FiniteFunction, b: FiniteFunction, x: FiniteFunction) -> 'OpenHypergraph':
        if len(x) != 0:
            raise ValueError(f"twist(a, b, x) must have len(x) == 0, but len(x) == {len(x)}")
        s = cls.FiniteFunction().twist(len(a), len(b))
        t = cls.FiniteFunction().identity(len(a) + len(b))
        # NOTE: because the twist is in the source map, the type of the wires in
        # this hypergraph is b + a instead of a + b! (this matters!)
        H = cls.Hypergraph().discrete(b + a, x)
        return cls(s, t, H)

    ##############################
    # Frobenius

    def dagger(self):
        return type(self)(self.t, self.s, self.H)

    @classmethod
    def spider(cls, s: FiniteFunction, t: FiniteFunction, w: FiniteFunction, x: FiniteFunction) -> Self:
        H = cls.Hypergraph().discrete(w, x)
        return cls(s, t, H)

    @classmethod
    def half_spider(cls, s: FiniteFunction, w: FiniteFunction, x: FiniteFunction) -> Self:
        t = cls.FiniteFunction().identity(len(w))
        return cls.spider(s, t, w, x)

    @classmethod
    def singleton(cls, x: FiniteFunction, a: FiniteFunction, b: FiniteFunction) -> 'OpenHypergraph':
        """ Given FiniteFunctions ``a : A → Σ₀`` and ``b : B → Σ₀`` and an
        operation ``x : 1 → Σ₁``, create an open hypergraph with a single
        operation ``x`` with type ``A → B``. """
        if len(x) != 1:
            raise ValueError(f"len(x) must be 1, but was {len(x)}")

        if a.target != b.target:
            raise ValueError(f"a and b must have same target, but a.target == {a.target} and b.target == {b.target}")

        s = cls.FiniteFunction().inj0(len(a), len(b))
        t = cls.FiniteFunction().inj1(len(a), len(b))
        H = cls.Hypergraph()(
            s = cls.IndexedCoproduct().singleton(s),
            t = cls.IndexedCoproduct().singleton(t),
            w = a + b,
            x = x)

        return cls(s, t, H)

    def permute(self, w: FiniteFunction, x: FiniteFunction) -> Self:
        """ Lift a permutation of Hypergraphs into a permutation of OpenHypergraphs """
        # We have a morphism of cospans
        #
        #    s G t
        #     ↗ ↖
        #   A  ↓  B
        #    ↘ H ↙
        #    s'  t'
        #
        # with the ↓ arrow given by permutations w, x.
        #
        # So s' = s ; w
        #    t' = t ; w
        H = self.H.permute(w, x)
        return type(self)(self.s >> w, self.t >> w, H)

    ##############################
    # List operations

    @classmethod
    def tensor_operations(cls, x: FiniteFunction, a: IndexedCoproduct, b: IndexedCoproduct) -> 'OpenHypergraph':
        """ The N-fold tensoring of operations ``x``. Like 'singleton' but for many operations.

            x : N → Σ₁
            a : N → Σ₀*
            b : N → Σ₀*
        """
        if b.values.target != a.values.target or b.values.table.dtype != a.values.table.dtype:
            raise ValueError("a and b must have the same target and dtype")
        w = a.values.to_initial()

        if len(x) != len(a) or len(x) != len(b):
            raise ValueError("must have len(x) == len(a) == len(b)")

        s = cls.FiniteFunction().inj0(len(a.values), len(b.values))
        t = cls.FiniteFunction().inj1(len(a.values), len(b.values))
        H = cls.Hypergraph()(
            s = cls.IndexedCoproduct()(sources=a.sources, values=s),
            t = cls.IndexedCoproduct()(sources=b.sources, values=t),
            w = a.values + b.values,
            x = x)

        return cls(s, t, H)

    @classmethod
    def tensor_list(cls, ds: List['OpenHypergraph'], w=None, x=None) -> 'OpenHypergraph':
        if len(ds) == 0:
            return cls.unit(w, x)

        s = cls.FiniteFunction().tensor_list([d.s for d in ds])
        t = cls.FiniteFunction().tensor_list([d.t for d in ds])
        H = cls.Hypergraph().coproduct_list([d.H for d in ds])
        return cls(s, t, H)

class HasOpenHypergraph(HasHypergraph):
    @classmethod
    @abstractmethod
    def OpenHypergraph(cls) -> Type[OpenHypergraph]:
        ...

    @classmethod
    def Hypergraph(cls) -> Type[Hypergraph]:
        return cls.OpenHypergraph().Hypergraph()
