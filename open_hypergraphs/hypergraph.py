from dataclasses import dataclass
from open_hypergraphs.finite_function import AbstractFiniteFunction, AbstractIndexedCoproduct

@dataclass
class AbstractHypergraph:
    s: AbstractIndexedCoproduct # sources : Σ_{x ∈ X} τ₀(e) → X
    t: AbstractIndexedCoproduct # targets : Σ_{x ∈ X} τ₀(e) → X
    w: AbstractFiniteFunction # w : X → Σ₀
    x: AbstractFiniteFunction # x : X → Σ₁

    # number of vertices
    @property
    def W(self):
        return self.w.source

    # number of edges
    @property
    def X(self):
        return self.x.source

    def __post_init__(self):
        assert self.s.target == self.W
        assert self.t.target == self.W

        assert type(self.s) == self._Fun.IndexedCoproduct
        assert type(self.t) == self._Fun.IndexedCoproduct

        assert len(self.s.sources) == self.X
        assert len(self.t.sources) == self.X

    @classmethod
    def empty(cls, w, x):
        e = cls._Fun.initial(0)
        return cls(e, e, w, x)

    @classmethod
    def discrete(cls, w, x):
        W = w.source
        X = x.source

        assert X == 0 # we need this for Σ₁, but it must be initial.

        return cls(
            s = cls._Fun.IndexedCoproduct.initial(W),
            t = cls._Fun.IndexedCoproduct.initial(W),
            w = w,
            x = x)

    def coproduct(f, g):
        assert f.w.target == g.w.target
        assert f.x.target == g.x.target
        return type(f)(f.s @ g.s, f.t @ g.t, f.w + g.w, f.x + g.x)

    def __matmul__(f, g):
        return f.coproduct(g)

    def coequalize_vertices(self, q: AbstractFiniteFunction):
        assert q.source == self.W
        u = universal(q, self.w)
        if not (q >> u) == self.w:
            raise ValueError(f"Universal morphism doesn't make {q};{u}, {self.w} commute. Is q really a coequalizer?")

        return type(self)(self.s >> q, self.t >> q, u, self.x)

def universal(q: AbstractFiniteFunction, f: AbstractFiniteFunction):
    """
    Given a coequalizer q : B → Q of morphisms a, b : A → B
    and some f : B → B' such that f(a) = f(b),
    Compute the universal map u : Q → B'
    such that q ; u = f.
    """
    target = f.target
    u = q._Array.zeros(q.target, dtype=f.table.dtype)
    # TODO: in the below we assume the PRAM CRCW model: multiple writes to the
    # same memory location in the 'u' array can happen in parallel, with an
    # arbitrary write succeeding.
    # Note that this doesn't affect correctness because q and f are co-forks,
    # and q is a coequalizer.
    # However, this won't perform well on e.g., GPU hardware. FIXME!
    u[q.table] = f.table
    return type(f)(target, u)
