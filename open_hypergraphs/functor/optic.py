from typing import Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

from open_hypergraphs.finite_function import FiniteFunction, IndexedCoproduct
from open_hypergraphs.open_hypergraph import OpenHypergraph
from open_hypergraphs.functor import Functor, FrobeniusFunctor

class Optic(FrobeniusFunctor, ABC):
    # Fwd maps
    F: FrobeniusFunctor

    # Reverse maps
    R: FrobeniusFunctor

    # Compute the object M for each operation x[i] : a[i] → b[i]
    @abstractmethod
    def residual(self, x: FiniteFunction, a: IndexedCoproduct, b: IndexedCoproduct) -> IndexedCoproduct:
        ...

    def map_objects(self, A: FiniteFunction) -> IndexedCoproduct:
        # Each object A is mapped to F(A) ● R(A)
        FA = self.F.map_objects(A)
        RA = self.R.map_objects(A)

        assert len(FA) == len(RA)
        n = len(FA)
        paired = FA + RA
        p = self.FiniteFunction().transpose(2, n)

        # TODO: Exposing internals of FiniteFunction here isn't nice.
        sources = self.FiniteFunction()(None, FA.sources.table + RA.sources.table)
        values = paired.map_indexes(p).values
        return self.IndexedCoproduct()(sources, values)

    def map_operations(self, x: FiniteFunction, A: IndexedCoproduct, B: IndexedCoproduct) -> OpenHypergraph:
        # F(x₀) ● F(x₁) ... F(xn)   :   FA₀ ● FA₁ ... FAn   →   (FB₀ ● M₀) ● (FB₁ ● M₁) ... (FBn ● Mn)
        fwd = self.F.map_operations(x, A, B)

        # R(x₀) ● R(x₁) ... R(xn)   :   (M₀ ● RB₀) ● (M₁ ● RB₁) ... (Mn ● RBn)   →   RA₀ ● RA₁ ... RAn
        rev = self.R.map_operations(x, A, B)

        cls = self.OpenHypergraph()

        # We'll need these types to build identities and interleavings
        FA = self.F.map_objects(A.values)
        FB = self.F.map_objects(B.values)
        RA = self.R.map_objects(A.values)
        RB = self.R.map_objects(B.values)
        M  = self.residual(x, A, B)

        # NOTE: we use flatmap here to ensure that each "block" of FB, which
        # might be e.g., F(B₀ ● B₁ ● ... ● Bn) is correctly interleaved:
        # consider that if M = I, then we would need to interleave
        fwd_interleave = self.interleave_blocks(B.flatmap(FB), M, x.to_initial()).dagger()
        rev_cointerleave = self.interleave_blocks(M, B.flatmap(RB), x.to_initial())

        assert fwd.target == fwd_interleave.source
        assert rev_cointerleave.target == rev.source

        i_FB = self.OpenHypergraph().identity(FB.values, x.to_initial())
        i_RB = self.OpenHypergraph().identity(RB.values, x.to_initial())

        # Make this diagram "c":
        #
        #       ┌────┐
        #       │    ├──────────────── FB
        # FA ───┤ Ff │  M
        #       │    ├───┐  ┌────┐
        #       └────┘   └──┤    │
        #                   │ Rf ├──── RA
        # RB ───────────────┤    │
        #                   └────┘
        lhs = (fwd >> fwd_interleave) @ i_RB
        rhs = i_FB @ (rev_cointerleave >> rev)
        c = lhs >> rhs

        # now adapt so that the wires labeled RB and RA are 'bent around'.
        d = partial_dagger(c, FA, FB, RA, RB)

        # finally interleave the FA with RA and FB with RB
        lhs = self.interleave_blocks(FA, RA, x.to_initial()).dagger()
        rhs = self.interleave_blocks(FB, RB, x.to_initial())
        return lhs >> d >> rhs
    

    def interleave_blocks(self, A: IndexedCoproduct, B: IndexedCoproduct, x: FiniteFunction) -> OpenHypergraph:
        """ An OpenHypergraph whose source is ``A+B`` and whose target is the 'interleaving'
        ``(A₀ + B₀) + (A₁ + B₁) + ... (An + Bn)`` """
        if len(A) != len(B):
            raise ValueError("Can't interleave types of unequal lengths")
        if len(x) != 0:
            raise ValueError(f"x must be initial, but {x.source=}")

        AB = A + B

        s = self.FiniteFunction().identity(len(AB.values))
        # NOTE: t is the dagger of transpose(2, N) because it appears in the target position!
        t = AB.sources.injections(self.FiniteFunction().transpose(2, len(A)))
        return self.OpenHypergraph().spider(s, t, AB.values, x)


    def adapt(self, c: OpenHypergraph, A: FiniteFunction, B: FiniteFunction):
        """ Given some optic ``c = Optic(c')`` where ``c' : A → B``,
        adapt ``c`` to have the type ``F(A) ● R(B) → F(B) ● R(A)``. """
        # we'll need these
        x = c.H.x.to_initial()
        FA = self.F.map_objects(A)
        FB = self.F.map_objects(B)
        RA = self.R.map_objects(A)
        RB = self.R.map_objects(B)

        # first, uninterleave to get d : FA●RA → FB●RB
        lhs = self.interleave_blocks(FA, RA, x)
        rhs = self.interleave_blocks(FB, RB, x).dagger()
        d = lhs >> c >> rhs

        # d is the uninterleaving of source/targets
        assert d.source == (FA + RA).values
        assert d.target == (FB + RB).values

        # now compute the partial daggering to obtain
        # d : FA●RB → FB●RA
        return partial_dagger(d, FA, FB, RB, RA)

# Bend around the A₁ and B₁ wires of a map like c:
#         ┌─────┐
# FA  ────┤     ├──── FB
#         │  c  │
# RB  ────┤     ├──── RA
#         └─────┘
#
# ... to get a map of type FA ● RA → FB ● RB
def partial_dagger(c: OpenHypergraph, FA: IndexedCoproduct, FB: IndexedCoproduct, RA: IndexedCoproduct, RB: IndexedCoproduct) -> OpenHypergraph:
    s_i = c.FiniteFunction().inj0(len(FA.values), len(RB.values)) >> c.s
    s_o = c.FiniteFunction().inj1(len(FB.values), len(RA.values)) >> c.t
    s = s_i + s_o

    t_i = c.FiniteFunction().inj0(len(FB.values), len(RA.values)) >> c.t
    t_o = c.FiniteFunction().inj1(len(FA.values), len(RB.values)) >> c.s
    t = t_i + t_o

    return type(c)(s, t, c.H)
