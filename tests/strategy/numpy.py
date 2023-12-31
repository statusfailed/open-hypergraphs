import numpy as np
from hypothesis import strategies as st

# NOTE: Hypothesis seems to have a bug where zero-length arrays trigger an
# error, so we'll just use numpy's random module instead.
@st.composite
def arrays(draw, n, high, dtype):
    # dummy call to draw to silence hypothesis warning
    _ = draw(st.integers(min_value=0, max_value=0))
    return np.random.randint(0, high=high, size=(n,), dtype=dtype)

@st.composite
def permutations(draw, n, dtype):
    # dummy call to draw to silence hypothesis warning
    _ = draw(st.integers(min_value=0, max_value=0))
    return np.random.permutation(n).astype(dtype) # TODO: a bit dodgy when n overflows!
