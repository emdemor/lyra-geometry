import pytest
import sympy as sp

import lyra_geometry as pl
from lyra_geometry import Tensor, U, D


def test_single_tensor_repeated_labels_contracts_to_scalar(space_flat):
    """Contract a single tensor when the same label appears in up/down positions."""
    a = space_flat.index("a")
    t = space_flat.generic("T", (U, D))
    contracted = t[+a, -a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 0


def test_tensor_tensor_multiplication_contracts_repeated_labels(space_flat):
    """Multiply two indexed tensors and contract matching labels."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    contracted = t[+a, -b] * s[+c, -a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 2
    assert contracted.signature == (D, U)


def test_indexed_tensor_reindex_and_contract(space_flat):
    """Reindex an IndexedTensor and contract with raised indices."""
    a, b = space_flat.index("a b")
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(D, D))
    f = t[-a, -b]
    contracted = f[-a, -b] * f[+a, +b]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 0
    assert contracted.components[()] == 30


def test_tensor_connection_multiplication_contracts_repeated_labels(space_flat):
    """Multiply tensor by connection and contract repeated labels."""
    a, b, c, d = space_flat.index("a b c d")
    t = space_flat.generic("T", (U, D))
    gamma = space_flat.christoffel2
    contracted = gamma[+a, -b, -c] * t[+d, -a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 3


def test_number_times_tensor_scales_components(space_flat):
    """Multiply a tensor by a number and scale its components."""
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(U, D))
    scaled = 3 * t
    assert isinstance(scaled, Tensor)
    assert scaled.components[0, 0] == 3


def test_number_times_connection_returns_tensor(space_flat):
    """Multiply a connection by a number and keep tensor-like behavior."""
    gamma = space_flat.christoffel2
    scaled = 2 * gamma
    assert isinstance(scaled, Tensor)
    assert scaled.rank == 3


def test_sympy_expression_times_tensor_returns_tensor(space_flat, coords):
    """Multiply a tensor by a SymPy expression."""
    t = space_flat.generic("T", (U, D))
    expr = sp.sin(coords[0])
    scaled = expr * t
    assert isinstance(scaled, Tensor)
    assert scaled.rank == 2


def test_sympy_expression_times_connection_returns_tensor(space_flat, coords):
    """Multiply a connection by a SymPy expression."""
    gamma = space_flat.christoffel2
    expr = sp.exp(coords[1])
    scaled = expr * gamma
    assert isinstance(scaled, Tensor)
    assert scaled.rank == 3


def test_number_times_indexed_tensor_then_connection_contracts(space_flat):
    """Mix number, indexed tensor, and connection with a contraction."""
    a, b, c, d = space_flat.index("a b c d")
    t = space_flat.generic("T", (U, D))
    gamma = space_flat.christoffel2
    mixed = 2 * t[+a, -b] * gamma[+c, -a, -d]
    assert isinstance(mixed, Tensor)
    assert mixed.rank == 3


def test_sympy_tensor_connection_chain_contracts_multiple_labels(space_flat, coords):
    """Chain SymPy, tensor, and connection with multiple contractions."""
    a, b, c, d, e = space_flat.index("a b c d e")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    gamma = space_flat.christoffel2
    expr = sp.cos(coords[0])
    mixed = expr * t[+a, -b] * s[+c, -a] * gamma[+d, -c, -e]
    assert isinstance(mixed, Tensor)
    assert mixed.rank == 3


def test_contract_raises_on_triplicate_label(space_flat):
    """Reject contractions when a label appears more than twice."""
    a, b, c, d = space_flat.index("a b c d")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    u = space_flat.generic("U", (U, D))
    with pytest.raises(ValueError):
        _ = space_flat.contract(t[+a, -b], s[+c, -a], u[+d, -a])


def test_contract_raises_on_same_variance_labels(space_flat):
    """Reject contractions with the same variance across tensors."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.generic("T", (U, U))
    s = space_flat.generic("S", (U, U))
    with pytest.raises(ValueError):
        _ = t[+a, +b] * s[+c, +a]

def test_index_order():
    """Compare terms using a matching component index order."""
    x, y = sp.symbols("x y", real=True)
    a = sp.Function("a")

    st = pl.SpaceTime(
        coords=(x, y),
        metric=sp.diag(y * a(x) ** 2, -a(x) ** 2),
    )

    _, a, b, g, d, e, m, n, l, s, h, k = st.index(
        "empty alpha beta gamma delta epsilon mu nu lambda sigma eta kappa"
    )

    st.set_scale(sp.Function("phi")(x))
    st.update()

    gamma = st.christoffel2
    term_01 = st.tensor(
        gamma[+s, -a, -n] * gamma[+l, -s, -m] - gamma[+s, -a, -m] * gamma[+l, -s, -n],
        index=(-a, -n, +l, -m),
    )

    def term_2_element(a_idx, n_idx, l_idx, m_idx):
        b_idx, g_idx, d_idx = st.index("beta gamma delta")
        gmm = gamma[+b_idx, -g_idx, -d_idx]
        return (
            sum(gmm(s, a_idx, n_idx) * gmm(l_idx, s, m_idx) for s in range(st.dim))
            - sum(gmm(s, a_idx, m_idx) * gmm(l_idx, s, n_idx) for s in range(st.dim))
        )

    term_02 = st.from_function(
        term_2_element, signature=(pl.D, pl.D, pl.U, pl.D), name="RiemannianCurv", label="RR"
    )

    assert term_01[-a, -n, +l, -m](0, 0) == term_02[-a, -n, +l, -m](0, 0)
    assert term_01[-a, -n, +l, -m](0, 1) == term_02[-a, -n, +l, -m](0, 1)
    assert term_01[-a, -n, +l, -m](1, 0) == term_02[-a, -n, +l, -m](1, 0)
    assert term_01[-a, -n, +l, -m](1, 1) == term_02[-a, -n, +l, -m](1, 1)
