import sympy as sp

from lyra_geometry import Tensor, U, D


def test_single_tensor_repeated_labels_contracts_to_scalar(space_flat):
    """Contract a single tensor when the same label appears in up/down positions."""
    a = space_flat.index("a")
    t = space_flat.generic("T", (U, D))
    contracted = t[a, a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 0


def test_tensor_tensor_multiplication_contracts_repeated_labels(space_flat):
    """Multiply two indexed tensors and contract matching labels."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    contracted = t[a, b] * s[c, a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 2
    assert contracted.signature == (D, U)


def test_tensor_connection_multiplication_contracts_repeated_labels(space_flat):
    """Multiply tensor by connection and contract repeated labels."""
    a, b, c, d = space_flat.index("a b c d")
    t = space_flat.generic("T", (U, D))
    gamma = space_flat.christoffel2
    contracted = gamma[a, b, c] * t[d, a]
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
    mixed = 2 * t[a, b] * gamma[c, a, d]
    assert isinstance(mixed, Tensor)
    assert mixed.rank == 3


def test_sympy_tensor_connection_chain_contracts_multiple_labels(space_flat, coords):
    """Chain SymPy, tensor, and connection with multiple contractions."""
    a, b, c, d, e = space_flat.index("a b c d e")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    gamma = space_flat.christoffel2
    expr = sp.cos(coords[0])
    mixed = expr * t[a, b] * s[c, a] * gamma[d, c, e]
    assert isinstance(mixed, Tensor)
    assert mixed.rank == 3
