import sympy as sp
import pytest

from lyra_geometry import ConnectionTensor, IndexedTensor, Tensor, U, D


def test_connection_tensor_rejects_raise_or_lower(space_flat):
    """Connections should forbid raising or lowering indices."""
    gamma = space_flat.christoffel2
    assert isinstance(gamma, ConnectionTensor)
    with pytest.raises(ValueError):
        gamma(U, U, D)


def test_connection_tensor_supports_einstein_contraction(space_flat):
    """Connections should still support Einstein contraction."""
    a, b, n, l, m = space_flat.index("a b n l m")
    gamma = space_flat.christoffel2
    contracted = gamma[+b, -a, -n] * gamma[+l, -b, -m]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 4


def test_connection_scalar_multiplication(space_flat):
    phi = sp.symbols("phi")
    gamma = space_flat.gamma
    scaled_left = phi * gamma
    scaled_right = gamma * phi
    assert scaled_left.components == sp.Array(gamma.components) * phi
    assert scaled_right.components == scaled_left.components
    tensor_scaled = space_flat.phi * gamma
    assert tensor_scaled.components == sp.Array(gamma.components) * space_flat.phi.expr


def test_connection_requires_explicit_variance(space_flat):
    a, b, c = space_flat.index("a b c")
    gamma = space_flat.gamma
    with pytest.raises(TypeError):
        _ = gamma[a, b, c]
    indexed = gamma[+a, -b, -c]
    assert isinstance(indexed, IndexedTensor)
    assert indexed.signature == (U, D, D)
