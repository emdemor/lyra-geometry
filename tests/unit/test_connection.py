import pytest

from lyra_geometry import ConnectionTensor, Tensor, U, D


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
    contracted = gamma[b, a, n] * gamma[l, b, m]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 4
