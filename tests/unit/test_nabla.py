import pytest
import sympy as sp


def test_nabla_of_scalar_equals_partial_derivative(space_flat, coords):
    """Check covariant derivative of a scalar matches the partial derivative."""
    f = sp.Function("f")(*coords)
    scalar = space_flat.scalar(f)
    nabla = scalar.nabla()
    assert nabla.rank == 1
    assert nabla.components[0] == sp.diff(f, coords[0])
    assert nabla.components[1] == sp.diff(f, coords[1])


def test_nabla_order_matches_repeated_application(space_flat, coords):
    f = sp.Function("f")(*coords)
    scalar = space_flat.scalar(f)
    nabla_twice = space_flat.nabla(scalar, order=2)
    nabla_nested = space_flat.nabla(space_flat.nabla(scalar))
    assert nabla_twice.components == nabla_nested.components


def test_nabla_rejects_non_positive_order(space_flat, coords):
    f = sp.Function("f")(*coords)
    scalar = space_flat.scalar(f)
    with pytest.raises(ValueError):
        space_flat.nabla(scalar, order=0)
    with pytest.raises(ValueError):
        space_flat.nabla(scalar, order=-1)
