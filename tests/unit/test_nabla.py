import sympy as sp


def test_nabla_of_scalar_equals_partial_derivative(space_flat, coords):
    """Check covariant derivative of a scalar matches the partial derivative."""
    f = sp.Function("f")(*coords)
    scalar = space_flat.scalar(f)
    nabla = scalar.nabla()
    assert nabla.rank == 1
    assert nabla.components[0] == sp.diff(f, coords[0])
    assert nabla.components[1] == sp.diff(f, coords[1])
