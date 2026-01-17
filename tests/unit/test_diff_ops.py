import sympy as sp

from lyra_geometry import D, U


def test_gradient_of_scalar_matches_partial(space_flat, coords):
    f = sp.Function("f")(*coords)
    scalar = space_flat.scalar(f)
    grad = space_flat.gradient(scalar)
    assert grad.rank == 1
    assert grad.signature == (D,)
    assert grad.components[0] == sp.diff(f, coords[0])
    assert grad.components[1] == sp.diff(f, coords[1])


def test_divergence_of_vector_matches_partial(space_flat, coords):
    x, y = coords
    v = space_flat.from_array([x**2, y], signature=(U,))
    div = space_flat.divergence(v)
    expected = sp.diff(x**2, x) + sp.diff(y, y)
    assert div.rank == 0
    assert sp.simplify(div.components[()] - expected) == 0


def test_laplacian_of_scalar_matches_trace(space_flat, coords):
    x, y = coords
    scalar = space_flat.scalar(x**2 + y**2)
    lap = space_flat.laplacian(scalar)
    expected = sp.diff(x**2 + y**2, x, x) + sp.diff(x**2 + y**2, y, y)
    assert lap.rank == 0
    assert sp.simplify(lap.components[()] - expected) == 0
