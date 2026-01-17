import sympy as sp

from lyra_geometry import euler_density, kretschmann_scalar, ricci_scalar


def test_ricci_scalar_zero_flat(space_flat):
    scalar = ricci_scalar(space_flat)
    assert scalar.rank == 0
    assert sp.simplify(scalar.components[()]) == 0


def test_kretschmann_scalar_zero_flat(space_flat):
    kret = kretschmann_scalar(space_flat)
    assert kret.rank == 0
    assert sp.simplify(kret.components[()]) == 0


def test_euler_density_zero_flat(space_flat):
    euler = euler_density(space_flat)
    assert euler.rank == 0
    assert sp.simplify(euler.components[()]) == 0
