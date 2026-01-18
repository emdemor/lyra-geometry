import pytest
import sympy as sp

from lyra_geometry import TensorSpace

pytestmark = pytest.mark.slow


def test_schwarzschild_ricci_scalar_zero():
    t, r, theta, phi = sp.symbols("t r theta phi")
    mass = sp.symbols("M")
    f = 1 - 2 * mass / r
    metric = [
        [-f, 0, 0, 0],
        [0, 1 / f, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta) ** 2],
    ]
    space = TensorSpace((t, r, theta, phi), metric=metric)
    scalar = space.ricci_scalar()
    assert sp.simplify(scalar.components[()]) == 0


def test_de_sitter_ricci_scalar_constant():
    t, x, y, z = sp.symbols("t x y z")
    hubble = sp.symbols("H")
    scale = sp.exp(hubble * t)
    metric = [
        [-1, 0, 0, 0],
        [0, scale**2, 0, 0],
        [0, 0, scale**2, 0],
        [0, 0, 0, scale**2],
    ]
    space = TensorSpace((t, x, y, z), metric=metric)
    scalar = space.ricci_scalar()
    assert sp.simplify(scalar.components[()] + 12 * hubble**2) == 0
