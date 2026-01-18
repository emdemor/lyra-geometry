import sympy as sp

from lyra_geometry import TensorSpace


def test_riemann_convention_flips_sign():
    theta, phi = sp.symbols("theta phi", real=True)
    metric = sp.diag(1, sp.sin(theta) ** 2)

    space_mtw = TensorSpace(coords=(theta, phi), metric=metric, riemann_convention="mtw")
    space_ll = TensorSpace(coords=(theta, phi), metric=metric, riemann_convention="landau-lifshitz")

    riem_mtw = space_mtw.riemann
    riem_ll = space_ll.riemann
    ricci_mtw = space_mtw.ricci
    ricci_ll = space_ll.ricci
    scalar_mtw = space_mtw.scalar_curvature.components[()]
    scalar_ll = space_ll.scalar_curvature.components[()]

    component_mtw = riem_mtw.comp[0, 1, 0, 1]
    component_ll = riem_ll.comp[0, 1, 0, 1]

    assert sp.simplify(component_mtw) != 0
    assert sp.simplify(component_mtw + component_ll) == 0
    assert sp.simplify(ricci_mtw.comp[0, 0] + ricci_ll.comp[0, 0]) == 0
    assert sp.simplify(ricci_mtw.comp[1, 1] + ricci_ll.comp[1, 1]) == 0
    assert sp.simplify(scalar_mtw + scalar_ll) == 0
