import pytest
import sympy as sp

from lyra_geometry import TensorSpace

pytestmark = pytest.mark.slow


t, r, theta, phi = sp.symbols("t r theta phi")
alpha = sp.Function("alpha")
coords = (t, r, theta, phi)


def _spherical_space(alpha_expr=None):
    A = alpha(r) if alpha_expr is None else alpha_expr
    metric = [
        [A, 0, 0, 0],
        [0, -1 / A, 0, 0],
        [0, 0, -r**2, 0],
        [0, 0, 0, -r**2 * sp.sin(theta) ** 2],
    ]
    return TensorSpace(coords, metric=metric), A


def _assert_expr_equal(actual, expected):
    diff = sp.simplify(sp.trigsimp(actual - expected))
    assert diff == 0


A = alpha(r)
A_r = sp.diff(A, r)
A_rr = sp.diff(A, r, 2)


CONNECTION_COMPONENTS = [
    ((0, 0, 1), A_r / (2 * A)),
    ((0, 1, 0), A_r / (2 * A)),
    ((1, 0, 0), A * A_r / 2),
    ((1, 1, 1), -A_r / (2 * A)),
    ((1, 2, 2), -r * A),
    ((1, 3, 3), -r * A * sp.sin(theta) ** 2),
    ((2, 1, 2), 1 / r),
    ((2, 2, 1), 1 / r),
    ((2, 3, 3), -sp.sin(2 * theta) / 2),
    ((3, 1, 3), 1 / r),
    ((3, 2, 3), sp.cot(theta)),
    ((3, 3, 1), 1 / r),
    ((3, 3, 2), sp.cot(theta)),
]


RIEMANN_COMPONENTS = [
    ((0, 1, 0, 1), -A_rr / (2 * A)),
    ((0, 1, 1, 0), A_rr / (2 * A)),
    ((0, 2, 0, 2), -r * A_r / 2),
    ((0, 2, 2, 0), r * A_r / 2),
    ((0, 3, 0, 3), -r * sp.sin(theta) ** 2 * A_r / 2),
    ((0, 3, 3, 0), r * sp.sin(theta) ** 2 * A_r / 2),
    ((1, 0, 0, 1), -A * A_rr / 2),
    ((1, 0, 1, 0), A * A_rr / 2),
    ((1, 2, 1, 2), -r * A_r / 2),
    ((1, 2, 2, 1), r * A_r / 2),
    ((1, 3, 1, 3), -r * sp.sin(theta) ** 2 * A_r / 2),
    ((1, 3, 3, 1), r * sp.sin(theta) ** 2 * A_r / 2),
    ((2, 0, 0, 2), -A * A_r / (2 * r)),
    ((2, 0, 2, 0), A * A_r / (2 * r)),
    ((2, 1, 1, 2), A_r / (2 * r * A)),
    ((2, 1, 2, 1), -A_r / (2 * r * A)),
    ((2, 3, 2, 3), (1 - A) * sp.sin(theta) ** 2),
    ((2, 3, 3, 2), (A - 1) * sp.sin(theta) ** 2),
    ((3, 0, 0, 3), -A * A_r / (2 * r)),
    ((3, 0, 3, 0), A * A_r / (2 * r)),
    ((3, 1, 1, 3), A_r / (2 * r * A)),
    ((3, 1, 3, 1), -A_r / (2 * r * A)),
    ((3, 2, 2, 3), A - 1),
    ((3, 2, 3, 2), 1 - A),
]


RICCI_COMPONENTS = [
    ((0, 0), -(r * A_rr + 2 * A_r) * A / (2 * r)),
    ((1, 1), (r * A_rr / 2 + A_r) / (r * A)),
    ((2, 2), r * A_r + A - 1),
    ((3, 3), (r * A_r + A - 1) * sp.sin(theta) ** 2),
]


@pytest.mark.parametrize("indices, expected", CONNECTION_COMPONENTS)
def test_spherical_connection_components(indices, expected):
    space, _ = _spherical_space()
    actual = space.connection.components[indices]
    _assert_expr_equal(actual, expected)


@pytest.mark.parametrize("indices, expected", RIEMANN_COMPONENTS)
def test_spherical_riemann_components(indices, expected):
    space, _ = _spherical_space()
    actual = space.riemann.components[indices]
    _assert_expr_equal(actual, expected)


@pytest.mark.parametrize("indices, expected", RICCI_COMPONENTS)
def test_spherical_ricci_components(indices, expected):
    space, _ = _spherical_space()
    actual = space.ricci.components[indices]
    _assert_expr_equal(actual, expected)


def _geodesic_expected_lhs(param):
    t_fun = sp.Function("t")(param)
    r_fun = sp.Function("r")(param)
    theta_fun = sp.Function("theta")(param)
    phi_fun = sp.Function("phi")(param)

    A_sub = A.subs(r, r_fun)
    A_r_sub = A_r.subs(r, r_fun)

    dt = sp.diff(t_fun, param)
    dr = sp.diff(r_fun, param)
    dtheta = sp.diff(theta_fun, param)
    dphi = sp.diff(phi_fun, param)

    d2t = sp.diff(t_fun, param, 2)
    d2r = sp.diff(r_fun, param, 2)
    d2theta = sp.diff(theta_fun, param, 2)
    d2phi = sp.diff(phi_fun, param, 2)

    lhs_t = d2t + (A_r_sub / A_sub) * dt * dr
    lhs_r = (
        d2r
        + (A_sub * A_r_sub / 2) * dt**2
        + (-A_r_sub / (2 * A_sub)) * dr**2
        - r_fun * A_sub * dtheta**2
        - r_fun * A_sub * sp.sin(theta_fun) ** 2 * dphi**2
    )
    lhs_theta = d2theta + 2 * dr * dtheta / r_fun - sp.sin(theta_fun) * sp.cos(theta_fun) * dphi**2
    lhs_phi = d2phi + 2 * dr * dphi / r_fun + 2 * sp.cot(theta_fun) * dtheta * dphi

    return [lhs_t, lhs_r, lhs_theta, lhs_phi]


def test_spherical_geodesic_equation_t():
    space, _ = _spherical_space()
    equations = space.geodesic_equations(parameter="tau")
    expected = _geodesic_expected_lhs(sp.Symbol("tau"))[0]
    _assert_expr_equal(equations[0].lhs, expected)


def test_spherical_geodesic_equation_r():
    space, _ = _spherical_space()
    equations = space.geodesic_equations(parameter="tau")
    expected = _geodesic_expected_lhs(sp.Symbol("tau"))[1]
    _assert_expr_equal(equations[1].lhs, expected)


def test_spherical_geodesic_equation_theta():
    space, _ = _spherical_space()
    equations = space.geodesic_equations(parameter="tau")
    expected = _geodesic_expected_lhs(sp.Symbol("tau"))[2]
    _assert_expr_equal(equations[2].lhs, expected)


def test_spherical_geodesic_equation_phi():
    space, _ = _spherical_space()
    equations = space.geodesic_equations(parameter="tau")
    expected = _geodesic_expected_lhs(sp.Symbol("tau"))[3]
    _assert_expr_equal(equations[3].lhs, expected)


def test_spherical_einstein_tt_vacuum():
    mass = sp.symbols("M")
    space, _ = _spherical_space(alpha_expr=1 - 2 * mass / r)
    _assert_expr_equal(space.einstein.components[0, 0], 0)


def test_spherical_einstein_rr_vacuum():
    mass = sp.symbols("M")
    space, _ = _spherical_space(alpha_expr=1 - 2 * mass / r)
    _assert_expr_equal(space.einstein.components[1, 1], 0)


def test_spherical_einstein_theta_theta_vacuum():
    mass = sp.symbols("M")
    space, _ = _spherical_space(alpha_expr=1 - 2 * mass / r)
    _assert_expr_equal(space.einstein.components[2, 2], 0)


def test_spherical_einstein_phi_phi_vacuum():
    mass = sp.symbols("M")
    space, _ = _spherical_space(alpha_expr=1 - 2 * mass / r)
    _assert_expr_equal(space.einstein.components[3, 3], 0)
