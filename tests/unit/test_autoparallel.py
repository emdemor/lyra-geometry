import sympy as sp

from lyra_geometry import TensorSpace, autoparallel_equations


def test_autoparallel_equations_minkowski_timelike():
    t, x, y, z = sp.symbols("t x y z")
    metric = sp.diag(1, -1, -1, -1)
    space = TensorSpace(coords=(t, x, y, z), metric=metric)
    equations = space.autoparallel_equations(parameter="timelike")
    assert len(equations) == 4

    tau = sp.Symbol("tau")
    coord_funcs = [sp.Function(str(c))(tau) for c in (t, x, y, z)]
    for eq, func in zip(equations, coord_funcs):
        assert eq.lhs == sp.diff(func, tau, 2)
        assert eq.rhs == 0


def test_autoparallel_equations_minkowski_null_lambda():
    t, x, y, z = sp.symbols("t x y z")
    metric = sp.diag(1, -1, -1, -1)
    equations = autoparallel_equations(metric, (t, x, y, z), parameter="null")
    assert len(equations) == 4

    lam = sp.Symbol("lambda")
    coord_funcs = [sp.Function(str(c))(lam) for c in (t, x, y, z)]
    for eq, func in zip(equations, coord_funcs):
        assert eq.lhs == sp.diff(func, lam, 2)
        assert eq.rhs == 0
