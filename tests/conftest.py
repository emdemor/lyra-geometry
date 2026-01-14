import sympy as sp
import pytest

from lyra_geometry import TensorSpace


@pytest.fixture
def coords():
    return sp.symbols("x y")


@pytest.fixture
def space_flat(coords):
    return TensorSpace(coords, metric=[[1, 0], [0, 1]])


@pytest.fixture
def space_1d():
    x = sp.symbols("x")
    a = sp.symbols("a")
    return TensorSpace((x,), metric=[[a]])
