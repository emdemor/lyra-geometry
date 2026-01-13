import sympy as sp

from pylyra import TensorSpace, U, D


def test_basic_tensor_space():
    x, y = sp.symbols("x y")
    space = TensorSpace((x, y))
    t = space.generic("T", (U, D))
    assert t.rank == 2
