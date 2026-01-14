import pytest
import sympy as sp

from lyra_geometry import ConnectionTensor, Tensor, TensorSpace, U, D


def test_basic_tensor_space():
    x, y = sp.symbols("x y")
    space = TensorSpace((x, y))
    t = space.generic("T", (U, D))
    assert t.rank == 2


def test_index_uses_tensor_signature_for_variance():
    x, y = sp.symbols("x y")
    space = TensorSpace((x, y))
    a, b, c = space.index("a b c")
    t = space.generic("T", (U, D))
    indexed = t[a, b]
    assert indexed.signature == (U, D)
    contracted = t[a, b] * t[c, a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 2


def test_connection_tensor_disallows_raise_lower_and_contracts():
    x, y = sp.symbols("x y")
    space = TensorSpace((x, y), metric=[[1, 0], [0, 1]])
    a, b, n, l, m = space.index("a b n l m")
    gamma = space.christoffel2
    assert isinstance(gamma, ConnectionTensor)
    with pytest.raises(ValueError):
        gamma(U, U, D)
    contracted = gamma[b, a, n] * gamma[l, b, m]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 4
