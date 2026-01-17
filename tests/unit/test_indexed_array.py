import pytest

from lyra_geometry import IndexedArray, Tensor, TensorSpace, U, D


def test_indexed_array_variance_change_keeps_components(space_flat):
    a, b, c = space_flat.index("a b c")
    gamma = space_flat.christoffel2
    assert gamma[+a, -b, -c] == gamma[-a, +b, -c]
    assert gamma[+a, -b, -c] == gamma[-a, -b, +c]


def test_indexed_array_contracts_repeated_up_down(space_flat):
    a, c = space_flat.index("a c")
    gamma = space_flat.christoffel2
    contracted = gamma[+a, -a, -c]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 1


def test_indexed_array_rejects_repeated_same_variance(space_flat):
    a, c = space_flat.index("a c")
    gamma = space_flat.christoffel2
    with pytest.raises(ValueError):
        _ = gamma[+a, +a, -c]


def test_indexed_array_contraction_with_tensor_returns_tensor(space_flat):
    a, b, c = space_flat.index("a b c")
    gamma = space_flat.christoffel2
    t = space_flat.generic("T", (U, U))
    contracted = gamma[+a, -b, -c] * t[+b, +c]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 1
    assert contracted.signature == (U,)


def test_kronecker_delta_is_indexed_array(space_flat):
    delta = space_flat.delta
    assert isinstance(delta, IndexedArray)
    assert delta[0, 0] == 1
    assert delta[0, 1] == 0


def test_levi_civita_symbol_is_indexed_array(space_flat):
    epsilon = space_flat.levi_civita
    assert isinstance(epsilon, IndexedArray)
    assert epsilon[0, 1] == 1
    assert epsilon[1, 0] == -1
    assert epsilon[0, 0] == 0


def test_indexed_array_init_without_metric():
    x, y = (1, 2)
    st = TensorSpace(coords=(x, y))
    assert isinstance(st.delta, IndexedArray)
    assert isinstance(st.levi_civita, IndexedArray)
