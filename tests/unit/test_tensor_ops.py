import pytest
import sympy as sp

from lyra_geometry import Tensor, U, D


def test_index_infers_variance_from_tensor_signature(space_flat):
    """Ensure Index labels inherit variance from the tensor signature."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.generic("T", (U, D))
    indexed = t[a, b]
    assert indexed.signature == (U, D)
    contracted = t[a, b] * t[c, a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 2


def test_einstein_contraction_on_repeated_labels(space_flat):
    """Contract repeated labels across tensors into a lower-rank result."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.generic("T", (U, D))
    s = space_flat.generic("S", (U, D))
    contracted = t[a, b] * s[c, a]
    assert isinstance(contracted, Tensor)
    assert contracted.rank == 2


def test_raise_index_uses_inverse_metric_in_1d(space_1d):
    """Verify raising an index in 1D divides by the metric component."""
    v = sp.symbols("v")
    t = space_1d.from_array([v], signature=(D,))
    raised = t.as_signature((U,))
    assert raised[0] == v / space_1d.metric.components[0, 0]


def test_contract_same_variance_without_metric_raises(space_flat):
    """Reject contraction of same-variance indices without a metric."""
    t = space_flat.generic("T", (U, U))
    with pytest.raises(ValueError):
        t.contract(0, 1, use_metric=False)


def test_subtraction_of_contracted_tensors_returns_tensor(space_flat):
    """Support subtraction of tensors with matching signature and space."""
    a, s, l, m, n = space_flat.index("a s l m n")
    gamma = space_flat.christoffel2
    expr = gamma[s, a, n] * gamma[l, s, m] - gamma[s, a, m] * gamma[l, s, n]
    assert isinstance(expr, Tensor)
    assert expr.rank == 4
