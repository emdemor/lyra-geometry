import pytest
import sympy as sp

from lyra_geometry import Tensor, TensorSpace, U, D


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


def test_subtraction_aligns_labels_before_combining(space_flat):
    """Align labels so subtraction is invariant to index ordering."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(U, D))
    s = space_flat.from_array([[5, 6], [7, 8]], signature=(U, D))
    expr1 = t[a, b] * s[c, a]
    expr2 = t[a, c] * s[b, a]
    diff = expr1 - expr2
    expected = sp.ImmutableDenseNDimArray(
        [
            sum(t.components[i, j] * s.components[k, i] for i in range(space_flat.dim))
            - sum(t.components[i, k] * s.components[j, i] for i in range(space_flat.dim))
            for j in range(space_flat.dim)
            for k in range(space_flat.dim)
        ],
        (space_flat.dim, space_flat.dim),
    )
    assert diff.components == expected


def test_contracted_connection_terms_follow_label_order(space_flat):
    """Compare contracted terms after aligning axes explicitly."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    a_func = sp.Function("a")
    st = TensorSpace(
        coords=(t, x, y, z),
        metric=sp.diag(1, -a_func(t) ** 2, -a_func(t) ** 2, -a_func(t) ** 2),
    )
    _, a, b, g, d, e, m, n, l, s, h, k = st.index(
        "empty alpha beta gamma delta epsilon mu nu lambda sigma eta kappa"
    )
    st.set_scale(sp.Function("phi")(t))
    st.update()
    gamma = st.christoffel2
    expr1 = gamma[s, a, n] * gamma[l, s, m] - gamma[s, a, m] * gamma[l, s, n]

    def teste(l_idx, a_idx, m_idx, n_idx):
        gamma_local = st.christoffel2
        return (
            sum(gamma_local[s, a_idx, n_idx] * gamma_local[l_idx, s, m_idx] for s in range(st.dim))
            - sum(gamma_local[s, a_idx, m_idx] * gamma_local[l_idx, s, n_idx] for s in range(st.dim))
        )

    expr2 = st.from_function(teste, signature=(U, D, D, D), name="RiemannianCurv", label="RR")
    perm = [2, 0, 3, 1]  # (a, n, l, m) -> (l, a, m, n)
    permuted = sp.permutedims(expr1.components, perm)
    diff = sp.simplify(permuted - expr2.components)
    assert diff == sp.ImmutableDenseNDimArray([0] * (st.dim**4), (st.dim,) * 4)


def test_tensor_reorders_axes_by_explicit_index(space_flat):
    """Reorder tensor axes to a caller-specified index order."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(U, D))
    s = space_flat.from_array([[5, 6], [7, 8]], signature=(U, D))
    expr = t[a, b] * s[c, a]
    reordered = TensorSpace.tensor(space_flat, expr, index=(+c, -b))
    expected = sp.permutedims(expr.components, (1, 0))
    assert reordered.components == expected


def test_tensor_factory_call_reorders_axes(space_flat):
    """Allow st.tensor(...) to reorder axes via TensorFactory call."""
    a, b, c = space_flat.index("a b c")
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(U, D))
    s = space_flat.from_array([[5, 6], [7, 8]], signature=(U, D))
    expr = t[a, b] * s[c, a]
    reordered = TensorSpace.tensor(space_flat, expr, index=(+c, -b))
    via_factory = space_flat.tensor(expr, index=(+c, -b))
    assert via_factory.components == reordered.components


def test_indexed_tensor_addition_aligns_labels(space_flat):
    """Align labels before adding indexed tensors with permuted axes."""
    a, b = space_flat.index("a b")
    t = space_flat.from_array([[1, 2], [3, 4]], signature=(U, D))
    sum_ab = t[a, b] + t[b, a]
    expected = t.components + sp.permutedims(t.components, (1, 0))
    assert sum_ab.components == expected


def test_indexed_tensor_partial_derivative_uses_coord_index(space_flat):
    """Differentiate using coordinate-bound indices."""
    x, y = space_flat.coords
    m, n = space_flat.coord_index("m n")
    a, b = space_flat.index("a b")
    t = space_flat.generic("T", (U, D))
    dt = t[a, b].d(-m)
    assert dt.components[0, 0, 0] == sp.diff(t.components[0, 0], x)
    assert dt.signature == (U, D, D)
