import sympy as sp


def test_tensor_space_dimension_matches_coordinate_count(space_flat, coords):
    """Confirm TensorSpace dimension matches the coordinate count."""
    assert space_flat.dim == len(coords)


def test_metric_inverse_tensor_is_computed_and_registered(space_flat):
    """Ensure the inverse metric tensor is computed and registered."""
    assert space_flat.metric_inv_tensor is not None
    assert space_flat.metric_inv_tensor.components == sp.Array(sp.eye(2))
