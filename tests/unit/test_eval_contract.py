from lyra_geometry import Tensor, U, D


def test_eval_contract_parses_index_notation_into_contraction(space_flat):
    """Parse contracted indices in eval_contract and return a tensor."""
    space_flat.generic("A", (U, D))
    space_flat.generic("B", (U, D))
    result = space_flat.eval_contract("A^a_b B^b_c")
    assert isinstance(result, Tensor)
    assert result.signature == (U, D)
