import sympy as sp


def test_nabla_phi_shortcuts_match_nabla(space_flat, coords):
    phi = sp.Function("phi")(coords[0])
    space_flat.set_scale(phi)
    space_flat.update()

    nabla_phi = space_flat.nabla_phi
    expected_nabla = space_flat.nabla(space_flat.phi)
    assert nabla_phi.signature == expected_nabla.signature
    assert nabla_phi.components == expected_nabla.components
    assert nabla_phi.rank == 1

    nabla_nabla_phi = space_flat.nabla_nabla_phi
    expected_nabla2 = space_flat.nabla(space_flat.phi, order=2)
    assert nabla_nabla_phi.signature == expected_nabla2.signature
    assert nabla_nabla_phi.components == expected_nabla2.components
    assert nabla_nabla_phi.rank == 2
