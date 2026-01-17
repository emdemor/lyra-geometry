# TODO / Issues

This file tracks planned work, known issues, and tech debt. Keep entries short,
actionable, and dated when added.

## How to use
- Use checkboxes for planned work; mark completed items and move to CHANGELOG.
- Open a GitHub issue (if/when the project uses one) for user-facing bugs.
- Keep scope small; split large items into sub-tasks.

## Roadmap
- [ ] Publish minimal API reference in README.
- [x] Add examples for curvature and connection workflows.
- [ ] Expand smoke tests for tensor contraction edge cases.

## Known issues
- [ ] Missing explicit error messages for incompatible tensor ranks.
- [ ] Performance degrades with large symbolic expressions (SymPy simplify).

## Tech debt / Maintenance
- [ ] Add type hints for core public APIs.
- [ ] Add lightweight benchmarking script.
- [ ] Document versioning policy and release steps.

## Ideas / Backlog
- [ ] Add geodesic solver (symbolic + numeric) with initial conditions.
- [ ] Add differential operators (grad/div/Laplacian) on tensor fields.
- [ ] Support orthonormal frames (vielbein) and torsionful connections.
- [ ] Differential operators: gradient, divergence, Laplacian, and Hodge star on generic manifolds.
- [ ] Invariants: curvature scalars (Ricci, Kretschmann), simplified Euler/Chern classes.
- [ ] Integration with sympy.diffgeom/sympy.tensor for interoperability.
- [ ] Simplification API: automatic symmetry rules (Riemann, Bianchi) and normal forms.
