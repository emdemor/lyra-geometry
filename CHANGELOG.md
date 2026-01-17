# Changelog

## Release steps
- Ensure `Unreleased` entries are complete and grouped by change type.
- Create a new version header and move the `Unreleased` items into it.
- Bump the version in `src/lyra_geometry/__init__.py`.
- Run `python -m pytest` and note the outcome in the new release notes.
- If notebooks or formulas changed, add a minimal reproduction snippet.

## Unreleased
- tests: add GR solution invariants for Schwarzschild and de Sitter
- tests: add spherical symmetry checks for connection, curvature, Ricci, Einstein, and geodesics

## v0.1.18
- docs: translate user-facing strings and notebook outputs to English
- tests: `python -m pytest` (53 passed in 26.09s)

## v0.1.17
- add IndexedArray for non-tensor indexed objects, with variance-only indexing and Einstein contraction rules
- change Christoffel symbols to IndexedArray and add Kronecker delta + Levi-Civita symbol to TensorSpace
- add geodesic equations helper using christoffel2 and nabla_phi terms
- tests: `python -m pytest` (53 passed in 26.31s)

## v0.1.16
- add Lyra autoparallel equations helper for 4D metrics with tau/lambda parameter choice
- fix: allow scalar multiplication of Connection objects (sympy scalars and scalar tensors)
- fix: support explicit index variance when indexing Connection (gamma)
- add shortcuts for covariant derivatives of phi via st.nabla_phi and st.nabla_nabla_phi
- tests: `python -m pytest` (44 passed in 18.66s)

## v0.1.15
- refactor module layout into core/tensors/diff_ops/invariants/utils with compatibility re-exports
- docs: add module organization overview
- tests: `python -m pytest` (39 passed in 11.29s)

## v0.1.14
- add curvature invariants helpers (ricci_scalar, kretschmann_scalar) and Euler density
- add unit tests covering invariants on flat metric
- tests: `python -m pytest` (39 passed in 11.19s)

## v0.1.13
- add gradient/divergence/laplacian helpers with notebook example
- tests: `python -m pytest` (36 passed in 10.95s)

## v0.1.12
- expand README with project scope and comparisons
- add .kanban card metadata scaffolding
- raise explicit errors on tensor add/sub rank mismatches
- tests: `python -m pytest` (33 passed in 10.67s)

## v0.1.11
- require explicit index variance (+/-) in tensor indexing
- prevent label reuse after contraction in chained multiplications
- document explicit index variance in README

## v0.1.10
- aceaf76 bump version to 0.1.10
- b9adcc9 add tag trigger on cicd
- 56dd992 disable publish tag trigger
- 07ac582 update changelog
- fc1f4b9 add readme.md
- 96c692a add changelog
- 395345c add nabla order recursion

## v0.1.9
- bd5622f bump version to 0.1.9
- 350bd47 update notebook
- 24c745e fix indexed tensor addition test
- c3d40aa support tensor division
- bd31837 align indexed tensor sums
- c4a8ace add tensor reindexing

## v0.1.8
- 89f1437 bump version to 0.1.8
- 5de0260 fix labeled tensor subtraction

## v0.1.7
- 8f908ab bump version to 0.1.7
- c96dfe2 support tensor subtraction

## v0.1.6
- 779f9b7 bump version to 0.1.6
- 52f696b fix version sync regex

## v0.1.5
- 829d8ec bump version to 0.1.5
- 4b81e4e fix version regex

## v0.1.4
- 3e78ff2 bump version to 0.1.4
- 92434bb fix publish version sync

## v0.1.3
- 3b55de2 sync publish version
- faf3469 bump version to 0.1.3
- 0485227 add multiplication contraction tests
- bd36a08 add unit test suite
- ac83dc4 add connection tensor

## v0.1.2
- 47cb234 bump version to 0.1.2
- 3cacfdc fix fmt for scalar tensors
- 9836b58 update tutorial in notebook
- df1d73a update example notebook

## v0.1.1
- e861fe2 rename package

## v0.1.0
- f50b9ef move notebook update examples
- 7f546b2 restructure package
- 30f7f45 update examples
- 3d19353 allow numeric scaling of indexed tensors
- 6ddac06 fix fmt for array tensors
- 69f14f4 support numeric scaling for tensors and guard _sympy_
- 8871a95 allow scalar tensor multiply tensors
- 86f105a accept Index in tensor getitem
- 2123fa3 allow nabla on sympy expr
- b698128 add repo guidelines, ignore rules, and notebook update
- 54c8df3 add subs helper for tensors and indexed tensors
- 10d5ce7 support tensor products/scalars and default nabla prepend
- 6d6ca25 hotfix: ineverted index at connection
- d47f7ee use metric components when lowering indices
- fdce8eb fix fmt to simplify array elements
- ddd620c add fmt helper for simplifying tensors
- 15f2325 add fmt
- f70cc6d hotfix salar repr
- 8d7d85a fix repr html
- 0ad4555 add automatic sum
- bf0f5c1 fix Christoffel naming and add Connection repr
- 3cdd07b allow TensorSpace to infer dim from coords
- 65243cd add equals
- ab9c25c add symetric and antisymmetric parts
- 0799212 update sum and subtraction
- 451ef8f add greek name-to-symbol helper
- 20e6b13 allow partial indexing for IndexedTensor
- 9b4cafd update repr_html
- b23ee7f add HTML repr for IndexedTensor
- ae2924f standardize signature symbols to U/D and adjust indexing
- 8410a52 support +/- index syntax and tensor call indexing
- 83e083d add fluent up/down index builder
- 605e64d allow labeled tensor contraction via multiplication and add example
- b4dc280 introduce connection/curvature strategies and index types
- c4be6e7 fix table rank handling and ordered index parsing
- 13d395c fix detg calculation to use metric components
- f7f97d8 add R
- f91d43b add geometric tensors
- 59e02c5 wrap metric as Metric tensor and use components
- 59816aa add phi alias for scale tensor
- 5f7a59a improve tensor HTML/LaTeX repr with labels and signature
- a806f23 compute detg/Christoffel symbols and update Lyra connection
- 34a3372 support scalar tensor arithmetic and sympy conversion
- dd32418 revise example notebook content and trim redundant cells
- f2a4602 add tensor registry, labeled indexing, and contraction helpers
- 00a6e65 add metric tensors, scale/torsion/nonmetricity, and tensor contraction
- 7a95492 add dcov
- 327b9e6 rename pylyra.py to pylyra_sketch and update notebook refs
- 68bea78 first-commit
