# Repository Guidelines

## Project Structure & Module Organization
- `src/lyra_geometry/core.py`: Core library code (tensor space, connections, curvature, and helpers).
- `src/lyra_geometry/__init__.py`: Public package exports and version.
- `examples/example.ipynb`: Usage examples and exploratory calculations.
- `__pycache__/`: Local Python bytecode cache (ignore for commits).
- `tests/`: Pytest smoke tests.

## Build, Test, and Development Commands
- `python -m pip install -e .[dev]`: Install in editable mode with test deps.
- `python -c "import lyra_geometry"`: Quick import sanity check for the package.
- `python -m pytest`: Run tests.
- `jupyter notebook examples/example.ipynb`: Run the notebook examples (if you use Jupyter).

## Coding Style & Naming Conventions
- Use 4-space indentation and follow PEP 8 conventions.
- Class names use `CamelCase` (e.g., `TensorSpace`); functions and variables use `snake_case` (e.g., `from_function`).
- Prefer explicit, short names for mathematical symbols (`g`, `Gamma`, `Riem`) but keep public APIs readable.
- No formatter or linter is configured; format consistently with existing code.

## Testing Guidelines
- Keep tests close to the module (e.g., `tests/test_tensor.py`) and make them runnable with `python -m pytest`.
- Name tests descriptively (e.g., `test_raise_index_roundtrip`).

## Commit & Pull Request Guidelines
- Commit messages are short and imperative in this repo (e.g., `fix repr html`, `add fmt`, `hotfix: ineverted index at connection`).
- Keep commits focused; include a brief description and any relevant math/context in the PR body.
- If changes affect outputs or formulas, include a notebook snippet or minimal reproduction steps.

## Configuration & Usage Notes
- This project relies on `sympy` for symbolic math; ensure it is available in your environment.
- Avoid committing generated files like `__pycache__` and large notebook outputs unless needed.
