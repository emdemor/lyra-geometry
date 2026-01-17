# Repository Guidelines

## How to update versions?
- Bump the version in `pyproject.toml` and `src/lyra_geometry/__init__.py` to the same value.
- Update `CHANGELOG.md` by moving `Unreleased` entries into a new `vX.Y.Z` section (and add the test note if run).

## How publish is done?
- Publish runs via GitHub Actions on tag pushes matching `v*` in `.github/workflows/publish.yml`.
- The workflow syncs `pyproject.toml` version from `src/lyra_geometry/__init__.py`, builds with `python -m build`, and publishes to PyPI using `PYPI_API_TOKEN`.
- Create a version tag (`vX.Y.Z`) and push it to the remote to trigger the publish workflow.

## Project Structure & Module Organization
- `src/lyra_geometry/core.py`: Core library code (tensor space, connections, curvature, and helpers).
- `src/lyra_geometry/__init__.py`: Public package exports and version.
- `examples/example.ipynb`: Usage examples and exploratory calculations.
- `__pycache__/`: Local Python bytecode cache (ignore in commits).
- `tests/`: Pytest smoke tests.

## Build, Test, and Development Commands
- `python -m pip install -e .[dev]`: Install in editable mode with development/test dependencies.
- `python -c "import lyra_geometry"`: Quick sanity check for package import.
- `python -m pytest`: Run the test suite.
- `jupyter notebook examples/example.ipynb`: Run the notebook examples (if using Jupyter).

## Coding Style & Naming Conventions
- Use 4-space indentation and follow PEP 8 conventions.
- Class names use `CamelCase` (e.g., `TensorSpace`); functions and variables use `snake_case` (e.g., `from_function`).
- Prefer explicit, concise names for mathematical symbols (`g`, `Gamma`, `Riem`) while keeping public APIs readable.
- No formatter or linter is configured; format code consistently with the existing codebase.

## Testing Guidelines
- Keep tests close to the corresponding module (e.g., `tests/test_tensor.py`).
- Tests must be runnable via `python -m pytest`.
- Use descriptive test names (e.g., `test_raise_index_roundtrip`).

## Commit & Pull Request Guidelines
- Commit messages should be short and imperative (e.g., `fix repr html`, `add fmt`, `hotfix: inverted index at connection`).
- Keep commits focused and minimal.
- Include a brief description and relevant mathematical/contextual notes in the PR body.
- If changes affect outputs or formulas, include a notebook snippet or minimal reproduction steps.

## Configuration & Usage Notes
- This project relies on `sympy` for symbolic mathematics; ensure it is installed in your environment.
- Do not commit generated files such as `__pycache__` or large notebook outputs unless strictly necessary.

## Documentation & TODO Hygiene
- Track planned work and known issues in `TODO.md`; keep entries short and actionable.
- When completing TODO items, mark them as done and move user-facing changes to `CHANGELOG.md`.

## Release / Publication Steps
- Update `CHANGELOG.md` with user-facing changes and any math/formula updates.
- Bump the package version in `src/lyra_geometry/__init__.py`.
- Run `python -m pytest` and include a brief note of the result in the release notes.
- If notebook outputs or formulas change, add a minimal reproduction snippet in the PR or release notes.

---

## Integrated Kanban Structure for Agents

This repository adopts an **integrated Kanban structure** oriented toward **automatic agents (LLMs, bots, CI/workflows)**.  
The goal is to ensure deterministic execution based on **explicit contracts**.

---

## Folder Structure

```

.kanban/
├── board.yml
└── cards/
├── 0001.yml
├── 0002.yml
└── ...

```

### General Rules

- A **`.kanban`** directory must exist at the project root.
- All **task metadata** must live inside `.kanban`.
- Each Kanban card is represented by **a single YAML file** inside `.kanban/cards`.
- When a task reaches **`Done`** status, it **MUST be removed from the board** (`board.yml`).

---

## Board

File: **`.kanban/board.yml`**

- Simulates **only** the following swimlanes:
  - `Ready`
  - `Doing`
  - `Review`
- Cards in `Done` **do not appear** on the board.
- Cards must be listed using the template: `- id - [Type] - Title`. Example:
```

* 0001 - [Bug] Add explicit errors for incompatible ranks
* 0002 - [Research] Map performance bottlenecks with large SymPy expressions
* 0003 - [TechDebt] Add type hints to main public APIs

````

---

# Kanban Card Contract for Agents

This document defines the **minimum information contract** that **every Kanban card MUST contain** so that **automatic agents** can operate correctly.

---

## 1. Mandatory Card Structure

Every card **MUST** contain the following fields.

### 1.1 Identification

```yaml
id: int
title: string
type: Feature | Bug | TechDebt | Research | Spike
priority: P1 | P2 | P3
service_class: Standard | Expedite | FixedDate | Intangible
status: Backlog | Ready | Doing | Review | Done
owner: string
````

#### Rules

* `id` must be **auto-incrementing**
* `title` must be **actionable** (verb in infinitive form is recommended)
* `owner` must be **exactly one** (human or agent)
* Cards without an `owner` are **invalid**

---

## 2. Context (Mandatory)

```yaml
context: |
  Clear explanation of the problem or objective.
  Must allow full understanding of the card in isolation.
```

### Fundamental Rule

> If an agent cannot explain the card in **a single paragraph**,
> the context is considered **insufficient**.

---

## 3. Acceptance Criteria (Mandatory)

```yaml
acceptance_criteria:
  - objective and verifiable condition
  - testable condition (yes / no)
```

### Rules

* Must allow **binary decisions**
* Never use vague terms such as:

  * “improve”
  * “optimize”
  * “adjust”
  * “evaluate”

**Bad example**:

```
- Improve performance
```

**Correct example**:

```
- Average latency < 200ms
```

---

## 4. Dependencies and Blockers

### 4.1 Dependencies

```yaml
dependencies:
  - card_id
  - external_system
```

### 4.2 Blockers

```yaml
blockers:
  - clear description of the current impediment
```

### Execution Rule

> If `blockers` is **not empty**,
> the agent **MUST ABORT** execution.

---

## 5. Artifacts (Recommended)

```yaml
artifacts:
  - link_to_PR
  - documentation
  - dataset
  - notebook
```

Artifacts serve as:

* technical context
* input data
* output evidence

---

## 6. Technical Metadata (Recommended)

```yaml
technical:
  estimate: S | M | L
  risk: Low | Medium | High
  tags:
    - LLM
    - Infra
    - Data
```

These fields **do not block execution**, but help planning and analysis.

---

## 7. Inferable Fields (Optional)

```yaml
created_at: datetime
started_at: datetime
done_at: datetime
```

### Rule

Agents **MAY use** but **MUST NOT assume** these fields.

They are intended for **metrics**, not execution.

---

## 8. Agent Validation Checklist

Before executing any action, the agent **MUST validate**:

* [ ] `title` exists and is actionable
* [ ] `type` is known
* [ ] `owner` exists and is unique
* [ ] `context` is understandable in isolation
* [ ] `acceptance_criteria` exists and is verifiable
* [ ] `blockers` is empty

### Rule

> If any item fails → **ABORT EXECUTION**

---

## 9. Expected Agent Behavior

### 9.1 Valid Card

The agent **MUST**:

1. Summarize the objective in **1 sentence**
2. Identify required actions
3. Execute **only** the defined scope
4. Produce outputs aligned with the acceptance criteria
5. Reference generated artifacts

---

### 9.2 Invalid Card

The agent **MUST**:

* Inform **which field is missing**
* Suggest **exactly what content is required**
* **Not execute any action**

---

## 10. Example of a Valid Card

```yaml
id: 45
title: Adjust chunking for legal RAG
type: TechDebt
priority: P2
service_class: Standard
status: Ready
owner: agent-rag-optimizer

context: |
  The current chunking strategy is reducing recall for long questions
  in legal documents with more than 20 pages.

acceptance_criteria:
  - Recall@5 >= 0.75
  - Average latency <= 200ms

dependencies:
  - legal_dataset_v2

blockers: []

artifacts:
  - docs/rag/chunking.md
  - notebooks/eval_chunking.ipynb

technical:
  estimate: M
  risk: Medium
  tags: [LLM, RAG, NLP]
```

---

## 11. Fundamental Principle

> **Cards are contracts.**
> **Agents do not interpret intentions — only explicit contracts.**

---

## 12. Card Execution Flow

### When executing a card:

* Read the context from its `.yml` file
* Update the card status to `Doing`
* Update the board
* Apply the required changes
* When appropriate, add relevant unit tests
* Run the test pipeline to ensure everything works

### When finishing a card:

* Update the card status to `Done`
* Update the board
* Update the changelogs
* Commit the changes using the template:
  `{{card-id}}-{{type-with-no-spaces}}-{{title-with-no-spaces}}`

```
