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

## Documentation & TODO Hygiene
- Track planned work and known issues in `TODO.md`; keep entries short and actionable.
- When completing TODO items, mark them done and move user-facing changes into `CHANGELOG.md`.

## Release / Publication Steps
- Update `CHANGELOG.md` with user-facing changes and any math/formula updates.
- Ensure the package version in `src/lyra_geometry/__init__.py` is bumped.
- Run `python -m pytest` and add a brief note of the outcome in the release notes.
- If notebook outputs or formulas change, add a minimal reproduction snippet in the PR/body.

## Estrutura Kanban Integrada para Agentes

Este repositório adota uma **estrutura de Kanban integrada** orientada a **agentes automáticos (LLMs, bots, CI/workflows)**. O objetivo é garantir execução determinística baseada em **contratos explícitos**.

---

## Estrutura de Pastas

```
.kanban/
├── board.yml
└── cards/
    ├── 0001.yml
    ├── 0002.yml
    └── ...
```

### Regras Gerais

* Deve existir uma pasta **`.kanban`** na raiz do projeto.
* Todos os **metadados de tarefas** vivem dentro de `.kanban`.
* Cada card Kanban é representado por **um arquivo YAML** dentro de `.kanban/cards`.
* Quando uma task atingir o status **`Done`**, ela **DEVE sair do board** (`board.yml`).

---

## Board

Arquivo: **`.kanban/board.yml`**

* Simula **apenas** as raias:

  * `Ready`
  * `Doing`
  * `Review`
* Cards em `Done` **não aparecem** no board.
* Os cards no board deve ser listados seguindo o template: `- id - [Type] - Title`. Por exemplo:
```
- 0001 - [Bug] Adicionar erros explicitos para ranks incompativeis
- 0002 - [Research] Mapear gargalos de desempenho com expressoes SymPy grandes
- 0003 - [TechDebt] Adicionar type hints nas APIs publicas principais
```

---

# Kanban Card Contract for Agents

Este documento define o **contrato mínimo de informação** que **todo card Kanban DEVE conter** para que **agentes automáticos** consigam operar corretamente.

---

## 1. Estrutura Obrigatória do Card

Todo card **DEVE** conter os campos abaixo.

### 1.1 Identificação

```yaml
id: int
title: string
type: Feature | Bug | TechDebt | Research | Spike
priority: P1 | P2 | P3
service_class: Standard | Expedite | FixedDate | Intangible
status: Backlog | Ready | Doing | Review | Done
owner: string
```

#### Regras

* `id` deve ser **autoincrementável**
* `title` deve ser **acionável** (verbo no infinitivo é recomendado)
* `owner` deve ser **exatamente um** (humano ou agente)
* Cards sem `owner` são **inválidos**

---

## 2. Contexto (Obrigatório)

```yaml
context: |
  Explicação clara do problema ou objetivo.
  Deve permitir entendimento completo do card de forma isolada.
```

### Regra Fundamental

> Se um agente não consegue explicar o card em **um único parágrafo**,
> o contexto é considerado **insuficiente**.

---

## 3. Critérios de Aceitação (Obrigatório)

```yaml
acceptance_criteria:
  - condição objetiva e verificável
  - condição testável (sim / não)
```

### Regras

* Devem permitir **decisão binária**
* Nunca usar termos vagos como:

  * “melhorar”
  * “otimizar”
  * “ajustar”
  * “avaliar”

**Exemplo ruim**:

```
- Melhorar performance
```

**Exemplo correto**:

```
- Latência média < 200ms
```

---

## 4. Dependências e Bloqueios

### 4.1 Dependências

```yaml
dependencies:
  - card_id
  - sistema_externo
```

### 4.2 Bloqueios

```yaml
blockers:
  - descrição clara do impedimento atual
```

### Regra de Execução

> Se `blockers` **não estiver vazio**,
> o agente **DEVE ABORTAR** a execução.

---

## 5. Artefatos (Recomendado)

```yaml
artifacts:
  - link_para_PR
  - documentação
  - dataset
  - notebook
```

Artefatos servem como:

* contexto técnico
* entrada de dados
* evidência de saída

---

## 6. Metadados Técnicos (Recomendado)

```yaml
technical:
  estimate: S | M | L
  risk: Low | Medium | High
  tags:
    - LLM
    - Infra
    - Data
```

Esses campos **não bloqueiam execução**, mas ajudam planejamento e análise.

---

## 7. Campos Inferíveis (Não Obrigatórios)

```yaml
created_at: datetime
started_at: datetime
done_at: datetime
```

### Regra

Agentes **PODEM usar**, mas **NÃO DEVEM assumir** esses campos.

Eles são destinados a **métricas**, não à execução.

---

## 8. Checklist de Validação do Agente

Antes de executar qualquer ação, o agente **DEVE validar**:

* [ ] `title` existe e é acionável
* [ ] `type` é conhecido
* [ ] `owner` existe e é único
* [ ] `context` é compreensível isoladamente
* [ ] `acceptance_criteria` existe e é verificável
* [ ] `blockers` está vazio

### Regra

> Se qualquer item falhar → **ABORTAR EXECUÇÃO**

---

## 9. Comportamento Esperado do Agente

### 9.1 Card Válido

O agente **DEVE**:

1. Resumir o objetivo em **1 frase**
2. Identificar ações necessárias
3. Executar apenas o escopo definido
4. Produzir saídas alinhadas aos critérios de aceitação
5. Referenciar artefatos gerados

---

### 9.2 Card Inválido

O agente **DEVE**:

* Informar **qual campo está faltando**
* Sugerir **exatamente o conteúdo necessário**
* **Não executar nenhuma ação**

---

## 10. Exemplo de Card Válido

```yaml
id: 45
title: Ajustar chunking para RAG jurídico
type: TechDebt
priority: P2
service_class: Standard
status: Ready
owner: agent-rag-optimizer

context: |
  O chunking atual está reduzindo recall em perguntas longas
  em documentos jurídicos com mais de 20 páginas.

acceptance_criteria:
  - Recall@5 >= 0.75
  - Latência média <= 200ms

dependencies:
  - dataset_juridico_v2

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

## 11. Princípio Fundamental

> **Cards são contratos.**
> **Agentes não interpretam intenções — apenas contratos explícitos.**

## 12. Cards Executions

* Quando for executar um card:
  - leia o contexto no seu arquivo .yml
  - atualize o status no card para Doing
  - atualize o board
  - faça as alterações

* Quando finalizar um card
  - atualize o status no card para Doing
  - atualize o board
  - atualize o changelogs
  - comite as alterações