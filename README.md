# Lyra Geometry

Biblioteca em Python para geometria diferencial simbolica, com suporte a espacos tensoriais, conexoes e curvatura usando SymPy.

## Instalacao

```bash
pip install lyra-geometry
```

## Exemplo rapido

```python
import sympy as sp
from lyra_geometry import TensorSpace, U, D

x, y = sp.symbols("x y")
space = TensorSpace((x, y))
T = space.generic("T", (U, D))

a, b = space.index("a b")
expr = T[U(a), D(b)]
print(expr)
```

## Desenvolvimento local

```bash
python -m pip install -e .[dev]
python -m pytest
```
