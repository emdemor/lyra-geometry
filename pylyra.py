import itertools
import inspect
import sympy as sp


class Up:
    pass


class Down:
    pass


u = Up()

d = Down()


def _norm_sig(sig, rank):
    if len(sig) != rank:
        raise ValueError(f"Assinatura tem tamanho {len(sig)} mas rank e {rank}.")
    out = []
    for s in sig:
        if s in (u, Up, "u", "^", +1, True):
            out.append(u)
        elif s in (d, Down, "d", "_", -1, False):
            out.append(d)
        else:
            raise ValueError(f"Elemento de assinatura invalido: {s!r}. Use u/d.")
    return tuple(out)


def _validate_signature(signature, rank):
    if not isinstance(signature, (tuple, list)):
        raise TypeError("signature deve ser tupla/lista, ex.: (u,d,d).")
    if len(signature) != rank:
        raise ValueError(f"signature tem tamanho {len(signature)}, mas rank={rank}.")
    return _norm_sig(signature, rank)


def table(func, dim):
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("Use apenas argumentos posicionais para indices.")
    rank = len(sig.parameters)
    shape = (dim,) * rank
    flat = [func(*idx) for idx in itertools.product(range(dim), repeat=rank)]
    return sp.ImmutableDenseNDimArray(flat, shape)


class TensorSpace:
    def __init__(self, dim, coords, metric=None, metric_inv=None, connection=None):
        self.dim = dim
        self.coords = tuple(coords)
        self._tensor_count = 0
        self._label_count = 0
        self._registry = {}
        self.metric = Metric(sp.Array(metric), self, signature=(d, d), name="g", label="g") if metric is not None else None
        self._metric_inv = None
        if metric is not None:
            self._metric_inv = (
                sp.Array(metric_inv) if metric_inv is not None else sp.Array(sp.Matrix(metric).inv())
            )
        self.metric_tensor = None
        self.metric_inv_tensor = None
        self.g = None
        self._detg = None
        self.christoffell1 = None
        self.christoffel2 = None
        self.christoffel1 = None
        if self.metric is not None:
            self.metric_tensor = self.register(self.metric)
        if self._metric_inv is not None:
            self.metric_inv_tensor = self.register(
                Tensor(self._metric_inv, self, signature=(u, u), name="g_inv", label="g_inv")
            )
        self.gamma = Connection(connection) if connection is not None else Connection(None)
        self.scale = self.scalar(1, name="phi", label="phi")
        self.phi = self.scale
        self.torsion = self.zeros((d, d, d), name="tau", label="tau")
        self.nonmetricity = self.zeros((u, d, d), name="M", label="M")
        self.metric_compatible = None
        self.tensor = TensorFactory(self)
        self.riemann = None
        self.ricci = None
        self.einstein = None
        self.update()

    def set_metric(self, metric, metric_inv=None):
        self.metric = Metric(sp.Array(metric), self, signature=(d, d), name="g", label="g")
        if metric_inv is None:
            self._metric_inv = sp.Array(sp.Matrix(metric).inv())
        else:
            self._metric_inv = sp.Array(metric_inv)
        self.metric_tensor = self.register(self.metric)
        self.metric_inv_tensor = self.register(
            Tensor(self._metric_inv, self, signature=(u, u), name="g_inv", label="g_inv")
        )

    @property
    def metric_inv(self):
        if self._metric_inv is None and self.metric is not None:
            self._metric_inv = sp.Array(sp.Matrix(self.metric.components).inv())
            self.metric_inv_tensor = self.register(
                Tensor(self._metric_inv, self, signature=(u, u), name="g_inv", label="g_inv")
            )
        return self._metric_inv

    @property
    def detg(self):
        if self._detg is None and self.metric is not None:
            self._detg = sp.simplify(sp.Matrix(self.metric).det())
        return self._detg

    @property
    def connection(self):
        return self.gamma.components

    def _next_tensor_name(self):
        self._tensor_count += 1
        return f"T{self._tensor_count}"

    def _next_label(self):
        self._label_count += 1
        return f"_{self._label_count}"

    def register(self, tensor):
        self._registry[tensor.name] = tensor
        return tensor

    def get(self, name):
        return self._registry.get(name)

    def set_connection(self, connection):
        self.gamma = Connection(connection)

    def set_scale(self, phi=None, coord_index=None):
        if phi is None:
            if coord_index is None:
                coord_index = 1 if len(self.coords) > 1 else 0
            phi = sp.Function("phi")(self.coords[coord_index])
        self.scale = self.scalar(phi, name="phi", label="phi")
        self.phi = self.scale
        return self.scale

    def set_torsion(self, torsion_tensor):
        if isinstance(torsion_tensor, Tensor):
            if torsion_tensor.space is not self:
                raise ValueError("Torsion tensor pertence a outro TensorSpace.")
            self.torsion = torsion_tensor
        else:
            self.torsion = self.from_array(torsion_tensor, signature=(d, d, d))
        return self.torsion

    def set_nonmetricity(self, nonmetricity_tensor):
        if isinstance(nonmetricity_tensor, Tensor):
            if nonmetricity_tensor.space is not self:
                raise ValueError("Non-metricity tensor pertence a outro TensorSpace.")
            self.nonmetricity = nonmetricity_tensor
        else:
            self.nonmetricity = self.from_array(nonmetricity_tensor, signature=(u, d, d))
        return self.nonmetricity

    def set_metric_compatibility(self, compatible=True):
        self.metric_compatible = bool(compatible)
        return self.metric_compatible

    def _update_metric_related(self):
        self.g = self.metric
        if self.metric is None:
            self._detg = None
            self.christoffell1 = None
            self.christoffel2 = None
            self.christoffel1 = None
            return

        g = self.metric.components
        coords = self.coords
        dim = self.dim
        self._detg = sp.simplify(sp.Matrix(g).det())

        chris1 = [[[
            sp.Rational(1, 2)
            * (
                sp.diff(g[a, c], coords[b])
                + sp.diff(g[a, b], coords[c])
                - sp.diff(g[b, c], coords[a])
            )
            for c in range(dim)
        ] for b in range(dim)] for a in range(dim)]
        self.christoffell1 = sp.Array(chris1)
        self.christoffel1 = self.christoffell1

        g_inv = self.metric_inv
        chris2 = [[[
            sum(g_inv[a, d] * self.christoffell1[d, b, c] for d in range(dim))
            for c in range(dim)
        ] for b in range(dim)] for a in range(dim)]
        self.christoffel2 = sp.Array(chris2)

    def _update_connection(self):
        if self.metric is None:
            self.gamma = Connection(None)
            return

        dim = self.dim
        coords = self.coords
        g = self.metric.components
        g_inv = self.metric_inv
        phi = self.scale.expr if isinstance(self.scale, Tensor) else self.scale
        M = self.nonmetricity
        tau = self.torsion
        chris = self.christoffel2

        def connection_element(b, n, l):
            return (
                1 / phi * chris[b, n, l]
                - sp.Rational(1, 2) * M(u, d, d)[b, n, l]
                + 1 / (phi) * (
                    sp.KroneckerDelta(b, n) * 1 / phi * sp.diff(phi, coords[l])
                    - sum(1 / phi * g[n, l] * g_inv[b, s] * sp.diff(phi, coords[s]) for s in range(dim))
                )
                + sp.Rational(1, 2) * sum(
                    g_inv[m, b] * (
                        tau(d, d, d)[l, m, n] - tau(d, d, d)[n, l, m] - tau(d, d, d)[m, l, n]
                    )
                    for m in range(dim)
                )
            )

        Gamma = table(connection_element, dim=dim)
        self.gamma = Connection(Gamma)

    def _update_riemann(self):
        if self.gamma.components is None or self.metric is None:
            self.riemann = None
            self.ricci = None
            self.einstein = None
            return

        dim = self.dim
        coords = self.coords
        Gamma = self.gamma.components
        phi = self.phi.expr if isinstance(self.phi, Tensor) else self.phi

        def curvature_element(l, a, m, n):
            return (
                1 / (phi**2) * sp.diff(phi * Gamma[l, a, n], coords[m])
                - 1 / (phi**2) * sp.diff(phi * Gamma[l, a, m], coords[n])
                + sum(Gamma[r, a, n] * Gamma[l, r, m] for r in range(dim))
                - sum(Gamma[r, a, m] * Gamma[l, r, n] for r in range(dim))
            )

        Riem = self.from_function(curvature_element, signature=(u, d, d, d), name="Riemann", label="R")

        def ricci_element(a, m):
            return sp.simplify(sum(Riem(u, d, d, d).comp[l, a, m, l] for l in range(dim)))

        Ricc = self.from_function(ricci_element, signature=(d, d), name="Ricci", label="Ric")

        g_inv = self.metric_inv
        scalar_R = sp.simplify(sum(g_inv[a, b] * Ricc.comp[a, b] for a in range(dim) for b in range(dim)))

        def einstein_element(a, b):
            return sp.simplify(Ricc.comp[a, b] - sp.Rational(1, 2) * self.g.components[a, b] * scalar_R)

        Ein = self.from_function(einstein_element, signature=(d, d), name="Einstein", label="G")
        self.riemann = Riem
        self.ricci = Ricc
        self.einstein = Ein

    def update(self, include=None, exclude=()):
        available = {
            "scale",
            "metric",
            "detg",
            "christoffel",
            "connection",
            "riemann",
            "ricci",
            "einstein",
        }
        if include is None:
            steps = set(available)
        else:
            steps = set(include)
        steps -= set(exclude)

        if "metric" in steps or "detg" in steps or "christoffel" in steps:
            self._update_metric_related()
        if "connection" in steps:
            self._update_connection()
        if "riemann" in steps or "ricci" in steps or "einstein" in steps:
            self._update_riemann()

    def from_function(self, func, signature, name=None, label=None):
        rank = _infer_rank(func)
        signature = _validate_signature(signature, rank)
        shape = (self.dim,) * rank
        flat = [func(*idx) for idx in itertools.product(range(self.dim), repeat=rank)]
        arr = sp.ImmutableDenseNDimArray(flat, shape)
        return self.register(Tensor(arr, self, signature=signature, name=name, label=label))

    def from_array(self, array, signature, name=None, label=None):
        if not isinstance(array, (sp.Array, sp.ImmutableDenseNDimArray)):
            array = sp.Array(array)
        rank = len(array.shape)
        signature = _validate_signature(signature, rank)
        if not isinstance(array, sp.ImmutableDenseNDimArray):
            array = sp.ImmutableDenseNDimArray(array)
        return self.register(Tensor(array, self, signature=signature, name=name, label=label))

    def zeros(self, signature, name=None, label=None):
        signature = _validate_signature(signature, len(signature))
        shape = (self.dim,) * len(signature)
        arr = sp.ImmutableDenseNDimArray([0] * (self.dim ** len(signature)), shape)
        return self.register(Tensor(arr, self, signature=signature, name=name, label=label))

    def scalar(self, expr, name=None, label=None):
        return self.register(Tensor(sp.Array(expr), self, signature=(), name=name, label=label))

    def generic(self, name, signature, coords=None, label=None):
        signature = _validate_signature(signature, len(signature))
        coords = self.coords if coords is None else tuple(coords)
        rank = len(signature)
        shape = (self.dim,) * rank

        def comp(*idx):
            suf = "".join(map(str, idx))
            return sp.Function(f"{name}{suf}")(*coords)

        flat = [comp(*idx) for idx in itertools.product(range(self.dim), repeat=rank)]
        arr = sp.ImmutableDenseNDimArray(flat, shape)
        return self.register(Tensor(arr, self, signature=signature, name=name, label=label or name))

    def nabla(self, tensor, deriv_position="append"):
        """
        Derivada covariante de Lyra:
        ∇_k T = (1/phi) ∂_k T + Σ Γ^{a_i}{}_{m k} T^{...m...} - Σ Γ^{m}{}_{b_j k} T_{...m...}
        """
        if self.connection is None:
            raise ValueError("Defina a conexao (Gamma^a_{bc}) em TensorSpace.")
        if tensor.space is not self:
            raise ValueError("Tensor pertence a outro TensorSpace.")

        dim = self.dim
        coords = self.coords
        Gamma = self.connection
        T = tensor.components
        rank = tensor.rank
        sig = tensor.signature

        shape = (dim,) * (rank + 1)
        out_flat = []

        for full_idx in itertools.product(range(dim), repeat=rank + 1):
            if deriv_position == "append":
                idx = full_idx[:-1]
                k = full_idx[-1]
            elif deriv_position == "prepend":
                k = full_idx[0]
                idx = full_idx[1:]
            else:
                raise ValueError("deriv_position deve ser 'append' ou 'prepend'.")

            phi = self.phi.expr if isinstance(self.phi, Tensor) else self.phi
            base = (1 / phi) * sp.diff(T[idx], coords[k])
            idx_list = list(idx)

            for pos, s in enumerate(sig):
                if s is u:
                    acc = 0
                    for m in range(dim):
                        idx_list[pos] = m
                        acc += Gamma[idx[pos], m, k] * T[tuple(idx_list)]
                    base += acc
                else:
                    acc = 0
                    for m in range(dim):
                        idx_list[pos] = m
                        acc += Gamma[m, idx[pos], k] * T[tuple(idx_list)]
                    base -= acc
                idx_list[pos] = idx[pos]

            out_flat.append(sp.simplify(base))

        out = sp.ImmutableDenseNDimArray(out_flat, shape)
        if deriv_position == "append":
            new_sig = sig + (d,)
        else:
            new_sig = (d,) + sig
        return Tensor(out, self, signature=new_sig, name=None, label=tensor.label)

    def index(self, names):
        if isinstance(names, str):
            parts = [p for p in names.replace(",", " ").split() if p]
        else:
            parts = list(names)
        out = []
        for p in parts:
            if p in ("_", ".", "empty", None):
                out.append(None)
            else:
                out.append(Index(str(p)))
        return out[0] if len(out) == 1 else tuple(out)

    def contract(self, *indexed_tensors):
        if not indexed_tensors:
            raise ValueError("Informe ao menos um tensor indexado.")

        tensors = [it if isinstance(it, IndexedTensor) else it.idx() for it in indexed_tensors]
        A = tensors[0].components
        sig = list(tensors[0].signature)
        labels = list(tensors[0].labels)

        for t in tensors[1:]:
            A = sp.tensorproduct(A, t.components)
            sig.extend(t.signature)
            labels.extend(t.labels)

        label_map = {}
        for pos, (lab, s) in enumerate(zip(labels, sig)):
            if lab is None:
                continue
            label_map.setdefault(lab, []).append((pos, s))

        pairs = []
        to_remove = set()
        for lab, occ in label_map.items():
            if len(occ) == 1:
                continue
            if len(occ) != 2:
                raise ValueError(f"Indice {lab} aparece {len(occ)} vezes.")
            (p1, s1), (p2, s2) = occ
            if s1 is s2:
                raise ValueError(f"Indice {lab} aparece com mesma variancia.")
            pairs.append((p1, p2))
            to_remove.update([p1, p2])

        if pairs:
            A = sp.tensorcontraction(A, *pairs)

        new_sig = tuple(s for i, s in enumerate(sig) if i not in to_remove)
        new_labels = [lab for i, lab in enumerate(labels) if i not in to_remove]
        result = Tensor(A, self, signature=new_sig, name=None, label=None)
        result._labels = new_labels
        return result

    def eval_contract(self, expr):
        tensors = []
        for token in expr.split():
            name, up_labels, down_labels = _parse_tensor_token(token)
            tensor = self.get(name)
            if tensor is None:
                raise ValueError(f"Tensor '{name}' nao registrado.")
            up_full, down_full = _expand_indices(tensor.rank, up_labels, down_labels)
            indexed = tensor.idx(up=up_full, down=down_full)
            tensors.append(indexed)
        return self.contract(*tensors)


class Tensor:
    def __init__(self, components, space, signature, name=None, label=None):
        self.components = sp.Array(components)
        self.rank = self.components.rank()
        self.signature = _validate_signature(signature, self.rank)
        self.space = space
        self.name = name if name is not None else space._next_tensor_name()
        self.label = label if label is not None else self.name
        self._cache = {self.signature: self.components}

    def _as_scalar(self):
        if self.rank != 0:
            raise TypeError("Operacao escalar so e valida para tensores de rank 0.")
        return sp.sympify(self.components[()])

    @property
    def expr(self):
        return self._as_scalar()

    def _sympy_(self):
        return self._as_scalar()

    def _repr_latex_(self):
        if self.rank == 0:
            return self._as_scalar()._repr_latex_()
        sig = "".join("^" if s is u else "_" for s in self.signature)
        return r"\text{%s}(%s)\ \in\ \mathbb{R}^{%s}" % (self.label, self.components.shape, sig)

    def _repr_html_(self):
        if self.rank == 0:
            expr = self._as_scalar()
            if hasattr(expr, "_repr_html_"):
                return expr._repr_html_()
        sig = "".join("^" if s is u else "_" for s in self.signature)
        return (
            f"<div><b>{self.label}</b> &nbsp;"
            f"<code>shape={self.components.shape}</code> &nbsp;"
            f"<code>sig={sig}</code></div>"
        )

    def __call__(self, *sig):
        if len(sig) == 1 and isinstance(sig[0], (tuple, list)):
            sig = tuple(sig[0])
        arr = self.as_signature(sig, simplify=False)
        return Tensor(arr, self.space, signature=sig, name=self.name, label=self.label)

    def __add__(self, other):
        if self.rank == 0:
            return self._as_scalar() + other
        return NotImplemented

    def __radd__(self, other):
        if self.rank == 0:
            return other + self._as_scalar()
        return NotImplemented

    def __sub__(self, other):
        if self.rank == 0:
            return self._as_scalar() - other
        return NotImplemented

    def __rsub__(self, other):
        if self.rank == 0:
            return other - self._as_scalar()
        return NotImplemented

    def __mul__(self, other):
        if self.rank == 0:
            return self._as_scalar() * other
        return NotImplemented

    def __rmul__(self, other):
        if self.rank == 0:
            return other * self._as_scalar()
        return NotImplemented

    def __truediv__(self, other):
        if self.rank == 0:
            return self._as_scalar() / other
        return NotImplemented

    def __rtruediv__(self, other):
        if self.rank == 0:
            return other / self._as_scalar()
        return NotImplemented

    def __pow__(self, power):
        if self.rank == 0:
            return self._as_scalar() ** power
        return NotImplemented

    def __neg__(self):
        if self.rank == 0:
            return -self._as_scalar()
        return NotImplemented

    def __getitem__(self, idx):
        return self.components[idx]

    @property
    def comp(self):
        return self.components

    def _move_front_axis_to(self, A, pos):
        rank = A.rank()
        perm = []
        rest = list(range(1, rank))
        for i in range(rank):
            if i == pos:
                perm.append(0)
            else:
                perm.append(rest.pop(0))
        return sp.permutedims(A, perm)

    def _raise_at(self, A, pos):
        if self.space.metric_inv is None:
            raise ValueError("Metric inverse nao definido para subir indices.")
        TP = sp.tensorproduct(self.space.metric_inv, A)
        C = sp.tensorcontraction(TP, (1, pos + 2))
        return self._move_front_axis_to(C, pos)

    def _lower_at(self, A, pos):
        if self.space.metric is None:
            raise ValueError("Metric nao definida para descer indices.")
        TP = sp.tensorproduct(self.space.metric, A)
        C = sp.tensorcontraction(TP, (1, pos + 2))
        return self._move_front_axis_to(C, pos)

    def as_signature(self, target_signature, simplify=False):
        target_signature = _validate_signature(target_signature, self.rank)
        if target_signature in self._cache:
            return self._cache[target_signature]

        A = self._cache[self.signature]
        sig_cur = list(self.signature)
        for pos in range(self.rank):
            want = target_signature[pos]
            have = sig_cur[pos]
            if have is want:
                continue
            if have is d and want is u:
                A = self._raise_at(A, pos)
                sig_cur[pos] = u
            elif have is u and want is d:
                A = self._lower_at(A, pos)
                sig_cur[pos] = d
            else:
                raise RuntimeError("Estado impossivel na conversao de assinatura.")

        if simplify:
            A = sp.simplify(A)
        self._cache[target_signature] = A
        return A

    def nabla(self, deriv_position="append"):
        return self.space.nabla(self, deriv_position=deriv_position)

    def contract(self, pos1, pos2, use_metric=True):
        if pos1 == pos2:
            raise ValueError("pos1 e pos2 devem ser indices distintos.")
        if not (0 <= pos1 < self.rank and 0 <= pos2 < self.rank):
            raise IndexError("pos1/pos2 fora do rank do tensor.")

        sig = list(self.signature)
        s1 = sig[pos1]
        s2 = sig[pos2]
        A = self.components

        if s1 is s2:
            if not use_metric:
                raise ValueError("Indices com mesma variancia exigem use_metric=True.")
            if s1 is d:
                A = self.as_signature(
                    tuple(u if i == pos2 else s for i, s in enumerate(sig))
                )
                sig[pos2] = u
            else:
                A = self.as_signature(
                    tuple(d if i == pos2 else s for i, s in enumerate(sig))
                )
                sig[pos2] = d

        contracted = sp.tensorcontraction(A, (pos1, pos2))
        new_sig = tuple(s for i, s in enumerate(sig) if i not in (pos1, pos2))
        return Tensor(contracted, self.space, signature=new_sig)

    def idx(self, up=None, down=None):
        rank = self.rank
        if up is None and down is None:
            up = [None] * rank
            down = [None] * rank
        elif up is None or down is None:
            raise ValueError("Forneca up e down com o mesmo tamanho do rank.")

        up = list(up)
        down = list(down)
        if len(up) != rank or len(down) != rank:
            raise ValueError("up/down devem ter tamanho igual ao rank do tensor.")

        labels = []
        target_sig = []
        for i in range(rank):
            up_i = _parse_label(up[i], self.space)
            down_i = _parse_label(down[i], self.space)
            if up_i is not None and down_i is not None:
                raise ValueError("Indice nao pode ser up e down na mesma posicao.")
            if up_i is None and down_i is None:
                target_sig.append(self.signature[i])
                labels.append(self.space._next_label())
            elif up_i is not None:
                target_sig.append(u)
                labels.append(self.space._next_label() if up_i is NO_LABEL else up_i)
            else:
                target_sig.append(d)
                labels.append(self.space._next_label() if down_i is NO_LABEL else down_i)

        A = self.as_signature(tuple(target_sig), simplify=False)
        return IndexedTensor(self, A, tuple(target_sig), labels)


class Metric(Tensor):
    pass


class IndexedTensor:
    def __init__(self, tensor, components, signature, labels):
        self.tensor = tensor
        self.components = components
        self.signature = signature
        self.labels = labels


class Connection:
    def __init__(self, components):
        self.components = sp.Array(components) if components is not None else None

    def __getitem__(self, idx):
        if self.components is None:
            raise ValueError("Conexao nao definida.")
        return self.components[idx]


class TensorFactory:
    def __init__(self, space):
        self.space = space

    def from_function(self, func, signature, name=None, label=None):
        return self.space.from_function(func, signature, name=name, label=label)

    def from_array(self, array, signature, name=None, label=None):
        return self.space.from_array(array, signature, name=name, label=label)

    def generic(self, name, signature, coords=None, label=None):
        return self.space.generic(name, signature, coords=coords, label=label)

    def zeros(self, signature, name=None, label=None):
        return self.space.zeros(signature, name=name, label=label)

    def scalar(self, expr, name=None, label=None):
        return self.space.scalar(expr, name=name, label=label)


class Index:
    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return self.name


class _NoLabel:
    pass


NO_LABEL = _NoLabel()


def _parse_label(label, space):
    if label is NO_LABEL:
        return NO_LABEL
    if label in ("_", ".", "empty", None):
        return None
    if isinstance(label, Index):
        return label.name
    if isinstance(label, str):
        return label.strip()
    return str(label)


def _parse_tensor_token(token):
    name = ""
    up = []
    down = []
    i = 0
    while i < len(token) and token[i].isalnum():
        name += token[i]
        i += 1
    while i < len(token):
        if token[i] == "^":
            i += 1
            if i < len(token) and token[i] == "{":
                block, i = _read_block(token, i)
                up = _split_indices(block)
        elif token[i] == "_":
            i += 1
            if i < len(token) and token[i] == "{":
                block, i = _read_block(token, i)
                down = _split_indices(block)
        else:
            i += 1
    return name, up, down


def _expand_indices(rank, up_labels, down_labels):
    up_labels = [] if up_labels is None else list(up_labels)
    down_labels = [] if down_labels is None else list(down_labels)
    if len(up_labels) + len(down_labels) != rank:
        raise ValueError("Numero de indices nao bate com o rank do tensor.")
    up_full = [None] * rank
    down_full = [None] * rank
    for i, lab in enumerate(up_labels):
        up_full[i] = lab
    for i, lab in enumerate(down_labels):
        down_full[len(up_labels) + i] = lab
    return up_full, down_full


def _read_block(s, i):
    if s[i] != "{":
        raise ValueError("Esperado '{' na expressao de indices.")
    depth = 0
    start = i + 1
    i += 1
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            if depth == 0:
                return s[start:i], i + 1
            depth -= 1
        i += 1
    raise ValueError("Bloco de indices nao fechado.")


def _split_indices(block):
    out = []
    for part in block.split(","):
        part = part.strip()
        if part in ("", "_", ".", "empty"):
            out.append(NO_LABEL)
        else:
            out.append(part)
    return out


class SpaceTime(TensorSpace):
    pass


class Manifold(TensorSpace):
    pass


def _infer_rank(func):
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("Use apenas argumentos posicionais para indices.")
    return len(sig.parameters)
