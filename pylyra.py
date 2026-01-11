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
        self.metric = sp.Array(metric) if metric is not None else None
        self._metric_inv = None
        if metric is not None:
            self._metric_inv = (
                sp.Array(metric_inv) if metric_inv is not None else sp.Array(sp.Matrix(metric).inv())
            )
        self.metric_tensor = None
        self.metric_inv_tensor = None
        if self.metric is not None:
            self.metric_tensor = Tensor(self.metric, self, signature=(d, d))
        if self._metric_inv is not None:
            self.metric_inv_tensor = Tensor(self._metric_inv, self, signature=(u, u))
        self.connection = sp.Array(connection) if connection is not None else None
        self.scale = None
        self.torsion = None
        self.nonmetricity = None
        self.metric_compatible = None

    def set_metric(self, metric, metric_inv=None):
        self.metric = sp.Array(metric)
        if metric_inv is None:
            self._metric_inv = sp.Array(sp.Matrix(metric).inv())
        else:
            self._metric_inv = sp.Array(metric_inv)
        self.metric_tensor = Tensor(self.metric, self, signature=(d, d))
        self.metric_inv_tensor = Tensor(self._metric_inv, self, signature=(u, u))

    @property
    def metric_inv(self):
        if self._metric_inv is None and self.metric is not None:
            self._metric_inv = sp.Array(sp.Matrix(self.metric).inv())
            self.metric_inv_tensor = Tensor(self._metric_inv, self, signature=(u, u))
        return self._metric_inv

    def set_connection(self, connection):
        self.connection = sp.Array(connection)

    def set_scale(self, phi=None, coord_index=None):
        if phi is None:
            if coord_index is None:
                coord_index = 1 if len(self.coords) > 1 else 0
            phi = sp.Function("phi")(self.coords[coord_index])
        self.scale = self.scalar(phi)
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

    def from_function(self, func, signature):
        rank = _infer_rank(func)
        signature = _validate_signature(signature, rank)
        shape = (self.dim,) * rank
        flat = [func(*idx) for idx in itertools.product(range(self.dim), repeat=rank)]
        arr = sp.ImmutableDenseNDimArray(flat, shape)
        return Tensor(arr, self, signature=signature)

    def from_array(self, array, signature):
        if not isinstance(array, (sp.Array, sp.ImmutableDenseNDimArray)):
            array = sp.Array(array)
        rank = len(array.shape)
        signature = _validate_signature(signature, rank)
        if not isinstance(array, sp.ImmutableDenseNDimArray):
            array = sp.ImmutableDenseNDimArray(array)
        return Tensor(array, self, signature=signature)

    def zeros(self, signature):
        signature = _validate_signature(signature, len(signature))
        shape = (self.dim,) * len(signature)
        arr = sp.ImmutableDenseNDimArray([0] * (self.dim ** len(signature)), shape)
        return Tensor(arr, self, signature=signature)

    def scalar(self, expr):
        return Tensor(sp.Array(expr), self, signature=())

    def nabla(self, tensor, deriv_position="append"):
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

            base = sp.diff(T[idx], coords[k])
            idx_list = list(idx)

            for pos, s in enumerate(sig):
                if s is u:
                    acc = 0
                    for m in range(dim):
                        idx_list[pos] = m
                        acc += Gamma[idx[pos], k, m] * T[tuple(idx_list)]
                    base += acc
                else:
                    acc = 0
                    for m in range(dim):
                        idx_list[pos] = m
                        acc += Gamma[m, k, idx[pos]] * T[tuple(idx_list)]
                    base -= acc
                idx_list[pos] = idx[pos]

            out_flat.append(sp.simplify(base))

        out = sp.ImmutableDenseNDimArray(out_flat, shape)
        if deriv_position == "append":
            new_sig = sig + (d,)
        else:
            new_sig = (d,) + sig
        return Tensor(out, self, signature=new_sig)


class Tensor:
    def __init__(self, components, space, signature):
        self.components = sp.Array(components)
        self.rank = self.components.rank()
        self.signature = _validate_signature(signature, self.rank)
        self.space = space
        self._cache = {self.signature: self.components}

    def __call__(self, *sig):
        if len(sig) == 1 and isinstance(sig[0], (tuple, list)):
            sig = tuple(sig[0])
        arr = self.as_signature(sig, simplify=False)
        return Tensor(arr, self.space, signature=sig)

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
