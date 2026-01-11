import sympy as sp
import itertools
import inspect

class Up:   pass
class Down: pass

u = Up()
d = Down()

def _norm_sig(sig, rank):
    """Normaliza assinatura para uma tupla de Up/Down com tamanho = rank."""
    if len(sig) != rank:
        raise ValueError(f"Assinatura tem tamanho {len(sig)} mas rank é {rank}.")
    out = []
    for s in sig:
        if s in (u, Up, "u", "^", +1, True):
            out.append(u)
        elif s in (d, Down, "d", "_", -1, False):
            out.append(d)
        else:
            raise ValueError(f"Elemento de assinatura inválido: {s!r}. Use u/d.")
    return tuple(out)


def table(func, dim=4):

    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind not in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TypeError("Use apenas argumentos posicionais (sem *args/**kwargs) para índices.")
    rank = len(sig.parameters)

    shape = (dim,) * rank
    flat = [func(*idx) for idx in itertools.product(range(dim), repeat=rank)]
    arr = sp.ImmutableDenseNDimArray(flat, shape)
    return arr




class Tensor:
    """
    Wrapper para um tensor em componentes (sympy.Array),
    com assinatura de índices (u/d) e métrica para subir/descer.
    """
    def __call__(self, *sig):
        if len(sig) == 1 and isinstance(sig[0], (tuple, list)):
            sig = tuple(sig[0])
        A = self.as_signature(sig, simplify=False)
        return Tensor(A, self.g, self.g_inv, signature=sig)

    def __getitem__(self, idx):
        """
        Agora __getitem__ é APENAS para componentes: eps[0,1,2,3]
        (não mais para assinatura).
        """
        return self.T[idx]
    
    @property
    def comp(self):
        # acesso explícito aos componentes
        return self.T

    def __init__(self, T, g, g_inv=None, signature=None):
        self.T = sp.Array(T)
        self.rank = self.T.rank()

        self.g = sp.Array(g)
        self.g_inv = sp.Array(g_inv if g_inv is not None else sp.Matrix(g).inv())

        if signature is None:
            raise ValueError("Passe signature, ex.: (u,u,d) ou (d,d).")
        self.signature = _norm_sig(signature, self.rank)

        # cache: target_signature -> Array convertido
        self._cache = {self.signature: self.T}

    def _repr_latex_(self):
        return self.T._repr_latex_()

    def __repr__(self):
        return repr(self.T)

    def _move_front_axis_to(self, A, pos):
        """
        A vem com o eixo 0 sendo o 'novo índice' (vindo da métrica).
        Move esse eixo 0 para a posição 'pos', preservando a ordem relativa dos demais.
        """
        rank = A.rank()
        perm = []
        # vamos montar a permutação final dos eixos
        # eixo 0 (novo) deve ir para 'pos'
        # os demais eixos 1..rank-1 preenchem o resto em ordem
        rest = list(range(1, rank))
        for i in range(rank):
            if i == pos:
                perm.append(0)
            else:
                perm.append(rest.pop(0))
        return sp.permutedims(A, perm)

    def _raise_at(self, A, pos):
        """
        Sobe o índice na posição pos: ..._μ... -> ...^μ...
        Faz: g^{μ α} A_{... α ...}  (sem mudar a ordem final dos índices)
        """
        TP = sp.tensorproduct(self.g_inv, A)
        C  = sp.tensorcontraction(TP, (1, pos + 2))  # eixo 0 é o índice novo
        return self._move_front_axis_to(C, pos)

    def _lower_at(self, A, pos):
        """
        Desce o índice na posição pos: ...^μ... -> ..._μ...
        Faz: g_{μ α} A^{... α ...} (sem mudar a ordem final dos índices)
        """
        TP = sp.tensorproduct(self.g, A)
        C  = sp.tensorcontraction(TP, (1, pos + 2))  # eixo 0 é o índice novo
        return self._move_front_axis_to(C, pos)

    def as_signature(self, target_signature, simplify=False):
        """
        Converte o tensor para a assinatura desejada (u/d).
        """
        target_signature = _norm_sig(target_signature, self.rank)

        if target_signature in self._cache:
            return self._cache[target_signature]

        A = self._cache[self.signature]  # começa do "original"
        sig_cur = list(self.signature)

        # Converte índice a índice (ordem não muda o rank)
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
                raise RuntimeError("Estado impossível na conversão de assinatura.")

        if simplify:
            A = sp.simplify(A)

        self._cache[target_signature] = A
        return A


class TensorBuilder:
    def __init__(self, dim, g, g_inv=None):
        self.dim = dim
        self.g = sp.Array(g)
        self.g_inv = sp.Array(g_inv if g_inv is not None else sp.Matrix(g).inv())

    def _infer_rank(self, func):
        sig = inspect.signature(func)
        for p in sig.parameters.values():
            if p.kind not in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD):
                raise TypeError("Use apenas argumentos posicionais (sem *args/**kwargs) para índices.")
        return len(sig.parameters)

    def _validate_signature(self, signature, rank):
        if not isinstance(signature, tuple):
            raise TypeError("signature deve ser uma tupla, ex.: (u,d,d).")
        if len(signature) != rank:
            raise ValueError(f"signature tem tamanho {len(signature)}, mas rank={rank}.")
        for s in signature:
            if s is not u and s is not d:
                raise TypeError("signature deve conter APENAS as instâncias u e d (não strings).")
        return signature

    def from_function(self, func, input_signature):
        """
        Constrói um Tensor a partir de func(i,j,k,...) -> expr.
        - rank inferido automaticamente (nº de args de func)
        - signature deve ser tupla com instâncias (u/d)
        """
        rank = self._infer_rank(func)
        signature = self._validate_signature(input_signature, rank)

        shape = (self.dim,) * rank
        flat = [func(*idx) for idx in itertools.product(range(self.dim), repeat=rank)]
        arr = sp.ImmutableDenseNDimArray(flat, shape)
        return Tensor(arr, self.g, self.g_inv, signature=signature)

    def from_array(self, array, signature):
            """
            Cria um objeto Tensor a partir de um sp.Array ou lista multidimensional existente.
            - array: O objeto contendo os componentes numéricos/simbólicos.
            - signature: Tupla de instâncias (u, d) definindo a natureza dos índices.
            """
            # Converte para Array do SymPy se for lista ou matriz
            if not isinstance(array, (sp.Array, sp.ImmutableDenseNDimArray)):
                array = sp.Array(array)
                
            rank = len(array.shape)
            # Reutiliza sua lógica de validação existente
            signature = self._validate_signature(signature, rank)
            
            # Garante que o array seja imutável para consistência interna do framework
            if not isinstance(array, sp.ImmutableDenseNDimArray):
                array = sp.ImmutableDenseNDimArray(array)
                
            return Tensor(array, self.g, self.g_inv, signature=signature)

    def generic(self, name, signature, coords):
        """
        Tensor genérico: componentes são Function(f"{name}{i...}")(coords)
        """
        rank = len(signature)
        signature = self._validate_signature(signature, rank)

        shape = (self.dim,) * rank

        def comp(*idx):
            suf = ''.join(map(str, idx))
            return sp.Function(f"{name}{suf}")(*coords)

        flat = [comp(*idx) for idx in itertools.product(range(self.dim), repeat=rank)]
        arr = sp.ImmutableDenseNDimArray(flat, shape)
        return Tensor(arr, self.g, self.g_inv, signature=signature)



def euler_lagrange_high_order(L, function, variable, order=2):
    """
    Gera a equação de Euler-Lagrange para ordens superiores.
    L: Expressão da Lagrangiana
    function: A função dependente (ex: phi(r))
    variable: A variável independente (ex: r)
    order: A ordem máxima da derivada presente em L
    """
    equation = 0
    
    for i in range(order + 1):
        # 1. Calcula a derivada parcial de L em relação à derivada de ordem 'i' da função
        # Se i=0 -> dL/d(phi), se i=1 -> dL/d(phi') ...
        derivative_of_fn = function.diff(variable, i)
        partial_L = sp.diff(L, derivative_of_fn)
        
        # 2. Aplica a derivada total temporal/espacial (d^i / dr^i)
        total_derivative = sp.diff(partial_L, variable, i)
        
        # 3. O sinal alterna: (-1)^i
        equation += (-1)**i * total_derivative
        
    return sp.simplify(equation)