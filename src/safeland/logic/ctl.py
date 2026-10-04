"""Computation Tree Logic: parser and explicit-state model checker.

Base operators: atoms, !, &, |, EX, E[φ U ψ], EG. Derived operators are
rewritten on parsing:

    AX φ = !EX !φ          EF φ = E[true U φ]        AG φ = !EF !φ
    AF φ = !EG !φ          A[φ U ψ] = !E[!ψ U (!φ & !ψ)] & !EG !ψ

Model checking works on a Kripke structure given as a sparse adjacency
matrix. EX is one sparse mat-vec; E[φ U ψ] is a least fixed point and EG φ a
greatest fixed point, each computed by vectorised iteration (O(d·|E|) where d
is the number of iterations, bounded by the longest simple path; on the
battery-acyclic UAV models d <= battery + monitor depth).

Grammar::

    φ ::= true | false | atom | !φ | φ & φ | φ | φ | φ -> φ | (φ)
        | AX φ | EX φ | AF φ | EF φ | AG φ | EG φ | A[φ U φ] | E[φ U φ]
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CTrue:
    def __str__(self) -> str:
        return "true"


@dataclass(frozen=True)
class CAp:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CNot:
    arg: CFormula

    def __str__(self) -> str:
        return f"!{self.arg}"


@dataclass(frozen=True)
class CAnd:
    left: CFormula
    right: CFormula

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class COr:
    left: CFormula
    right: CFormula

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class EX:
    arg: CFormula

    def __str__(self) -> str:
        return f"EX {self.arg}"


@dataclass(frozen=True)
class EU:
    left: CFormula
    right: CFormula

    def __str__(self) -> str:
        return f"E[{self.left} U {self.right}]"


@dataclass(frozen=True)
class EG:
    arg: CFormula

    def __str__(self) -> str:
        return f"EG {self.arg}"


CFormula = Union[CTrue, CAp, CNot, CAnd, COr, EX, EU, EG]


def c_not(f: CFormula) -> CFormula:
    return f.arg if isinstance(f, CNot) else CNot(f)


def AX(f: CFormula) -> CFormula:
    return c_not(EX(c_not(f)))


def EF(f: CFormula) -> CFormula:
    return EU(CTrue(), f)


def AG(f: CFormula) -> CFormula:
    return c_not(EF(c_not(f)))


def AF(f: CFormula) -> CFormula:
    return c_not(EG(c_not(f)))


def AU(left: CFormula, right: CFormula) -> CFormula:
    nr = c_not(right)
    return CAnd(c_not(EU(nr, CAnd(c_not(left), nr))), c_not(EG(nr)))


# ---------------------------------------------------------------- parser
_TOKEN = re.compile(
    r"\s*(?:(?P<path>[AE]\[)|(?P<unary>AX|EX|AF|EF|AG|EG)|(?P<until>U)|(?P<op>->|[!&|()\]])"
    r"|(?P<const>true|false)(?![a-z0-9_])|(?P<atom>[a-z_][a-z0-9_]*))"
)


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks: list[tuple[str, str]] = []
        pos, s = 0, text.rstrip()
        while pos < len(s):
            m = _TOKEN.match(s, pos)
            if m is None or m.end() == pos:
                raise SyntaxError(f"unexpected input at {pos}: {s[pos:pos + 15]!r}")
            assert m.lastgroup is not None
            self.toks.append((m.lastgroup, m.group(m.lastgroup)))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None):
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise SyntaxError(f"expected {value or 'token'} in {self.text!r}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> CFormula:
        f = self.implies()
        if self.peek() is not None:
            raise SyntaxError(f"trailing input {self.peek()} in {self.text!r}")
        return f

    def implies(self) -> CFormula:
        left = self.disj()
        if self.peek() == ("op", "->"):
            self.take()
            return COr(c_not(left), self.implies())
        return left

    def disj(self) -> CFormula:
        f = self.conj()
        while self.peek() == ("op", "|"):
            self.take()
            f = COr(f, self.conj())
        return f

    def conj(self) -> CFormula:
        f = self.unary()
        while self.peek() == ("op", "&"):
            self.take()
            f = CAnd(f, self.unary())
        return f

    def unary(self) -> CFormula:
        kind, val = self.peek() or ("", "")
        if (kind, val) == ("op", "!"):
            self.take()
            return c_not(self.unary())
        if kind == "unary":
            self.take()
            arg = self.unary()
            return {"AX": AX, "EX": EX, "AF": AF, "EF": EF, "AG": AG, "EG": EG}[val](arg)
        if kind == "path":
            self.take()
            left = self.implies()
            self.take("U")
            right = self.implies()
            self.take("]")
            return AU(left, right) if val[0] == "A" else EU(left, right)
        return self.primary()

    def primary(self) -> CFormula:
        kind, val = self.take()
        if (kind, val) == ("op", "("):
            f = self.implies()
            self.take(")")
            return f
        if kind == "const":
            return CTrue() if val == "true" else CNot(CTrue())
        if kind == "atom":
            return CAp(val)
        raise SyntaxError(f"unexpected {val!r} in {self.text!r}")


def parse_ctl(text: str) -> CFormula:
    return _Parser(text).parse()


# ---------------------------------------------------------------- Kripke + checker
class Kripke:
    """Finite Kripke structure: total transition relation + atom valuation."""

    def __init__(self, adjacency: sp.spmatrix, atoms: Mapping[str, np.ndarray]):
        A = sp.csr_matrix(adjacency, dtype=np.float64, copy=True)  # never alias caller data
        A.sum_duplicates()
        A.data[:] = 1.0
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("adjacency must be square")
        if np.any(np.diff(A.indptr) == 0):
            raise ValueError("transition relation must be total (every state needs a successor)")
        for k, v in atoms.items():
            if np.shape(v) != (n,):
                raise ValueError(f"atom {k!r} has wrong shape")
        self.A = A
        self.n = n
        self.atoms = {k: np.asarray(v, dtype=bool) for k, v in atoms.items()}

    @classmethod
    def from_successors(cls, succ: list[list[int]], atoms: Mapping[str, np.ndarray]) -> Kripke:
        rows = [i for i, ss in enumerate(succ) for _ in ss]
        cols = [j for ss in succ for j in ss]
        n = len(succ)
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        return cls(A, atoms)

    def ex(self, S: np.ndarray) -> np.ndarray:
        return (self.A @ S.astype(np.float64)) > 0.0


def check(f: CFormula, K: Kripke) -> np.ndarray:
    """Set of states satisfying ``f`` as a boolean vector."""
    cache: dict[CFormula, np.ndarray] = {}

    def sat(g: CFormula) -> np.ndarray:
        if g in cache:
            return cache[g]
        if isinstance(g, CTrue):
            r = np.ones(K.n, dtype=bool)
        elif isinstance(g, CAp):
            if g.name not in K.atoms:
                raise KeyError(f"unknown atom {g.name!r}; known: {sorted(K.atoms)}")
            r = K.atoms[g.name]
        elif isinstance(g, CNot):
            r = ~sat(g.arg)
        elif isinstance(g, CAnd):
            r = sat(g.left) & sat(g.right)
        elif isinstance(g, COr):
            r = sat(g.left) | sat(g.right)
        elif isinstance(g, EX):
            r = K.ex(sat(g.arg))
        elif isinstance(g, EU):
            left, r = sat(g.left), sat(g.right).copy()
            while True:  # least fixed point Z = ψ ∨ (φ ∧ EX Z)
                new = r | (left & K.ex(r))
                if np.array_equal(new, r):
                    break
                r = new
        elif isinstance(g, EG):
            r = sat(g.arg).copy()
            while True:  # greatest fixed point Z = φ ∧ EX Z
                new = r & K.ex(r)
                if np.array_equal(new, r):
                    break
                r = new
        else:
            raise TypeError(f"not a CTL formula: {g!r}")
        cache[g] = r
        return r

    return sat(f)
