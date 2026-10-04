"""Linear Temporal Logic: syntax, parser, progression and monitor construction.

Formulas are kept in negation normal form (negation only on atoms) and built
with *smart constructors* that simplify on the fly, so syntactically different
but trivially equal obligations collapse to one object.

Monitor construction uses formula progression (Bacchus & Kabanza, 2000):

    prog(p, σ)        = true if p ∈ σ else false
    prog(X φ, σ)      = φ
    prog(G φ, σ)      = prog(φ, σ) ∧ G φ
    prog(F φ, σ)      = prog(φ, σ) ∨ F φ
    prog(φ U ψ, σ)    = prog(ψ, σ) ∨ (prog(φ, σ) ∧ φ U ψ)
    prog(φ R ψ, σ)    = prog(ψ, σ) ∧ (prog(φ, σ) ∨ φ R ψ)
    prog(F≤k φ, σ)    = prog(φ, σ) ∨ F≤k-1 φ          (F≤0 φ ≡ φ)
    prog(G≤k φ, σ)    = prog(φ, σ) ∧ G≤k-1 φ          (G≤0 φ ≡ φ)

A finite trace σ0…σt is a bad prefix of φ when progression yields ``false``.
For the safety specifications used here (Boolean combinations of G, G≤k, F≤k
and X over atoms) progression detects every bad prefix as soon as it occurs.
Exploring progression over a finite alphabet of labels yields a deterministic
monitor automaton; the bounded operators are merged (keep the tightest
deadline), which keeps that automaton finite and small.

Grammar (operators upper-case, atoms lower-case)::

    φ ::= true | false | atom | !φ | φ & φ | φ | φ | φ -> φ | φ <-> φ
        | X φ | G φ | F φ | G[<=k] φ | F[<=k] φ | φ U φ | φ R φ | (φ)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np


# ---------------------------------------------------------------- syntax
@dataclass(frozen=True)
class Top:
    def __str__(self) -> str:
        return "true"


@dataclass(frozen=True)
class Bot:
    def __str__(self) -> str:
        return "false"


@dataclass(frozen=True)
class Ap:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class NotAp:
    name: str

    def __str__(self) -> str:
        return f"!{self.name}"


@dataclass(frozen=True)
class And:
    args: frozenset

    def __str__(self) -> str:
        return "(" + " & ".join(sorted(str(a) for a in self.args)) + ")"


@dataclass(frozen=True)
class Or:
    args: frozenset

    def __str__(self) -> str:
        return "(" + " | ".join(sorted(str(a) for a in self.args)) + ")"


@dataclass(frozen=True)
class Next:
    arg: Formula

    def __str__(self) -> str:
        return f"X {self.arg}"


@dataclass(frozen=True)
class Always:
    arg: Formula

    def __str__(self) -> str:
        return f"G {self.arg}"


@dataclass(frozen=True)
class Eventually:
    arg: Formula

    def __str__(self) -> str:
        return f"F {self.arg}"


@dataclass(frozen=True)
class Until:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"


@dataclass(frozen=True)
class Release:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} R {self.right})"


@dataclass(frozen=True)
class BEventually:
    bound: int
    arg: Formula

    def __str__(self) -> str:
        return f"F[<={self.bound}] {self.arg}"


@dataclass(frozen=True)
class BAlways:
    bound: int
    arg: Formula

    def __str__(self) -> str:
        return f"G[<={self.bound}] {self.arg}"


Formula = Union[
    Top, Bot, Ap, NotAp, And, Or, Next, Always, Eventually, Until, Release, BEventually, BAlways
]
TRUE = Top()
FALSE = Bot()


# ---------------------------------------------------------------- smart constructors
def _merge_bounded(items: set, cls: type, keep) -> set:
    best: dict = {}
    rest = set()
    for it in items:
        if isinstance(it, cls):
            best[it.arg] = keep(best.get(it.arg, it.bound), it.bound)
        else:
            rest.add(it)
    return rest | {cls(k, f) for f, k in best.items()}


def conj(*xs: Formula) -> Formula:
    flat: set = set()
    for x in xs:
        if isinstance(x, Bot):
            return FALSE
        if isinstance(x, Top):
            continue
        flat.update(x.args if isinstance(x, And) else (x,))
    pos = {x.name for x in flat if isinstance(x, Ap)}
    neg = {x.name for x in flat if isinstance(x, NotAp)}
    if pos & neg:
        return FALSE
    flat = _merge_bounded(flat, BEventually, min)  # F<=a φ ∧ F<=b φ ≡ F<=min φ
    flat = _merge_bounded(flat, BAlways, max)  # G<=a φ ∧ G<=b φ ≡ G<=max φ
    if not flat:
        return TRUE
    return next(iter(flat)) if len(flat) == 1 else And(frozenset(flat))


def disj(*xs: Formula) -> Formula:
    flat: set = set()
    for x in xs:
        if isinstance(x, Top):
            return TRUE
        if isinstance(x, Bot):
            continue
        flat.update(x.args if isinstance(x, Or) else (x,))
    pos = {x.name for x in flat if isinstance(x, Ap)}
    neg = {x.name for x in flat if isinstance(x, NotAp)}
    if pos & neg:
        return TRUE
    flat = _merge_bounded(flat, BEventually, max)
    flat = _merge_bounded(flat, BAlways, min)
    if not flat:
        return FALSE
    return next(iter(flat)) if len(flat) == 1 else Or(frozenset(flat))


def nxt(f: Formula) -> Formula:
    return f if isinstance(f, (Top, Bot)) else Next(f)


def always(f: Formula) -> Formula:
    if isinstance(f, (Top, Bot, Always)):
        return f
    return Always(f)


def eventually(f: Formula) -> Formula:
    if isinstance(f, (Top, Bot, Eventually)):
        return f
    return Eventually(f)


def until(left: Formula, right: Formula) -> Formula:
    if isinstance(right, (Top, Bot)) or isinstance(left, Bot):
        return right
    if isinstance(left, Top):
        return eventually(right)
    return Until(left, right)


def release(left: Formula, right: Formula) -> Formula:
    if isinstance(right, (Top, Bot)) or isinstance(left, Top):
        return right
    if isinstance(left, Bot):
        return always(right)
    return Release(left, right)


def b_eventually(k: int, f: Formula) -> Formula:
    if k < 0:
        raise ValueError("bound must be >= 0")
    if k == 0 or isinstance(f, (Top, Bot)):
        return f
    return BEventually(k, f)


def b_always(k: int, f: Formula) -> Formula:
    if k < 0:
        raise ValueError("bound must be >= 0")
    if k == 0 or isinstance(f, (Top, Bot)):
        return f
    return BAlways(k, f)


def negate(f: Formula) -> Formula:
    """Push a negation through ``f`` (result stays in negation normal form)."""
    if isinstance(f, Top):
        return FALSE
    if isinstance(f, Bot):
        return TRUE
    if isinstance(f, Ap):
        return NotAp(f.name)
    if isinstance(f, NotAp):
        return Ap(f.name)
    if isinstance(f, And):
        return disj(*(negate(a) for a in f.args))
    if isinstance(f, Or):
        return conj(*(negate(a) for a in f.args))
    if isinstance(f, Next):
        return nxt(negate(f.arg))
    if isinstance(f, Always):
        return eventually(negate(f.arg))
    if isinstance(f, Eventually):
        return always(negate(f.arg))
    if isinstance(f, Until):
        return release(negate(f.left), negate(f.right))
    if isinstance(f, Release):
        return until(negate(f.left), negate(f.right))
    if isinstance(f, BEventually):
        return b_always(f.bound, negate(f.arg))
    if isinstance(f, BAlways):
        return b_eventually(f.bound, negate(f.arg))
    raise TypeError(f"not a formula: {f!r}")


def atoms_of(f: Formula) -> frozenset[str]:
    if isinstance(f, (Ap, NotAp)):
        return frozenset({f.name})
    if isinstance(f, (And, Or)):
        return frozenset().union(*(atoms_of(a) for a in f.args))
    if isinstance(f, (Next, Always, Eventually, BEventually, BAlways)):
        return atoms_of(f.arg)
    if isinstance(f, (Until, Release)):
        return atoms_of(f.left) | atoms_of(f.right)
    return frozenset()


# ---------------------------------------------------------------- progression
@lru_cache(maxsize=None)
def progress(f: Formula, sigma: frozenset[str]) -> Formula:
    """Obligation that the rest of the trace must satisfy after reading ``sigma``."""
    if isinstance(f, (Top, Bot)):
        return f
    if isinstance(f, Ap):
        return TRUE if f.name in sigma else FALSE
    if isinstance(f, NotAp):
        return FALSE if f.name in sigma else TRUE
    if isinstance(f, And):
        return conj(*(progress(a, sigma) for a in f.args))
    if isinstance(f, Or):
        return disj(*(progress(a, sigma) for a in f.args))
    if isinstance(f, Next):
        return f.arg
    if isinstance(f, Always):
        return conj(progress(f.arg, sigma), f)
    if isinstance(f, Eventually):
        return disj(progress(f.arg, sigma), f)
    if isinstance(f, Until):
        return disj(progress(f.right, sigma), conj(progress(f.left, sigma), f))
    if isinstance(f, Release):
        return conj(progress(f.right, sigma), disj(progress(f.left, sigma), f))
    if isinstance(f, BEventually):
        return disj(progress(f.arg, sigma), b_eventually(f.bound - 1, f.arg))
    if isinstance(f, BAlways):
        return conj(progress(f.arg, sigma), b_always(f.bound - 1, f.arg))
    raise TypeError(f"not a formula: {f!r}")


def progress_trace(f: Formula, trace: Iterable[frozenset[str]]) -> Formula:
    for sigma in trace:
        f = progress(f, sigma)
    return f


# ---------------------------------------------------------------- parser
_TOKEN = re.compile(
    r"\s*(?:(?P<bound>[FG]\[\s*<=\s*\d+\s*\])|(?P<op><->|->|[!&|()])"
    r"|(?P<const>true|false)(?![a-z0-9_])|(?P<temporal>[XGFUR])|(?P<atom>[a-z_][a-z0-9_]*))"
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise SyntaxError(f"unexpected input at {pos}: {text[pos:pos + 15]!r}")
        kind = m.lastgroup
        assert kind is not None
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise SyntaxError(f"expected {value or 'token'} in {self.text!r}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> Formula:
        f = self.iff()
        if self.peek() is not None:
            raise SyntaxError(f"trailing input {self.peek()} in {self.text!r}")
        return f

    def iff(self) -> Formula:
        left = self.implies()
        if self.peek() == ("op", "<->"):
            self.take()
            right = self.iff()
            return conj(disj(negate(left), right), disj(negate(right), left))
        return left

    def implies(self) -> Formula:
        left = self.disj()
        if self.peek() == ("op", "->"):
            self.take()
            return disj(negate(left), self.implies())
        return left

    def disj(self) -> Formula:
        parts = [self.conj()]
        while self.peek() == ("op", "|"):
            self.take()
            parts.append(self.conj())
        return disj(*parts)

    def conj(self) -> Formula:
        parts = [self.binary()]
        while self.peek() == ("op", "&"):
            self.take()
            parts.append(self.binary())
        return conj(*parts)

    def binary(self) -> Formula:
        left = self.unary()
        tok = self.peek()
        if tok in (("temporal", "U"), ("temporal", "R")):
            self.take()
            right = self.binary()
            return until(left, right) if tok[1] == "U" else release(left, right)
        return left

    def unary(self) -> Formula:
        tok = self.peek()
        if tok is None:
            raise SyntaxError(f"unexpected end of {self.text!r}")
        kind, val = tok
        if (kind, val) == ("op", "!"):
            self.take()
            return negate(self.unary())
        if kind == "temporal" and val in "XGF":
            self.take()
            arg = self.unary()
            return {"X": nxt, "G": always, "F": eventually}[val](arg)
        if kind == "bound":
            self.take()
            k = int(re.search(r"\d+", val).group())  # type: ignore[union-attr]
            arg = self.unary()
            return b_eventually(k, arg) if val[0] == "F" else b_always(k, arg)
        return self.primary()

    def primary(self) -> Formula:
        kind, val = self.take()
        if (kind, val) == ("op", "("):
            f = self.iff()
            self.take(")")
            return f
        if kind == "const":
            return TRUE if val == "true" else FALSE
        if kind == "atom":
            return Ap(val)
        raise SyntaxError(f"unexpected {val!r} in {self.text!r}")


def parse_ltl(text: str) -> Formula:
    return _Parser(text).parse()


# ---------------------------------------------------------------- monitor
class Monitor:
    """Deterministic monitor obtained by exploring progression over an alphabet.

    ``states[0]`` is the original formula (nothing read yet); ``delta[q, l]`` is
    the index of ``progress(states[q], alphabet[l])``. A monitor state equal to
    ``false`` means the specification has been violated irrevocably.
    """

    def __init__(
        self,
        formula: Formula,
        alphabet: Sequence[frozenset[str]],
        max_states: int = 10_000,
    ):
        known = frozenset().union(*alphabet) if alphabet else frozenset()
        unknown = atoms_of(formula) - known
        if unknown:
            raise ValueError(f"specification uses unknown atoms {sorted(unknown)}")
        self.formula = formula
        self.alphabet = tuple(alphabet)
        states: list[Formula] = [formula]
        index: dict[Formula, int] = {formula: 0}
        rows: list[list[int]] = []
        q = 0
        while q < len(states):
            row = []
            for sigma in self.alphabet:
                g = progress(states[q], sigma)
                if g not in index:
                    if len(states) >= max_states:
                        raise RuntimeError(f"monitor exceeds {max_states} states")
                    index[g] = len(states)
                    states.append(g)
                row.append(index[g])
            rows.append(row)
            q += 1
        self.states: tuple[Formula, ...] = tuple(states)
        self.index = index
        self.delta = np.asarray(rows, dtype=np.int64)
        self.is_bad = np.array([isinstance(s, Bot) for s in states])
        self.is_true = np.array([isinstance(s, Top) for s in states])

    @property
    def n_states(self) -> int:
        return len(self.states)

    def step(self, q: int, label_index: int) -> int:
        return int(self.delta[q, label_index])

    def describe(self) -> str:
        lines = [f"monitor with {self.n_states} states for {self.formula}"]
        for i, s in enumerate(self.states):
            tag = " [VIOLATED]" if self.is_bad[i] else (" [SATISFIED]" if self.is_true[i] else "")
            lines.append(f"  q{i}: {s}{tag}")
        return "\n".join(lines)
