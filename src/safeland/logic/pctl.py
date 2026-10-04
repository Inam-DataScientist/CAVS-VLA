"""Probabilistic CTL on discrete-time Markov chains (exact, linear algebra).

Supported queries (state formulas inside are CTL formulas, see ``ctl.py``)::

    P=? [ φ U ψ ]     P=? [ φ U<=k ψ ]     P=? [ F ψ ]    P=? [ F<=k ψ ]    P=? [ G φ ]
    R=? [ F ψ ]       expected number of steps until ψ (inf where P(F ψ) < 1)

Unbounded until follows the standard algorithm (Baier & Katoen, Principles of
Model Checking, Sec. 10.1): graph-based precomputation of the states with
probability exactly 0 and exactly 1, then a sparse linear solve on the rest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .ctl import Kripke, check, parse_ctl


def _validate(P: sp.csr_matrix) -> sp.csr_matrix:
    P = sp.csr_matrix(P, dtype=np.float64)
    rows = np.asarray(P.sum(axis=1)).ravel()
    if not np.allclose(rows, 1.0, atol=1e-9):
        raise ValueError("DTMC rows must sum to 1")
    return P


def backward_reach(P: sp.csr_matrix, targets: np.ndarray, through: np.ndarray) -> np.ndarray:
    """States that reach ``targets`` via a path whose intermediate states satisfy ``through``."""
    pattern = P.copy()
    pattern.data[:] = 1.0
    result = targets.astype(bool).copy()
    frontier = result.copy()
    while frontier.any():
        pre = (pattern @ frontier.astype(np.float64)) > 0.0
        new = pre & through & ~result
        result |= new
        frontier = new
    return result


def prob_until(P: sp.csr_matrix, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    P = _validate(P)
    phi, psi = phi.astype(bool), psi.astype(bool)
    path = phi & ~psi
    no = ~backward_reach(P, psi, path)
    yes = ~backward_reach(P, no, path)
    maybe = ~yes & ~no
    x = yes.astype(np.float64)
    if maybe.any():
        idx = np.flatnonzero(maybe)
        A = P[idx][:, idx]
        b = np.asarray(P[idx][:, np.flatnonzero(yes)].sum(axis=1)).ravel()
        M = sp.identity(idx.size, format="csc") - A.tocsc()
        x[idx] = spla.spsolve(M, b)
    return np.clip(x, 0.0, 1.0)


def prob_bounded_until(P: sp.csr_matrix, phi: np.ndarray, psi: np.ndarray, k: int) -> np.ndarray:
    P = _validate(P)
    phi, psi = phi.astype(bool), psi.astype(bool)
    x = psi.astype(np.float64)
    path = phi & ~psi
    for _ in range(k):
        x = np.where(psi, 1.0, np.where(path, P @ x, 0.0))
    return x


def expected_steps(P: sp.csr_matrix, target: np.ndarray) -> np.ndarray:
    P = _validate(P)
    target = target.astype(bool)
    no = ~backward_reach(P, target, ~target)
    yes = ~backward_reach(P, no, ~target)  # reach target with probability 1
    out = np.full(P.shape[0], np.inf)
    out[target] = 0.0
    solve = yes & ~target
    if solve.any():
        idx = np.flatnonzero(solve)
        M = sp.identity(idx.size, format="csc") - P[idx][:, idx].tocsc()
        out[idx] = spla.spsolve(M, np.ones(idx.size))
    return out


@dataclass(frozen=True)
class PctlQuery:
    text: str
    kind: str  # "P" or "R"
    op: str  # "U", "F", "G"
    left: str | None
    right: str
    bound: int | None


_HEAD = re.compile(r"^\s*([PR])\s*=\s*\?\s*\[(.*)\]\s*$", re.S)
_PATH_PREFIX = re.compile(r"^\s*([FG])(?:\s*<=\s*(\d+))?\s+(.*)$", re.S)


def _split_until(body: str) -> tuple[str, str, int | None] | None:
    depth = 0
    for i, ch in enumerate(body):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "U" and depth == 0:
            m = re.match(r"U(?:\s*<=\s*(\d+))?", body[i:])
            assert m is not None
            bound = int(m.group(1)) if m.group(1) else None
            return body[:i].strip(), body[i + m.end():].strip(), bound
    return None


def parse_pctl(text: str) -> PctlQuery:
    m = _HEAD.match(text)
    if not m:
        raise SyntaxError(f"PCTL query must look like P=? [ ... ] or R=? [ F ... ]: {text!r}")
    kind, body = m.group(1), m.group(2).strip()
    pm = _PATH_PREFIX.match(body)
    if pm:
        op, bound, arg = pm.group(1), pm.group(2), pm.group(3).strip()
        q = PctlQuery(text, kind, op, None, arg, int(bound) if bound else None)
    else:
        split = _split_until(body)
        if split is None:
            raise SyntaxError(f"no path operator in {text!r}")
        left, right, bound = split
        q = PctlQuery(text, kind, "U", left, right, bound)
    if kind == "R" and (q.op != "F" or q.bound is not None):
        raise SyntaxError("R=? queries support only unbounded F")
    # validate state formulas eagerly
    parse_ctl(q.right)
    if q.left:
        parse_ctl(q.left)
    return q


def evaluate(query: PctlQuery | str, P: sp.csr_matrix, K: Kripke) -> np.ndarray:
    """Per-state value of the query on DTMC ``P`` whose support graph is ``K``."""
    q = parse_pctl(query) if isinstance(query, str) else query
    right = check(parse_ctl(q.right), K)
    if q.kind == "R":
        return expected_steps(P, right)
    if q.op == "G":
        if q.bound is None:
            return 1.0 - prob_until(P, np.ones(K.n, bool), ~right)
        return 1.0 - prob_bounded_until(P, np.ones(K.n, bool), ~right, q.bound)
    left = np.ones(K.n, bool) if q.op == "F" else check(parse_ctl(q.left or "true"), K)
    if q.bound is None:
        return prob_until(P, left, right)
    return prob_bounded_until(P, left, right, q.bound)
