"""Shield synthesis from the LTL safety specification.

Sure shield (safety game, wind is adversarial)
    W_0     = { v : v not bad }
    W_{i+1} = { v in W_i : exists a, every successor of (v, a) is in W_i }
    W       = greatest fixed point;  allowed(v) = { a : succ(v, a) ⊆ W }.

    Theorem. If the run starts in W and every action is drawn from
    ``allowed``, every reachable product state lies in W, hence no bad state is
    ever visited and the LTL safety specification holds on every run, for any
    learning policy, any exploration noise and any gust sequence.
    Proof: W ∩ bad = ∅ and W is closed under allowed actions (induction on time).
    W is the largest set with this property, so the shield is maximally
    permissive among shields that guarantee sure safety.

    :func:`check_certificate` re-checks both proof obligations on the computed
    arrays, so the guarantee does not rest on trusting the fixed-point code.

Probabilistic shield (Jansen et al., CONCUR 2020 style)
    V(v) = max probability of never reaching a bad state (greatest fixed point),
    Q(v, a) = sum_k P_k V(next_k);  allowed(v) = { a : Q(v,a) >= λ max_a' Q(v,a') }.
    It is more permissive than the sure shield but only bounds the risk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .product import Product


@dataclass(frozen=True, eq=False)
class Shield:
    kind: str
    allowed: np.ndarray  # (n, A) bool, each row has at least one True
    winning: np.ndarray  # (n,) bool: states where the shield's guarantee applies
    safety_value: np.ndarray  # (n,) max probability of staying safe forever
    iterations: int
    synthesis_time: float
    info: dict = field(default_factory=dict)


def _support(product: Product) -> np.ndarray:
    return product.prob > 0.0


def max_safety_probability(product: Product, tol: float = 1e-12, max_iter: int = 1_000_000):
    """Greatest fixed point of V = max_a sum_k P V(next) on non-bad states, V = 0 on bad."""
    V = (~product.bad).astype(np.float64)
    for it in range(1, max_iter + 1):
        Q = np.einsum("nak,nak->na", product.prob, V[product.next])
        V_new = np.where(product.bad, 0.0, Q.max(axis=1))
        if np.max(np.abs(V_new - V)) <= tol:
            V = V_new
            break
        V = V_new
    else:  # pragma: no cover - cannot happen on the battery-acyclic world
        raise RuntimeError("safety value iteration did not converge")
    Q = np.einsum("nak,nak->na", product.prob, V[product.next])
    return V, Q, it


def _fallback_rows(allowed: np.ndarray, Q: np.ndarray, rows: np.ndarray) -> None:
    """Outside the guarantee region, keep the safest actions (never an empty row)."""
    if rows.any():
        best = Q[rows] >= Q[rows].max(axis=1, keepdims=True) - 1e-12
        allowed[rows] = best


def no_shield(product: Product) -> Shield:
    n = product.n_states
    return Shield(
        kind="none",
        allowed=np.ones((n, product.n_actions), dtype=bool),
        winning=np.ones(n, dtype=bool),
        safety_value=np.ones(n),
        iterations=0,
        synthesis_time=0.0,
    )


def synthesize_sure(product: Product, tol: float = 1e-12) -> Shield:
    t0 = time.perf_counter()
    sup = _support(product)
    W = ~product.bad
    it = 0
    while True:
        it += 1
        ok = np.all(W[product.next] | ~sup, axis=2)  # (n, A): action keeps all successors in W
        W_new = W & ok.any(axis=1)
        if np.array_equal(W_new, W):
            break
        W = W_new
    allowed = np.all(W[product.next] | ~sup, axis=2) & W[:, None]
    V, Q, _ = max_safety_probability(product, tol)
    _fallback_rows(allowed, Q, ~W)
    return Shield("sure", allowed, W, V, it, time.perf_counter() - t0)


def synthesize_prob(product: Product, lam: float, tol: float = 1e-12) -> Shield:
    if not 0.0 < lam <= 1.0:
        raise ValueError("lambda must be in (0, 1]")
    t0 = time.perf_counter()
    V, Q, it = max_safety_probability(product, tol)
    allowed = Q >= lam * Q.max(axis=1, keepdims=True) - 1e-12
    winning = V >= 1.0 - 1e-9
    return Shield("prob", allowed, winning, V, it, time.perf_counter() - t0, {"lambda": lam})


def build_shield(product: Product, method: str, lam: float = 0.95, tol: float = 1e-12) -> Shield:
    if method == "unshielded":
        return no_shield(product)
    if method == "sure_shield":
        return synthesize_sure(product, tol)
    if method == "prob_shield":
        return synthesize_prob(product, lam, tol)
    raise ValueError(f"unknown method {method!r}")


@dataclass(frozen=True)
class Certificate:
    ok: bool
    winning_size: int
    bad_in_winning: int
    empty_rows: int
    escaping_pairs: int
    example: str = ""


def check_certificate(product: Product, shield: Shield) -> Certificate:
    """Independently verify the two proof obligations of the sure-shield theorem.

    (1) W ∩ bad = ∅;  (2) for every v in W: allowed(v) ≠ ∅ and every successor
    (with positive probability) of every allowed action lies in W.
    """
    W = shield.winning
    sup = _support(product)
    bad_in_w = int(np.count_nonzero(W & product.bad))
    empty = int(np.count_nonzero(W & ~shield.allowed.any(axis=1)))
    escapes = shield.allowed[:, :, None] & sup & ~W[product.next] & W[:, None, None]
    n_esc = int(np.count_nonzero(escapes))
    example = ""
    if n_esc:
        v, a, k = (int(i) for i in np.argwhere(escapes)[0])
        example = (
            f"state {product.describe(v)} action {a} -> {product.describe(int(product.next[v, a, k]))}"
        )
    return Certificate(
        ok=bad_in_w == 0 and empty == 0 and n_esc == 0,
        winning_size=int(W.sum()),
        bad_in_winning=bad_in_w,
        empty_rows=empty,
        escaping_pairs=n_esc,
        example=example,
    )


def battery_reserve_map(product: Product, shield: Shield, z: int) -> np.ndarray:
    """Minimum battery from which the shield can guarantee safety, per cell at altitude ``z``.

    NaN marks cells that are not flyable at that altitude or not safe at any battery level.
    """
    world = product.world
    out = np.full((world.W, world.D), np.nan)
    for x, y in world.flying_cells(z):
        for b in range(1, world.B + 1):
            if shield.winning[product.initial(world.encode(x, y, z, b))]:
                out[x, y] = b
                break
    return out
