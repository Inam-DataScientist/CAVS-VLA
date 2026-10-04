"""Post-training certification of a deployed policy.

Given a deterministic policy over product states, the closed loop is a DTMC
(gust probabilities) whose support graph is a Kripke structure (gusts as
nondeterminism). We then

* model-check CTL formulas on the Kripke structure (worst case over gusts),
* compute PCTL probabilities and expected times exactly on the DTMC,
* cross-check one probability by statistical model checking that simulates
  the *world* with a table-free sampler and progresses the *LTL formula*
  directly, sharing no transition tables with the exact computation,
* return a shortest counterexample path when ``AG !spec_violated`` fails.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import stats

from .logic import ctl, pctl
from .logic.ltl import Bot, progress
from .product import Product


def closed_loop(product: Product, policy: np.ndarray, init: np.ndarray):
    """DTMC + Kripke structure of the policy, restricted to states reachable from ``init``."""
    n = product.n_states
    rows = np.arange(n)
    succ = product.next[rows, policy]  # (n, 2)
    prob = product.prob[rows, policy]
    # reachability over the support graph
    reach = np.zeros(n, dtype=bool)
    reach[init] = True
    frontier = np.unique(init)
    while frontier.size:
        nxt = succ[frontier][prob[frontier] > 0]
        nxt = np.unique(nxt[~reach[nxt]])
        reach[nxt] = True
        frontier = nxt
    idx = np.flatnonzero(reach)
    remap = -np.ones(n, dtype=np.int64)
    remap[idx] = np.arange(idx.size)
    r = np.repeat(np.arange(idx.size), 2)
    c = remap[succ[idx].ravel()]
    p = prob[idx].ravel()
    keep = p > 0
    m = idx.size
    P = sp.csr_matrix((p[keep], (r[keep], c[keep])), shape=(m, m))
    P.sum_duplicates()
    atoms = {k: v[idx] for k, v in product.atoms().items()}
    K = ctl.Kripke(P, atoms)
    return P, K, idx, remap


def random_policy_baseline(product: Product, allowed: np.ndarray) -> dict:
    """Exact metrics of the policy that picks uniformly among ``allowed`` actions.

    Separates what the shield contributes (safety) from what learning contributes
    (efficiency): a shield alone can already be safe, but it does not land quickly.
    """
    n, A = product.n_states, product.n_actions
    w = allowed / allowed.sum(axis=1, keepdims=True)
    rows = np.repeat(np.arange(n), A * 2)
    vals = (w[:, :, None] * product.prob).ravel()
    keep = vals > 0
    P = sp.csr_matrix((vals[keep], (rows[keep], product.next.ravel()[keep])), shape=(n, n))
    P.sum_duplicates()
    atoms = product.atoms()
    init = product.initial_states()
    ps = pctl.prob_until(P, ~atoms["spec_violated"], atoms["landed"])[init]
    es = pctl.expected_steps(P, atoms["terminal"])[init]
    return {
        "P_safe_landing": float(ps.mean()),
        "P_safe_landing_min_start": float(ps.min()),
        "expected_steps": float(es.mean()),
    }


@dataclass(frozen=True)
class SmcResult:
    estimate: float
    ci_low: float
    ci_high: float
    successes: int
    samples: int


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def smc_safe_landing(
    product: Product,
    policy: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> SmcResult:
    """Monte-Carlo estimate of P[!spec_violated U landed] from a uniform random start."""
    world, monitor = product.world, product.monitor
    starts = world.start_states
    success = 0
    for _ in range(n):
        s = int(rng.choice(starts))
        f = progress(monitor.formula, world.label_of(s))
        while True:
            if isinstance(f, Bot):
                break
            if s == world.sink["landed"]:
                success += 1
                break
            if s >= world.n_fly:
                break
            v = s * product.nq + monitor.index[f]
            s = world.sample_direct(s, int(policy[v]), rng)
            f = progress(f, world.label_of(s))
    lo, hi = clopper_pearson(success, n)
    return SmcResult(success / n, lo, hi, success, n)


def counterexample(
    product: Product, policy: np.ndarray, init: np.ndarray, max_len: int = 200
) -> list[str]:
    """Shortest gust sequence leading the closed loop from ``init`` to a bad state."""
    rows = np.arange(product.n_states)
    succ = product.next[rows, policy]
    prob = product.prob[rows, policy]
    parent: dict[int, int] = {}
    q = deque(int(v) for v in np.unique(init))
    seen = set(q)
    goal = None
    while q:
        v = q.popleft()
        if product.bad[v]:
            goal = v
            break
        for k in range(2):
            if prob[v, k] > 0:
                w = int(succ[v, k])
                if w not in seen:
                    seen.add(w)
                    parent[w] = v
                    q.append(w)
    if goal is None:
        return []
    path = [goal]
    while path[-1] in parent and len(path) < max_len:
        path.append(parent[path[-1]])
    from .world import ACTIONS

    out = []
    for v in reversed(path):
        act = "" if product.terminal[v] else f"  --{ACTIONS[int(policy[v])]}-->"
        out.append(product.describe(v) + act)
    return out


def certify(
    product: Product,
    policy: np.ndarray,
    ctl_specs: tuple[str, ...],
    pctl_specs: tuple[str, ...],
) -> dict:
    """Exact CTL/PCTL results over the uniform start distribution."""
    init = product.initial_states()
    P, K, idx, remap = closed_loop(product, policy, init)
    loc = remap[init]
    out: dict = {"closed_loop_states": int(idx.size), "ctl": {}, "pctl": {}}
    for text in ctl_specs:
        sat = ctl.check(ctl.parse_ctl(text), K)[loc]
        out["ctl"][text] = {"fraction_initial_satisfied": float(sat.mean()), "all": bool(sat.all())}
    for text in pctl_specs:
        vals = pctl.evaluate(text, P, K)[loc]
        finite = vals[np.isfinite(vals)]
        out["pctl"][text] = {
            "mean": float(vals.mean()) if np.all(np.isfinite(vals)) else float("inf"),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean_finite": float(finite.mean()) if finite.size else float("nan"),
            "fraction_finite": float(np.isfinite(vals).mean()),
        }
    return out
