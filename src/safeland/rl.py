"""Tabular Q-learning on the product MDP with (optional) preemptive shielding.

The shield removes unsafe actions *before* the agent chooses (preemptive
shielding, Alshiekh et al., AAAI 2018), both for exploration and for the
bootstrap target max_{a' in allowed(v')} Q(v', a'). With the sure shield the
agent therefore never executes an unsafe action, not even while exploring.

No dataset is used: all experience is generated online by the simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import RLConfig
from .product import Product
from .shield import Shield


@dataclass(frozen=True, eq=False)
class TrainResult:
    Q: np.ndarray
    returns: np.ndarray
    violated: np.ndarray  # bool per episode: LTL safety spec violated
    landed: np.ndarray  # bool per episode: reached a pad safely
    steps: np.ndarray
    start_in_winning: np.ndarray  # bool per episode
    train_time: float


def epsilon_at(episode: int, cfg: RLConfig) -> float:
    horizon = max(1, int(cfg.episodes * cfg.eps_decay_frac))
    frac = min(1.0, episode / horizon)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _greedy(q_row: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None) -> int:
    vals = np.where(mask, q_row, -np.inf)
    best = np.flatnonzero(vals == vals.max())
    if rng is None or best.size == 1:
        return int(best[0])
    return int(rng.choice(best))


def train_q_learning(
    product: Product,
    shield: Shield,
    cfg: RLConfig,
    start_states: np.ndarray,
    rng: np.random.Generator,
) -> TrainResult:
    """Train with epsilon-greedy exploration restricted to ``shield.allowed``."""
    n, A = product.n_states, product.n_actions
    Q = np.zeros((n, A))
    allowed = shield.allowed
    nxt, prob = product.next, product.prob
    bad, terminal, landed = product.bad, product.terminal, product.landed
    max_steps = product.world.B + 1  # every action drains >= 1 battery unit
    E = cfg.episodes
    returns = np.zeros(E)
    violated = np.zeros(E, dtype=bool)
    reached = np.zeros(E, dtype=bool)
    steps = np.zeros(E, dtype=np.int64)
    start_in_w = np.zeros(E, dtype=bool)
    t0 = time.perf_counter()
    for ep in range(E):
        eps = epsilon_at(ep, cfg)
        v = int(product.initial(int(rng.choice(start_states))))
        start_in_w[ep] = shield.winning[v]
        ret = 0.0
        for t in range(max_steps):
            if terminal[v]:
                break
            mask = allowed[v]
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a = _greedy(Q[v], mask, rng)
            v2 = int(nxt[v, a, 0] if rng.random() < prob[v, a, 0] else nxt[v, a, 1])
            r = -cfg.step_cost
            if bad[v2]:
                r -= cfg.penalty_violation
                violated[ep] = True
            elif landed[v2]:
                r += cfg.reward_land
                reached[ep] = True
            if terminal[v2]:
                target = r
            else:
                target = r + cfg.gamma * np.max(np.where(allowed[v2], Q[v2], -np.inf))
            Q[v, a] += cfg.alpha * (target - Q[v, a])
            ret += r
            steps[ep] = t + 1
            v = v2
        returns[ep] = ret
    return TrainResult(Q, returns, violated, reached, steps, start_in_w, time.perf_counter() - t0)


def greedy_policy(Q: np.ndarray, shield: Shield) -> np.ndarray:
    """Deterministic deployment policy: best shield-allowed action, lowest index on ties."""
    vals = np.where(shield.allowed, Q, -np.inf)
    return np.argmax(vals, axis=1).astype(np.int64)
