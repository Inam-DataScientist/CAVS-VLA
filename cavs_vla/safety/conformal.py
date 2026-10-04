"""Split-conformal prediction sets for agent futures.

Score per agent (normalised, horizon-joint):
    s = max_t  ||p_hat_t - p_t|| / sigma_t,   sigma_t = max(mean Laplace scale_t, min_scale)
With q = the ceil((n+1)(1-alpha))-th smallest calibration score (per agent type),
the disc of radius r_t = q * sigma_t around p_hat_t contains the true position for
*all* valid t with probability >= 1 - alpha, for an exchangeable test agent.
Closed-loop interaction breaks exchangeability, so coverage is re-measured in
closed loop and reported (see evaluate.py).
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

TYPE_NAMES = ["vehicle", "pedestrian", "cyclist", "static"]


def scores(pred: np.ndarray, scale: np.ndarray, gt: np.ndarray, valid: np.ndarray, min_scale: float) -> np.ndarray:
    """pred/gt (n, H, 2), scale (n, H, 2) Laplace scales, valid (n, H) -> scores (n,), nan where no valid step."""
    sigma = np.maximum(scale.mean(-1), min_scale)
    err = np.linalg.norm(pred - gt, axis=-1) / sigma
    err = np.where(valid, err, -np.inf)
    s = err.max(-1)
    return np.where(np.isfinite(s), s, np.nan)


def conformal_quantile(s: np.ndarray, alpha: float) -> float:
    s = np.sort(s[np.isfinite(s)])
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def calibrate(score_list: List[np.ndarray], type_list: List[np.ndarray], alpha: float, min_scale: float) -> Dict:
    s = np.concatenate(score_list) if score_list else np.zeros(0)
    ty = np.concatenate(type_list) if type_list else np.zeros(0, dtype=int)
    out = dict(alpha=alpha, min_scale=min_scale, q={}, n={}, q_all=conformal_quantile(s, alpha), n_all=int(np.isfinite(s).sum()))
    for t, name in enumerate(TYPE_NAMES):
        sel = (ty == t) & np.isfinite(s)
        out["n"][name] = int(sel.sum())
        # Types with too few samples fall back to the pooled quantile.
        out["q"][name] = conformal_quantile(s[sel], alpha) if sel.sum() >= 50 else out["q_all"]
    return out


def coverage(score_list: List[np.ndarray], type_list: List[np.ndarray], table: Dict) -> Dict:
    s = np.concatenate(score_list)
    ty = np.concatenate(type_list)
    ok = np.isfinite(s)
    q = np.array([table["q"][n] for n in TYPE_NAMES])
    covered = s[ok] <= q[ty[ok]]
    res = dict(overall=float(covered.mean()) if covered.size else float("nan"), n=int(ok.sum()))
    for t, name in enumerate(TYPE_NAMES):
        sel = ty[ok] == t
        res[name] = float(covered[sel].mean()) if sel.any() else float("nan")
    return res


def save(table: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(table, f, indent=2)


def load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def radius_torch(table: Dict, agent_logscale, agent_type):
    """Per-agent, per-step radius (B, N, H) from predicted log-scales (B, N, H, 2) and types (B, N)."""
    import torch

    q = torch.tensor([table["q"][n] for n in TYPE_NAMES], device=agent_logscale.device, dtype=torch.float32)
    q = torch.nan_to_num(q, posinf=1e3)
    sigma = torch.exp(agent_logscale.float()).mean(-1).clamp(min=table["min_scale"])
    return q[agent_type.long().clamp(0, 3)][..., None] * sigma
