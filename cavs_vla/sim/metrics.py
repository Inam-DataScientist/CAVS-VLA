"""Aggregation with uncertainty: bootstrap CIs, Wilson intervals, paired tests across methods."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

BINARY = ["collision", "at_fault_collision", "deadlock"]
CONTINUOUS = ["min_distance", "min_ttc", "offroad_fraction", "max_lane_dev", "progress", "distance_m", "mean_speed",
              "max_accel", "min_accel", "max_jerk", "max_lat_acc", "comfort_violation_rate", "intervention_rate",
              "mpc_rate", "fallback_rate", "emergency_rate", "intervention_mag_a", "intervention_mag_kappa",
              "certified_fraction", "min_safety_margin", "stl_robustness", "cert_width_accel", "cert_width_kappa",
              "cl_coverage_overall"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def bootstrap_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 0):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(records: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m, rs in by_method.items():
        res: Dict[str, Dict[str, float]] = {"n": {"value": len(rs)}}
        for k in BINARY:
            vals = [bool(r[k]) for r in rs if k in r]
            p, lo, hi = wilson(sum(vals), len(vals))
            res[k] = dict(rate=p, ci_lo=lo, ci_hi=hi, count=int(sum(vals)), n=len(vals))
        km = sum(r.get("distance_m", 0.0) for r in rs) / 1000.0
        res["collisions_per_km"] = dict(value=(sum(bool(r["collision"]) for r in rs) / km) if km > 0 else float("nan"),
                                        km=km)
        for k in CONTINUOUS:
            vals = np.array([r[k] for r in rs if k in r], dtype=np.float64)
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_mean(vals)
            res[k] = dict(mean=mean, ci_lo=lo, ci_hi=hi, median=float(np.nanmedian(vals[np.isfinite(vals)]))
                          if np.isfinite(vals).any() else float("nan"))
        out[m] = res
    return out


def _key(r: Dict):
    return (r["seed"], r["scene"], r["vehicle"])


def paired_tests(records: List[Dict], reference: str, others: Sequence[str]) -> Dict[str, Dict]:
    """McNemar (exact binomial) on collisions and paired bootstrap on progress, matched by (seed, scene, vehicle)."""
    ref = {_key(r): r for r in records if r["method"] == reference}
    out = {}
    for m in others:
        if m == reference:
            continue
        cur = {_key(r): r for r in records if r["method"] == m}
        keys = sorted(set(ref) & set(cur))
        if not keys:
            continue
        a = np.array([ref[k]["collision"] for k in keys], dtype=bool)
        b = np.array([cur[k]["collision"] for k in keys], dtype=bool)
        b01 = int((~a & b).sum())          # reference safe, method collides
        b10 = int((a & ~b).sum())          # reference collides, method safe
        n_disc = b01 + b10
        p = float(stats.binomtest(min(b01, b10), n_disc, 0.5).pvalue) if n_disc else 1.0
        dp = np.array([cur[k]["progress"] - ref[k]["progress"] for k in keys])
        mean, lo, hi = bootstrap_mean(dp)
        out[m] = dict(n_pairs=len(keys), collisions_ref_only=b10, collisions_method_only=b01, mcnemar_p=p,
                      progress_diff_mean=mean, progress_diff_ci=(lo, hi))
    return out


def markdown_table(summary: Dict, methods: Sequence[str]) -> str:
    cols = [("collision", "rate"), ("at_fault_collision", "rate"), ("min_distance", "mean"), ("min_ttc", "median"),
            ("progress", "mean"), ("offroad_fraction", "mean"), ("deadlock", "rate"), ("intervention_rate", "mean"),
            ("certified_fraction", "mean"), ("min_safety_margin", "median"), ("max_jerk", "mean")]
    head = "| Method | n | " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        cells = []
        for c, stat in cols:
            if c not in s:
                cells.append("–")
                continue
            v = s[c]
            if stat == "rate":
                cells.append(f"{100 * v['rate']:.2f}% [{100 * v['ci_lo']:.2f}, {100 * v['ci_hi']:.2f}]")
            else:
                cells.append(f"{v[stat]:.3g}")
        lines.append(f"| {m} | {s['n']['value']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
