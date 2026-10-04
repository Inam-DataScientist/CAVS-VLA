"""Collect the wind-probability sweep into one markdown table.

Usage: python scripts/collect_sweep.py results/main results/wind_0.1 results/wind_0.3 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def main(dirs: list[str]) -> str:
    rows = []
    for d in dirs:
        root = Path(d)
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        p = cfg["world"]["wind_prob"]
        summ = json.loads((root / "summary.json").read_text())
        for method, s in summ.items():
            rows.append((p, method, s))
    order = {"unshielded": 0, "prob_shield": 1, "sure_shield": 2}
    rows.sort(key=lambda r: (r[0], order.get(r[1], 9)))
    out = [
        "| gust p | method | seeds | train violations | P(safe landing) | worst-start P | AG safe (frac.) | E[steps] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, m, s in rows:
        out.append(
            f"| {p} | {m} | {s['train_violations']['n']} | "
            f"{s['train_violations']['mean']:.0f} ± {s['train_violations']['ci95']:.0f} | "
            f"{s['P_safe_landing']['mean']:.4f} | {s['P_safe_landing_min_start']['mean']:.4f} | "
            f"{s['AG_safe_fraction']['mean']:.3f} | {s['expected_steps']['mean']:.2f} |"
        )
    return "\n".join(out)


if __name__ == "__main__":
    print(main(sys.argv[1:]))
