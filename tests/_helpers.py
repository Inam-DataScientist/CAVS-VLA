"""Shared small configuration so the whole suite runs in seconds."""

from __future__ import annotations

from safeland.config import Config
from safeland.experiment import Context, build_context

SMALL = {
    "world": {
        "width": 4,
        "depth": 4,
        "max_alt": 3,
        "battery": 10,
        "low_battery": 4,
        "pads": [[3, 3]],
        "nfz": [[1, 1]],
        "buildings": [[2, 0, 1]],
        "wind_dir": [1, 0],
        "wind_prob": 0.3,
        "start_alt": 2,
        "start_battery_min": 6,
    },
    "spec": {
        "safety_ltl": "G(!crash & !collision & !nfz & !geofence & !empty) & G(low -> F[<=4] landed)"
    },
    "rl": {"episodes": 3000},
    "experiment": {"seeds": [0], "smc_samples": 400, "name": "test"},
}


def small_cfg(**sections) -> Config:
    data = {k: dict(v) for k, v in SMALL.items()}
    for sec, vals in sections.items():
        data.setdefault(sec, {}).update(vals)
    return Config.from_dict(data)


_CTX: Context | None = None


def small_ctx() -> Context:
    global _CTX
    if _CTX is None:
        _CTX = build_context(small_cfg())
    return _CTX
