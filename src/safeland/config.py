"""Typed, frozen configuration.

Every experiment is fully described by one :class:`Config`. Configs are loaded
from YAML, can be overridden with ``section.key=value`` strings, and are hashed
so that every result directory is traceable to the exact settings that made it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Cell = tuple[int, int]

METHODS = ("unshielded", "prob_shield", "sure_shield")


def _tuplify(value: Any) -> Any:
    """Convert (nested) lists from YAML into tuples so configs stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


@dataclass(frozen=True)
class WorldConfig:
    """Discrete UAV safe-landing world.

    Positions are grid cells ``(x, y)``; flying altitudes are ``1..max_alt``;
    altitude 0 is the ground. Battery levels are ``1..battery`` while flying.
    """

    width: int = 8
    depth: int = 8
    max_alt: int = 4
    battery: int = 30
    low_battery: int = 10
    pads: tuple[Cell, ...] = ((6, 6), (1, 6))
    nfz: tuple[Cell, ...] = ((3, 3), (3, 4), (4, 3), (4, 4))
    buildings: tuple[tuple[int, int, int], ...] = ((5, 2, 2), (2, 5, 3), (6, 4, 1))
    wind_dir: Cell = (1, 0)
    wind_prob: float = 0.2
    wind_min_alt: int = 2
    cost_lateral: int = 1
    cost_hover: int = 1
    cost_up: int = 2
    cost_down: int = 1
    start_alt: int = 3
    start_battery_min: int = 20

    def __post_init__(self) -> None:
        def inside(c: Cell) -> bool:
            return 0 <= c[0] < self.width and 0 <= c[1] < self.depth

        if self.width < 2 or self.depth < 2 or self.max_alt < 1 or self.battery < 2:
            raise ValueError("world too small")
        if not 0 <= self.low_battery < self.battery:
            raise ValueError("low_battery must be in [0, battery)")
        if not 0.0 <= self.wind_prob <= 1.0:
            raise ValueError("wind_prob must be a probability")
        if not self.pads:
            raise ValueError("at least one landing pad is required")
        for c in (*self.pads, *self.nfz):
            if not inside(c):
                raise ValueError(f"cell {c} outside the grid")
        for x, y, h in self.buildings:
            if not inside((x, y)) or h < 1:
                raise ValueError(f"invalid building {(x, y, h)}")
        if set(self.pads) & set(self.nfz):
            raise ValueError("a landing pad may not lie in a no-fly zone")
        if set(self.pads) & {(x, y) for x, y, _ in self.buildings}:
            raise ValueError("a landing pad may not lie under a building")
        if min(self.cost_lateral, self.cost_hover, self.cost_up, self.cost_down) < 1:
            raise ValueError("every action must cost >= 1 battery (guarantees termination)")
        if not 1 <= self.start_alt <= self.max_alt:
            raise ValueError("start_alt out of range")
        if not 1 <= self.start_battery_min <= self.battery:
            raise ValueError("start_battery_min out of range")


@dataclass(frozen=True)
class SpecConfig:
    """Temporal-logic requirements.

    ``safety_ltl`` is enforced by the shield and checked exactly on the product.
    ``ctl`` and ``pctl`` are the post-training certification queries.
    Atoms available: airborne, low, landed, crash, collision, nfz, geofence,
    empty, terminal, violation (world) plus spec_violated, spec_satisfied
    (product, verification only).
    """

    safety_ltl: str = (
        "G(!crash & !collision & !nfz & !geofence & !empty) & G(low -> F[<=8] landed)"
    )
    ctl: tuple[str, ...] = (
        "AG !spec_violated",
        "AF landed",
        "EF landed",
        "A[!spec_violated U landed]",
    )
    pctl: tuple[str, ...] = (
        "P=? [ !spec_violated U landed ]",
        "P=? [ F spec_violated ]",
        "P=? [ F landed ]",
        "R=? [ F terminal ]",
    )
    max_monitor_states: int = 10_000


@dataclass(frozen=True)
class RLConfig:
    episodes: int = 100_000
    alpha: float = 0.1
    gamma: float = 0.98
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7
    reward_land: float = 10.0
    penalty_violation: float = 10.0
    step_cost: float = 0.1

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if not 0 < self.alpha <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("alpha and gamma must be in (0, 1]")
        if not 0 <= self.eps_end <= self.eps_start <= 1:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if not 0 < self.eps_decay_frac <= 1:
            raise ValueError("eps_decay_frac must be in (0, 1]")


@dataclass(frozen=True)
class ShieldConfig:
    # Probabilistic shield keeps action a iff Q_safe(s, a) >= prob_lambda * max_a' Q_safe(s, a').
    prob_lambda: float = 0.95
    value_tol: float = 1e-12

    def __post_init__(self) -> None:
        if not 0 < self.prob_lambda <= 1:
            raise ValueError("prob_lambda must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default"
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = tuple(range(10))
    out_dir: str = "results"
    smc_samples: int = 2000

    def __post_init__(self) -> None:
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
        if not self.seeds:
            raise ValueError("at least one seed is required")


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ---------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        data = dict(data or {})
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        kwargs = {}
        for f in dataclasses.fields(cls):
            sub_cls = f.default_factory  # type: ignore[misc]
            raw = dict(data.get(f.name) or {})
            known = {g.name for g in dataclasses.fields(sub_cls)}
            bad = set(raw) - known
            if bad:
                raise ValueError(f"unknown keys in [{f.name}]: {sorted(bad)}")
            kwargs[f.name] = sub_cls(**{k: _tuplify(v) for k, v in raw.items()})
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            key, raw = item.split("=", 1)
            parts = key.strip().split(".")
            if len(parts) != 2:
                raise ValueError(f"override key must be section.key, got {key!r}")
            data.setdefault(parts[0], {})[parts[1]] = yaml.safe_load(raw)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
