"""UAV safe-landing world as a finite Markov decision process.

State space
    Flying states ``(x, y, z, b)`` with ``z in 1..Z`` and battery ``b in 1..B``,
    plus six absorbing sinks: ``landed`` (the goal) and the five violations
    ``crash`` (touch-down off a pad), ``collision`` (building), ``nfz``,
    ``geofence`` (left the grid or exceeded the ceiling) and ``empty`` (battery
    exhausted while airborne).

Actions
    N, S, E, W (lateral), HOVER, UP, DOWN. Every action costs >= 1 battery unit,
    so every run reaches a sink within ``B`` steps: the flying part of the MDP is
    acyclic (a DAG ordered by battery). This makes all fixed points below exact
    after at most ``B + 1`` iterations.

Wind
    After the commanded move, if the new altitude is >= ``wind_min_alt`` a gust
    displaces the UAV one cell along ``wind_dir`` with probability ``wind_prob``.
    Near the ground (altitude below ``wind_min_alt``) the vehicle is sheltered.
    For the *sure* shield the gust is treated as adversarial (it may or may not
    happen); for PCTL it is a probability.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .config import WorldConfig

ACTIONS: tuple[str, ...] = ("N", "S", "E", "W", "HOVER", "UP", "DOWN")
_DELTA: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),
    (0, -1, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
)
SINKS: tuple[str, ...] = ("landed", "crash", "collision", "nfz", "geofence", "empty")
VIOLATIONS: frozenset[str] = frozenset(SINKS[1:])
WORLD_ATOMS: tuple[str, ...] = ("airborne", "low", "terminal", "violation", *SINKS)


class World:
    """Finite MDP with transition tables ``next[s, a, k]`` and ``prob[s, a, k]``.

    Each (state, action) pair has exactly two outcome slots ``k in {0, 1}``
    (no gust / gust). When there is no gust the second slot duplicates the first
    with probability 0, so vectorised "all successors" checks need no masking.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.W, self.D, self.Z, self.B = cfg.width, cfg.depth, cfg.max_alt, cfg.battery
        self.shape = (self.W, self.D, self.Z, self.B)
        self.n_fly = self.W * self.D * self.Z * self.B
        self.n_states = self.n_fly + len(SINKS)
        self.n_actions = len(ACTIONS)
        self.sink = {name: self.n_fly + i for i, name in enumerate(SINKS)}
        self._pads = frozenset(cfg.pads)
        self._nfz = frozenset(cfg.nfz)
        self._height = np.zeros((self.W, self.D), dtype=int)
        for x, y, h in cfg.buildings:
            self._height[x, y] = h
        self._cost = (
            cfg.cost_lateral,
            cfg.cost_lateral,
            cfg.cost_lateral,
            cfg.cost_lateral,
            cfg.cost_hover,
            cfg.cost_up,
            cfg.cost_down,
        )
        self.next, self.prob = self._build_transitions()
        self.labels, self.label_id, self.alphabet = self._build_labels()
        self.is_sink = np.arange(self.n_states) >= self.n_fly
        self.start_states = self._build_starts()

    # ------------------------------------------------------------ encoding
    def encode(self, x: int, y: int, z: int, b: int) -> int:
        return int(np.ravel_multi_index((x, y, z - 1, b - 1), self.shape))

    def decode(self, s: int) -> tuple[int, int, int, int]:
        if s >= self.n_fly:
            raise ValueError(f"state {s} is a sink ({self.sink_name(s)})")
        x, y, zi, bi = np.unravel_index(s, self.shape)
        return int(x), int(y), int(zi) + 1, int(bi) + 1

    def sink_name(self, s: int) -> str:
        return SINKS[s - self.n_fly]

    def describe(self, s: int) -> str:
        if s >= self.n_fly:
            return self.sink_name(s).upper()
        x, y, z, b = self.decode(s)
        return f"(x={x}, y={y}, z={z}, battery={b})"

    # ------------------------------------------------------------ dynamics
    def resolve(self, x: int, y: int, z: int, b: int) -> int:
        """Map a tentative configuration to a state index (event priority is fixed)."""
        if not (0 <= x < self.W and 0 <= y < self.D) or z > self.Z:
            return self.sink["geofence"]
        if z <= 0:
            return self.sink["landed"] if (x, y) in self._pads else self.sink["crash"]
        if (x, y) in self._nfz:
            return self.sink["nfz"]
        if self._height[x, y] >= z:
            return self.sink["collision"]
        if b <= 0:
            return self.sink["empty"]
        return self.encode(x, y, z, b)

    def outcomes(self, s: int, a: int) -> list[tuple[float, int]]:
        """Direct (table-free) successor distribution. Used to cross-check the tables."""
        if s >= self.n_fly:
            return [(1.0, s)]
        x, y, z, b = self.decode(s)
        dx, dy, dz = _DELTA[a]
        nx, ny, nz, nb = x + dx, y + dy, z + dz, b - self._cost[a]
        t0 = self.resolve(nx, ny, nz, nb)
        p = self.cfg.wind_prob
        if t0 >= self.n_fly or nz < self.cfg.wind_min_alt or p == 0.0:
            return [(1.0, t0)]
        wx, wy = self.cfg.wind_dir
        t1 = self.resolve(nx + wx, ny + wy, nz, nb)
        return [(1.0 - p, t0), (p, t1)]

    def _build_transitions(self) -> tuple[np.ndarray, np.ndarray]:
        nxt = np.empty((self.n_states, self.n_actions, 2), dtype=np.int64)
        prob = np.zeros((self.n_states, self.n_actions, 2), dtype=np.float64)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                outs = self.outcomes(s, a)
                if len(outs) == 1:
                    nxt[s, a] = outs[0][1]
                    prob[s, a] = (1.0, 0.0)
                else:
                    nxt[s, a] = (outs[0][1], outs[1][1])
                    prob[s, a] = (outs[0][0], outs[1][0])
        return nxt, prob

    def sample(self, s: int, a: int, rng: np.random.Generator) -> int:
        return int(self.next[s, a, 0] if rng.random() < self.prob[s, a, 0] else self.next[s, a, 1])

    def sample_direct(self, s: int, a: int, rng: np.random.Generator) -> int:
        """Sampler that bypasses the tables (independent path for statistical checks)."""
        outs = self.outcomes(s, a)
        u = rng.random()
        acc = 0.0
        for p, t in outs:
            acc += p
            if u < acc:
                return t
        return outs[-1][1]

    # ------------------------------------------------------------ labels
    def label_of(self, s: int) -> frozenset[str]:
        if s >= self.n_fly:
            name = self.sink_name(s)
            extra = {"violation"} if name in VIOLATIONS else set()
            return frozenset({name, "terminal", *extra})
        _, _, _, b = self.decode(s)
        return frozenset({"airborne", "low"} if b <= self.cfg.low_battery else {"airborne"})

    def _build_labels(self):
        labels = [self.label_of(s) for s in range(self.n_states)]
        alphabet: list[frozenset[str]] = []
        index: dict[frozenset[str], int] = {}
        ids = np.empty(self.n_states, dtype=np.int64)
        for s, lab in enumerate(labels):
            if lab not in index:
                index[lab] = len(alphabet)
                alphabet.append(lab)
            ids[s] = index[lab]
        return labels, ids, tuple(alphabet)

    def atom(self, name: str) -> np.ndarray:
        if name not in WORLD_ATOMS:
            raise KeyError(f"unknown atom {name!r}; known: {WORLD_ATOMS}")
        mask = np.array([name in lab for lab in self.alphabet])
        return mask[self.label_id]

    def atoms(self) -> dict[str, np.ndarray]:
        return {name: self.atom(name) for name in WORLD_ATOMS}

    # ------------------------------------------------------------ starts
    def _build_starts(self) -> np.ndarray:
        z = self.cfg.start_alt
        starts = []
        for x in range(self.W):
            for y in range(self.D):
                for b in range(self.cfg.start_battery_min, self.B + 1):
                    s = self.resolve(x, y, z, b)
                    if s < self.n_fly:
                        starts.append(s)
        if not starts:
            raise ValueError("no valid start states")
        return np.asarray(starts, dtype=np.int64)

    def flying_cells(self, z: int) -> Iterable[tuple[int, int]]:
        for x in range(self.W):
            for y in range(self.D):
                if self.resolve(x, y, z, 1) < self.n_fly:
                    yield x, y

    def ascii_map(self) -> str:
        """Top view; y grows upward. P pad, # no-fly zone, digit building height."""
        rows = []
        for y in reversed(range(self.D)):
            row = []
            for x in range(self.W):
                if (x, y) in self._pads:
                    row.append("P")
                elif (x, y) in self._nfz:
                    row.append("#")
                elif self._height[x, y] > 0:
                    row.append(str(self._height[x, y]))
                else:
                    row.append(".")
            rows.append(f"{y} " + " ".join(row))
        rows.append("  " + " ".join(str(x) for x in range(self.W)))
        wx, wy = self.cfg.wind_dir
        rows.append(f"wind -> ({wx},{wy}) p={self.cfg.wind_prob} at z>={self.cfg.wind_min_alt}")
        return "\n".join(rows)
