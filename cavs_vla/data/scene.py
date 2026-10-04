"""Raw scene format shared by nuPlan, WOMD, CARLA and the closed-loop simulator.

A raw scene stores *world facts* for one window in the frame of the ego at the
reference time t0 = frame K-1 (ego at the origin, heading +x). It contains no
features: the torch featurizer (features.py) turns a raw scene into model
inputs for any observer actor at any time index, which is exactly what the
closed-loop simulator needs. Training and simulation therefore share one
featurization code path.

Fixed shapes (A actors incl. ego at index 0, T = K + H frames, P polylines, Q points):
    traj          (A, T, 7)  float32  [x, y, yaw, vx, vy, length, width] (scene frame)
    valid         (A, T)     bool
    atype         (A,)       int8     0 vehicle, 1 pedestrian, 2 cyclist, 3 static/other, -1 empty slot
    center_offset (A,)       float32  reference point -> box centre along heading (nuPlan ego: 1.461 m)
    poly          (P, Q, 2)  float32
    poly_valid    (P, Q)     bool
    poly_attr     (P, 6)     float32  [lane, connector, crosswalk, boundary, on_route, speed_limit/30]
    poly_tl       (P, T)     int8     0 none/unknown, 1 red, 2 yellow, 3 green
    instr         (L,)       int16    instruction tokens (route-derived)
    intent        ()         int8     intent label from the logged future (target only)
    src           ()         int8     0 nuPlan, 1 WOMD, 2 CARLA, 3 synthetic
    route_ok      ()         bool     whether a navigation route was available
    t0            ()         float64  timestamp of the reference frame (s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..geometry_np import (chunk_polyline, resample_polyline, rotate_vec, to_frame, wrap_angle)
from ..language import (ATTR_BOUNDARY, ATTR_CONNECTOR, ATTR_CROSSWALK, ATTR_LANE, ATTR_ROUTE, ATTR_SPEED,
                        NUM_POLY_ATTR, instruction_from_route, intent_label, tokenize)

SRC_NUPLAN, SRC_WAYMO, SRC_CARLA, SRC_SYNTH = 0, 1, 2, 3
TL_NONE, TL_RED, TL_YELLOW, TL_GREEN = 0, 1, 2, 3
KIND_TO_ATTR = {"lane": ATTR_LANE, "connector": ATTR_CONNECTOR, "crosswalk": ATTR_CROSSWALK,
                "boundary": ATTR_BOUNDARY}


def scene_field_specs(cfg: Config) -> Dict[str, tuple]:
    d = cfg.data
    A, T, P, Q, L = d.max_actors, d.hist_steps + d.fut_steps, d.max_polylines, d.poly_points, cfg.feat.instr_len
    return {
        "traj": ((A, T, 7), np.float32),
        "valid": ((A, T), np.bool_),
        "atype": ((A,), np.int8),
        "center_offset": ((A,), np.float32),
        "poly": ((P, Q, 2), np.float32),
        "poly_valid": ((P, Q), np.bool_),
        "poly_attr": ((P, NUM_POLY_ATTR), np.float32),
        "poly_tl": ((P, T), np.int8),
        "instr": ((L,), np.int16),
        "intent": ((), np.int8),
        "src": ((), np.int8),
        "route_ok": ((), np.bool_),
        "t0": ((), np.float64),
    }


def empty_scene(cfg: Config) -> Dict[str, np.ndarray]:
    out = {}
    for k, (shape, dt) in scene_field_specs(cfg).items():
        out[k] = np.zeros(shape, dtype=dt)
    out["atype"][:] = -1
    return out


@dataclass
class ActorTrack:
    """World-frame states of one actor over the window's T frames."""
    states: np.ndarray        # (T, 7) [x, y, yaw, vx, vy, length, width] world frame
    valid: np.ndarray         # (T,) bool
    atype: int
    center_offset: float = 0.0


@dataclass
class WorldPolyline:
    pts: np.ndarray           # (n, 2) world frame
    kind: str
    on_route: bool
    speed_limit: float        # m/s or nan
    tl: Optional[np.ndarray] = None   # (T,) int8 traffic-light state, None if not signalised


def assemble_scene(cfg: Config, ego: ActorTrack, others: List[ActorTrack], polylines: List[WorldPolyline],
                   route_available: bool, src: int, t0: float) -> Dict[str, np.ndarray]:
    """Convert world-frame actors and polylines to one fixed-shape raw scene."""
    d = cfg.data
    K, H = d.hist_steps, d.fut_steps
    T = K + H
    A, P, Q = d.max_actors, d.max_polylines, d.poly_points
    if ego.states.shape != (T, 7) or not ego.valid.all():
        raise ValueError("Ego track must cover every frame of the window")
    scene = empty_scene(cfg)
    origin = ego.states[K - 1, :2].copy()
    yaw0 = float(ego.states[K - 1, 2])

    def to_scene(track: ActorTrack) -> np.ndarray:
        s = track.states.astype(np.float64).copy()
        s[:, :2] = to_frame(s[:, :2], origin, yaw0)
        s[:, 2] = wrap_angle(s[:, 2] - yaw0)
        s[:, 3:5] = rotate_vec(s[:, 3:5], yaw0)
        s[~track.valid] = 0.0
        return s

    scene["traj"][0] = to_scene(ego)
    scene["valid"][0] = True
    scene["atype"][0] = 0
    scene["center_offset"][0] = ego.center_offset

    # Actors: those present at t0 first (they are what the policy can see), then by
    # their closest approach to the ego during the window (needed for closed-loop replay).
    ego_xy = ego.states[:, :2]
    ranked = []
    for tr in others:
        if not tr.valid.any():
            continue
        dist = np.linalg.norm(tr.states[:, :2] - ego_xy, axis=1)
        dist = np.where(tr.valid, dist, np.inf)
        dmin = float(dist.min())
        if dmin > d.actor_radius:
            continue
        ranked.append((0 if tr.valid[K - 1] else 1, float(dist[K - 1]) if tr.valid[K - 1] else dmin, tr))
    ranked.sort(key=lambda r: (r[0], r[1]))
    for slot, (_, _, tr) in enumerate(ranked[: A - 1], start=1):
        scene["traj"][slot] = to_scene(tr)
        scene["valid"][slot] = tr.valid
        scene["atype"][slot] = tr.atype
        scene["center_offset"][slot] = tr.center_offset

    fill_polylines(cfg, scene, polylines, origin, yaw0, route_available)
    sentence = instruction_from_route(scene["poly"].astype(np.float64), scene["poly_valid"], scene["poly_attr"],
                                      route_available, cfg.feat.instr_len)
    scene["instr"][:] = tokenize(sentence, cfg.feat.instr_len)
    scene["intent"][()] = intent_label(scene["traj"][0].astype(np.float64), K)
    scene["src"][()] = src
    scene["route_ok"][()] = bool(route_available)
    scene["t0"][()] = t0
    check_scene(cfg, scene)
    return scene



def fill_polylines(cfg: Config, scene: Dict[str, np.ndarray], polylines: List[WorldPolyline], origin: np.ndarray,
                   yaw0: float, route_available: bool) -> None:
    """Chunk world polylines to bounded length, move them into the scene frame, keep the nearest P."""
    d = cfg.data
    P, Q = d.max_polylines, d.poly_points
    chunks = []
    for pl in polylines:
        if len(pl.pts) < 2 and pl.kind != "crosswalk":
            continue
        pts = pl.pts
        if pl.kind == "crosswalk" and len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.concatenate([pts, pts[:1]], axis=0)
        for piece in chunk_polyline(pts, d.poly_chunk_len):
            if len(piece) < 2:
                continue
            loc = to_frame(piece, origin, yaw0)
            dmin = float(np.min(np.linalg.norm(loc, axis=1)))
            if dmin <= d.map_radius:
                chunks.append((dmin, loc, pl))
    chunks.sort(key=lambda c: c[0])
    scene["poly"][:] = 0
    scene["poly_valid"][:] = False
    scene["poly_attr"][:] = 0
    scene["poly_tl"][:] = 0
    for i, (_, loc, pl) in enumerate(chunks[:P]):
        scene["poly"][i] = resample_polyline(loc, Q).astype(np.float32)
        scene["poly_valid"][i] = True
        if pl.kind in KIND_TO_ATTR:
            scene["poly_attr"][i, KIND_TO_ATTR[pl.kind]] = 1.0
        scene["poly_attr"][i, ATTR_ROUTE] = 1.0 if (pl.on_route and route_available) else 0.0
        spd = pl.speed_limit
        scene["poly_attr"][i, ATTR_SPEED] = float(spd) / 30.0 if np.isfinite(spd) and spd > 0 else 0.0
        if pl.tl is not None:
            tl = np.asarray(pl.tl, dtype=np.int8)
            if tl.ndim == 0 or tl.size == 1:
                scene["poly_tl"][i] = int(tl.reshape(-1)[0])
            else:
                scene["poly_tl"][i] = tl

def check_scene(cfg: Config, scene: Dict[str, np.ndarray]) -> None:
    for k, (shape, dt) in scene_field_specs(cfg).items():
        arr = scene[k]
        if arr.shape != shape:
            raise ValueError(f"scene field {k}: shape {arr.shape} != {shape}")
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"scene field {k} contains non-finite values")
    K = cfg.data.hist_steps
    ego = scene["traj"][0, K - 1]
    if np.abs(ego[:3]).max() > 1e-3:
        raise ValueError("Ego is not at the origin of the scene frame at t0")


def stack_scenes(scenes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = scenes[0].keys()
    return {k: np.stack([s[k] for s in scenes], axis=0) for k in keys}
