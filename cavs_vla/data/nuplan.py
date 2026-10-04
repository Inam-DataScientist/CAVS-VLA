"""nuPlan log database reader -> leak-free raw scenes at 10 Hz.

What changed relative to BASELINE-01 (see the evaluation document):
  * Inputs never use frames after t0: the map comes from the HD map, not from
    the ego's own future path, and no "command speed" is derived from the future.
  * Frames are the lidar_pc sweeps (20 Hz) with their associated ego_pose,
    down-sampled to 10 Hz; windows never straddle time gaps.
  * Static objects (cones, barriers, generic objects) are kept as obstacles.
  * The navigation route comes from scene.roadblock_ids (the mission route the
    nuPlan planners receive). If the column is missing, an oracle route from the
    whole log's ego path is used and flagged in the manifest.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..geometry_np import quat_to_yaw
from .maps import MapSource
from .scene import (SRC_NUPLAN, TL_GREEN, TL_NONE, TL_RED, TL_YELLOW, ActorTrack, WorldPolyline,
                    assemble_scene)

CATEGORY_TYPE = {"vehicle": 0, "pedestrian": 1, "bicycle": 2, "traffic_cone": 3, "barrier": 3,
                 "czone_sign": 3, "generic_object": 3}
TL_STATUS = {"red": TL_RED, "yellow": TL_YELLOW, "green": TL_GREEN, "unknown": TL_NONE}


def _tables(cur: sqlite3.Cursor) -> set:
    return {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(cur: sqlite3.Cursor, table: str) -> set:
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}


class NuPlanLog:
    """All 10 Hz frames, boxes, traffic lights and routes of one nuPlan .db file."""

    def __init__(self, db_path: str, dt: float = 0.1) -> None:
        self.db_path = db_path
        self.name = os.path.splitext(os.path.basename(db_path))[0]
        self.dt = dt
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            self._read(con)
        finally:
            con.close()

    def _read(self, con: sqlite3.Connection) -> None:
        cur = con.cursor()
        tabs = _tables(cur)
        for t in ("lidar_pc", "ego_pose", "lidar_box", "track", "category", "log"):
            if t not in tabs:
                raise ValueError(f"{self.db_path}: missing table '{t}' (not a nuPlan log database)")
        log_cols = _columns(cur, "log")
        sel = ["location" if "location" in log_cols else "''", "map_version" if "map_version" in log_cols else "''"]
        row = cur.execute(f"SELECT {', '.join(sel)} FROM log LIMIT 1").fetchone()
        self.location = row[0] if row else ""
        self.map_version = row[1] if row else ""

        lp_cols = _columns(cur, "lidar_pc")
        scene_col = "lp.scene_token" if "scene_token" in lp_cols else "NULL"
        rows = cur.execute(
            f"SELECT lp.token, lp.timestamp, {scene_col}, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy "
            f"FROM lidar_pc lp JOIN ego_pose ep ON lp.ego_pose_token = ep.token ORDER BY lp.timestamp").fetchall()
        if not rows:
            raise ValueError(f"{self.db_path}: no lidar_pc/ego_pose rows")
        ts = np.array([r[1] for r in rows], dtype=np.int64)
        # Greedy 10 Hz down-sampling on the sweep timestamps (microseconds).
        keep = [0]
        min_gap = int(round(self.dt * 0.95 * 1e6))
        for i in range(1, len(ts)):
            if ts[i] - ts[keep[-1]] >= min_gap:
                keep.append(i)
        keep = np.asarray(keep)
        rows = [rows[i] for i in keep]
        self.frame_tokens: List[bytes] = [r[0] for r in rows]
        self.ts_us = ts[keep]
        self.scene_tokens = [r[2] for r in rows]
        arr = np.array([r[3:] for r in rows], dtype=np.float64)
        x, y, qw, qx, qy, qz, vx_l, vy_l = arr.T
        yaw = quat_to_yaw(qw, qx, qy, qz)
        c, s = np.cos(yaw), np.sin(yaw)
        # ego_pose velocities are expressed in the vehicle frame -> rotate to world.
        self.ego = np.stack([x, y, yaw, vx_l * c - vy_l * s, vx_l * s + vy_l * c], axis=1)
        self.F = len(rows)
        tok_to_frame = {tok: i for i, tok in enumerate(self.frame_tokens)}

        # Contiguous segments: split at gaps larger than 1.5 dt.
        dts = np.diff(self.ts_us) / 1e6
        breaks = np.nonzero(dts > 1.5 * self.dt)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [self.F]])
        self.segments = list(zip(starts.tolist(), ends.tolist()))

        # Boxes (all categories that matter for collisions).
        lb_cols = _columns(cur, "lidar_box")
        need = {"lidar_pc_token", "track_token", "x", "y"}
        if not need.issubset(lb_cols):
            raise ValueError(f"{self.db_path}: lidar_box lacks columns {sorted(need - lb_cols)}")
        yaw_c = "lb.yaw" if "yaw" in lb_cols else "0.0"
        vx_c = "lb.vx" if "vx" in lb_cols else "0.0"
        vy_c = "lb.vy" if "vy" in lb_cols else "0.0"
        len_c = "lb.length" if "length" in lb_cols else "t.length"
        wid_c = "lb.width" if "width" in lb_cols else "t.width"
        q = (f"SELECT lb.lidar_pc_token, lb.track_token, lb.x, lb.y, {yaw_c}, {vx_c}, {vy_c}, {len_c}, {wid_c}, c.name "
             f"FROM lidar_box lb JOIN track t ON lb.track_token = t.token "
             f"JOIN category c ON t.category_token = c.token")
        frames, tracks, states, types = [], [], [], []
        track_index: Dict[bytes, int] = {}
        for r in cur.execute(q):
            f = tok_to_frame.get(r[0])
            if f is None:
                continue
            cat = r[9]
            if cat not in CATEGORY_TYPE:
                continue
            k = track_index.setdefault(r[1], len(track_index))
            if k == len(types):
                types.append(CATEGORY_TYPE[cat])
            frames.append(f)
            tracks.append(k)
            states.append(r[2:9])
        self.track_types = np.asarray(types, dtype=np.int8)
        if frames:
            frames_a = np.asarray(frames, dtype=np.int64)
            tracks_a = np.asarray(tracks, dtype=np.int64)
            st = np.asarray(states, dtype=np.float64)
            st = np.nan_to_num(st, nan=0.0)
            order = np.lexsort((tracks_a, frames_a))
            self.box_frame = frames_a[order]
            self.box_track = tracks_a[order]
            self.box_state = st[order]           # [x, y, yaw, vx, vy, length, width]
        else:
            self.box_frame = np.zeros(0, dtype=np.int64)
            self.box_track = np.zeros(0, dtype=np.int64)
            self.box_state = np.zeros((0, 7))
        self.frame_row_start = np.searchsorted(self.box_frame, np.arange(self.F + 1), side="left")
        self.num_tracks = len(track_index)
        keys = self.box_track * self.F + self.box_frame
        korder = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[korder]
        self.sorted_rows = korder

        # Traffic lights per frame: {lane_connector_id: state}.
        self.tl: List[Dict[str, int]] = [dict() for _ in range(self.F)]
        if "traffic_light_status" in tabs:
            for tok, lc_id, status in cur.execute(
                    "SELECT lidar_pc_token, lane_connector_id, status FROM traffic_light_status"):
                f = tok_to_frame.get(tok)
                if f is None:
                    continue
                self.tl[f][str(lc_id)] = TL_STATUS.get(str(status).lower(), TL_NONE)

        # Mission route per scene token.
        self.routes: Dict[bytes, set] = {}
        self.route_source = "none"
        if "scene" in tabs and "roadblock_ids" in _columns(cur, "scene"):
            for tok, ids in cur.execute("SELECT token, roadblock_ids FROM scene"):
                if ids:
                    self.routes[tok] = {s for s in str(ids).replace(",", " ").split() if s}
            if self.routes:
                self.route_source = "scene.roadblock_ids"

    def lookup_tracks(self, tracks: np.ndarray, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """States (len(tracks), len(frames), 7) and validity for given track ids and frame indices."""
        keys = tracks[:, None] * self.F + frames[None, :]
        if len(self.sorted_keys) == 0:
            return np.zeros(keys.shape + (7,)), np.zeros(keys.shape, dtype=bool)
        pos = np.searchsorted(self.sorted_keys, keys)
        pos_c = np.minimum(pos, len(self.sorted_keys) - 1)
        found = self.sorted_keys[pos_c] == keys
        rows = self.sorted_rows[pos_c]
        states = np.where(found[..., None], self.box_state[rows], 0.0)
        return states, found


def oracle_route_ids(log: NuPlanLog, map_src: MapSource, step_m: float = 2.0, radius: float = 1.0) -> set:
    """Roadblocks touched by the logged ego path of the whole log (used only when no mission route exists)."""
    ids: set = set()
    last = None
    for p in log.ego[:, :2]:
        if last is not None and np.linalg.norm(p - last) < step_m:
            continue
        last = p
        for pl in map_src.query(log.location, float(p[0]), float(p[1]), radius):
            if pl["kind"] in ("lane", "connector") and pl["roadblock_id"] is not None:
                ids.add(pl["roadblock_id"])
    return ids


def build_log_scenes(cfg: Config, db_path: str, map_src: MapSource) -> Tuple[List[Dict[str, np.ndarray]], dict]:
    d = cfg.data
    K, H = d.hist_steps, d.fut_steps
    T = K + H
    log = NuPlanLog(db_path, d.dt)
    stride = max(1, int(round(d.window_stride_s / d.dt)))
    has_map = map_src.backend != "none"
    if has_map and not map_src.location_ok(log.location):
        raise KeyError(f"{log.name}: map has no location '{log.location}'")
    oracle_ids: Optional[set] = None
    route_source = log.route_source
    if d.use_route and has_map and route_source == "none":
        oracle_ids = oracle_route_ids(log, map_src)
        route_source = "oracle(ego path of the whole log)"
    scenes: List[Dict[str, np.ndarray]] = []
    ego_len, ego_wid = d.ego_length, d.ego_width
    for s0, s1 in log.segments:
        for r in range(s0 + K - 1, s1 - H, stride):
            if len(scenes) >= d.max_windows_per_log:
                break
            frames = np.arange(r - K + 1, r + H + 1)
            ego_states = np.zeros((T, 7))
            ego_states[:, :5] = log.ego[frames]
            ego_states[:, 5] = ego_len
            ego_states[:, 6] = ego_wid
            ego = ActorTrack(states=ego_states, valid=np.ones(T, dtype=bool), atype=0,
                             center_offset=d.ego_rear_to_center)
            lo, hi = log.frame_row_start[frames[0]], log.frame_row_start[frames[-1] + 1]
            cand = np.unique(log.box_track[lo:hi])
            others: List[ActorTrack] = []
            if len(cand):
                st, ok = log.lookup_tracks(cand, frames)
                for j, k in enumerate(cand):
                    others.append(ActorTrack(states=st[j], valid=ok[j], atype=int(log.track_types[k])))
            polylines: List[WorldPolyline] = []
            route_ok = False
            if has_map:
                route_ids: Optional[set] = None
                if d.use_route:
                    route_ids = log.routes.get(log.scene_tokens[r]) if log.routes else oracle_ids
                    route_ok = route_ids is not None and len(route_ids) > 0
                ex, ey = float(log.ego[r, 0]), float(log.ego[r, 1])
                for pl in map_src.query(log.location, ex, ey, d.map_radius):
                    tl = None
                    if pl["kind"] == "connector":
                        states_t = np.array([log.tl[f].get(pl["id"], TL_NONE) for f in frames], dtype=np.int8)
                        if states_t.any():
                            tl = states_t
                    on_route = bool(route_ok and pl["roadblock_id"] is not None and pl["roadblock_id"] in route_ids)
                    polylines.append(WorldPolyline(pts=pl["pts"], kind=pl["kind"], on_route=on_route,
                                                   speed_limit=pl["speed_limit"], tl=tl))
            scene = assemble_scene(cfg, ego, others, polylines, route_ok, SRC_NUPLAN, float(log.ts_us[r]) / 1e6)
            scenes.append(scene)
    meta = dict(log=log.name, location=log.location, frames_10hz=log.F, segments=len(log.segments),
                windows=len(scenes), route_source=route_source, tracks=log.num_tracks)
    return scenes, meta
