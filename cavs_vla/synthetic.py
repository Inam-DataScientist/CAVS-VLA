"""Synthetic nuPlan-schema logs, a JSON map and WOMD-like scenarios for the self-test.

These fixtures let every stage (data build -> featurize -> train -> verify ->
HJ -> shield -> closed loop) run end to end in minutes without the real
datasets, so wiring errors surface before a multi-day job. The data is
deliberately simple but exercises every code path: lanes, connectors, a
crosswalk, traffic lights, a mission route, a lead vehicle that brakes hard,
a lane-changing vehicle, oncoming traffic, a crossing pedestrian and a static cone.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

LOCATION = "synthtown"
LANE_W = 3.5


def _tok(*parts) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:16]


def synthetic_map() -> List[dict]:
    polys: List[dict] = []
    xs = np.linspace(-200.0, 290.0, 50)
    xs2 = np.linspace(310.0, 800.0, 50)
    polys.append(dict(pts=np.stack([xs, np.zeros_like(xs)], 1).tolist(), kind="lane", id="laneA1",
                      roadblock_id="rb_e1", speed_limit=13.4))
    polys.append(dict(pts=np.stack([xs, np.full_like(xs, LANE_W)], 1).tolist(), kind="lane", id="laneB1",
                      roadblock_id="rb_e1", speed_limit=13.4))
    polys.append(dict(pts=np.stack([xs2, np.zeros_like(xs2)], 1).tolist(), kind="lane", id="laneA2",
                      roadblock_id="rb_e2", speed_limit=13.4))
    polys.append(dict(pts=np.stack([xs2, np.full_like(xs2, LANE_W)], 1).tolist(), kind="lane", id="laneB2",
                      roadblock_id="rb_e2", speed_limit=13.4))
    xw = np.linspace(800.0, -200.0, 100)
    polys.append(dict(pts=np.stack([xw, np.full_like(xw, -LANE_W)], 1).tolist(), kind="lane", id="laneW",
                      roadblock_id="rb_w", speed_limit=13.4))
    xc = np.linspace(290.0, 310.0, 10)
    polys.append(dict(pts=np.stack([xc, np.zeros_like(xc)], 1).tolist(), kind="connector", id="conA",
                      roadblock_id="rbc_straight", speed_limit=13.4))
    polys.append(dict(pts=np.stack([xc, np.full_like(xc, LANE_W)], 1).tolist(), kind="connector", id="conB",
                      roadblock_id="rbc_straight", speed_limit=13.4))
    th = np.linspace(-np.pi / 2, 0.0, 12)
    arc = np.stack([290.0 + 12.0 * np.cos(th), 12.0 + 12.0 * np.sin(th)], 1)
    polys.append(dict(pts=arc.tolist(), kind="connector", id="conLeft", roadblock_id="rbc_left", speed_limit=8.0))
    yn = np.linspace(12.0, 300.0, 40)
    polys.append(dict(pts=np.stack([np.full_like(yn, 302.0), yn], 1).tolist(), kind="lane", id="laneN",
                      roadblock_id="rb_n", speed_limit=11.0))
    for yb in (-LANE_W - 1.75, LANE_W + 1.75):
        xb = np.linspace(-200.0, 800.0, 100)
        polys.append(dict(pts=np.stack([xb, np.full_like(xb, yb)], 1).tolist(), kind="boundary",
                          id=f"bnd{yb}", roadblock_id=None))
    cw = [[280.0, -6.0], [284.0, -6.0], [284.0, 6.0], [280.0, 6.0], [280.0, -6.0]]
    polys.append(dict(pts=cw, kind="crosswalk", id="cw1", roadblock_id=None))
    return polys


def _speed_profile(t: np.ndarray, v0: float, rng: np.random.Generator) -> np.ndarray:
    a = 0.6 * np.sin(2 * np.pi * t / rng.uniform(12, 20)) + 0.3 * np.sin(2 * np.pi * t / rng.uniform(5, 8))
    v = v0 + np.cumsum(a) * (t[1] - t[0])
    return np.clip(v, 0.5, 16.0)


def _integrate_lane(t: np.ndarray, v: np.ndarray, x0: float, y_of_x) -> np.ndarray:
    dt = t[1] - t[0]
    x = x0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    y = y_of_x(x)
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, yaw], 1)


def make_synthetic_nuplan(root: str, n_logs: int = 6, duration_s: float = 40.0, seed: int = 0) -> Dict[str, str]:
    os.makedirs(os.path.join(root, "db"), exist_ok=True)
    map_path = os.path.join(root, "map.json")
    with open(map_path, "w") as f:
        json.dump({LOCATION: synthetic_map()}, f)
    paths = []
    for li in range(n_logs):
        rng = np.random.default_rng(seed + li)
        name = f"2021.01.0{li % 9 + 1}.10.00.00_synth-{li:02d}"
        path = os.path.join(root, "db", f"{name}.db")
        if os.path.exists(path):
            os.remove(path)
        _write_log(path, name, rng, duration_s, li)
        paths.append(path)
    return dict(db_root=os.path.join(root, "db"), map_json=map_path, logs=json.dumps(paths))


def _write_log(path: str, name: str, rng: np.random.Generator, duration_s: float, li: int) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE log (token BLOB PRIMARY KEY, vehicle_name TEXT, date TEXT, timestamp INTEGER,
                          logfile TEXT, location TEXT, map_version TEXT);
        CREATE TABLE ego_pose (token BLOB PRIMARY KEY, timestamp INTEGER, x REAL, y REAL, z REAL,
                               qw REAL, qx REAL, qy REAL, qz REAL, vx REAL, vy REAL, vz REAL,
                               acceleration_x REAL, acceleration_y REAL, acceleration_z REAL,
                               angular_rate_x REAL, angular_rate_y REAL, angular_rate_z REAL,
                               epsg INTEGER, log_token BLOB);
        CREATE TABLE scene (token BLOB PRIMARY KEY, log_token BLOB, name TEXT, goal_ego_pose_token BLOB,
                            roadblock_ids TEXT);
        CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, next_token BLOB, prev_token BLOB, ego_pose_token BLOB,
                               lidar_token BLOB, scene_token BLOB, filename TEXT, timestamp INTEGER);
        CREATE TABLE category (token BLOB PRIMARY KEY, name TEXT, description TEXT);
        CREATE TABLE track (token BLOB PRIMARY KEY, category_token BLOB, width REAL, length REAL, height REAL);
        CREATE TABLE lidar_box (token BLOB PRIMARY KEY, lidar_pc_token BLOB, track_token BLOB, next_token BLOB,
                                prev_token BLOB, x REAL, y REAL, z REAL, width REAL, length REAL, height REAL,
                                vx REAL, vy REAL, vz REAL, yaw REAL, confidence REAL);
        CREATE TABLE traffic_light_status (token BLOB PRIMARY KEY, lidar_pc_token BLOB, lane_connector_id INTEGER,
                                           status TEXT);
    """)
    log_tok = _tok(name, "log")
    cur.execute("INSERT INTO log VALUES (?,?,?,?,?,?,?)",
                (log_tok, "synth", "2021-01-01", 0, name, LOCATION, "synthetic-map"))
    cats = {n: _tok("cat", n) for n in ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier",
                                        "generic_object", "czone_sign")}
    for n, t in cats.items():
        cur.execute("INSERT INTO category VALUES (?,?,?)", (t, n, n))
    scene_tok = _tok(name, "scene")
    cur.execute("INSERT INTO scene VALUES (?,?,?,?,?)",
                (scene_tok, log_tok, "scene0", None, "rb_e1 rbc_straight rb_e2"))
    # Ego at 100 Hz.
    t100 = np.arange(0.0, duration_s, 0.01)
    v_ego = _speed_profile(t100, rng.uniform(6.0, 11.0), rng)
    lc_x = rng.uniform(60.0, 140.0)

    def ego_y(x):
        return LANE_W / (1.0 + np.exp(-(x - lc_x) / 6.0)) if li % 2 == 0 else np.zeros_like(x)

    ego = _integrate_lane(t100, v_ego, rng.uniform(-50.0, 0.0), ego_y)
    t0_us = 1_600_000_000_000_000 + li * 10_000_000_000
    ego_tokens = []
    for i in range(len(t100)):
        yaw = ego[i, 2]
        tok = _tok(name, "ego", i)
        ego_tokens.append(tok)
        cur.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tok, int(t0_us + t100[i] * 1e6), ego[i, 0], ego[i, 1], 0.0,
                     np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2), float(v_ego[i]), 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32619, log_tok))
    # Agents (20 Hz).
    t20 = np.arange(0.0, duration_s, 0.05)
    agents = []
    v_lead = _speed_profile(t20, rng.uniform(7.0, 10.0), rng)
    brake_t = rng.uniform(10.0, 25.0)
    v_lead = np.where((t20 > brake_t) & (t20 < brake_t + 2.5), np.maximum(v_lead - 5.0 * (t20 - brake_t), 0.5),
                      v_lead)
    lead = _integrate_lane(t20, v_lead, ego[0, 0] + rng.uniform(18.0, 30.0), lambda x: np.zeros_like(x))
    agents.append(("lead", "vehicle", lead, v_lead, 4.6, 1.9))
    v_b = _speed_profile(t20, rng.uniform(8.0, 12.0), rng)
    b = _integrate_lane(t20, v_b, ego[0, 0] - rng.uniform(5.0, 20.0), lambda x: np.full_like(x, LANE_W))
    agents.append(("laneB", "vehicle", b, v_b, 4.8, 2.0))
    v_w = np.full_like(t20, rng.uniform(8.0, 12.0))
    xw0 = ego[0, 0] + rng.uniform(150.0, 250.0)
    w = np.stack([xw0 - np.cumsum(v_w) * 0.05, np.full_like(t20, -LANE_W), np.full_like(t20, np.pi)], 1)
    agents.append(("oncoming", "vehicle", w, v_w, 4.5, 1.9))
    ped_t0 = rng.uniform(5.0, 20.0)
    ped_y = np.clip(-6.0 + 1.3 * np.maximum(t20 - ped_t0, 0.0), -6.0, 6.0)
    ped = np.stack([np.full_like(t20, 282.0), ped_y, np.full_like(t20, np.pi / 2)], 1)
    v_ped = np.where((t20 > ped_t0) & (ped_y < 6.0), 1.3, 0.0)
    agents.append(("ped", "pedestrian", ped, v_ped, 0.6, 0.6))
    cone = np.stack([np.full_like(t20, ego[0, 0] + 120.0), np.full_like(t20, -LANE_W - 1.2),
                     np.zeros_like(t20)], 1)
    agents.append(("cone", "traffic_cone", cone, np.zeros_like(t20), 0.4, 0.4))
    track_tok = {}
    for an, cat, _, _, L, W in agents:
        track_tok[an] = _tok(name, "track", an)
        cur.execute("INSERT INTO track VALUES (?,?,?,?,?)", (track_tok[an], cats[cat], W, L, 1.5))
    rows_lp, rows_lb, rows_tl = [], [], []
    for i, t in enumerate(t20):
        lp_tok = _tok(name, "lp", i)
        ego_i = min(int(round(t / 0.01)), len(t100) - 1)
        rows_lp.append((lp_tok, None, None, ego_tokens[ego_i], b"lidar", scene_tok, f"f{i}",
                        int(t0_us + t * 1e6)))
        for an, cat, st, vv, L, W in agents:
            if np.hypot(st[i, 0] - ego[ego_i, 0], st[i, 1] - ego[ego_i, 1]) > 120.0:
                continue
            yaw = st[i, 2]
            rows_lb.append((_tok(name, "lb", i, an), lp_tok, track_tok[an], None, None, st[i, 0], st[i, 1], 0.0,
                            W, L, 1.5, vv[i] * np.cos(yaw), vv[i] * np.sin(yaw), 0.0, yaw, 1.0))
        phase = (t + li * 7.0) % 43.0
        status = "green" if phase < 20 else ("yellow" if phase < 23 else "red")
        for cid in ("conA", "conB", "conLeft"):
            rows_tl.append((_tok(name, "tl", i, cid), lp_tok, cid, status))
    cur.executemany("INSERT INTO lidar_pc VALUES (?,?,?,?,?,?,?,?)", rows_lp)
    cur.executemany("INSERT INTO lidar_box VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_lb)
    cur.executemany("INSERT INTO traffic_light_status VALUES (?,?,?,?)", rows_tl)
    con.commit()
    con.close()


# ------------------------------------------------------------- WOMD-like scenario objects


class _Feature(SimpleNamespace):
    def WhichOneof(self, name: str) -> str:  # mimics the protobuf oneof accessor
        return self.which


def make_fake_womd_scenario(seed: int = 0, n_frames: int = 91, current: int = 10) -> SimpleNamespace:
    """Duck-typed stand-in for waymo_open_dataset Scenario with the attributes the converter reads."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n_frames) * 0.1

    def mk_track(xs, ys, yaw, v, L, W, otype, valid=None):
        states = []
        for i in range(n_frames):
            ok = True if valid is None else bool(valid[i])
            states.append(SimpleNamespace(center_x=float(xs[i]), center_y=float(ys[i]), heading=float(yaw[i]),
                                          velocity_x=float(v[i] * np.cos(yaw[i])),
                                          velocity_y=float(v[i] * np.sin(yaw[i])),
                                          length=L, width=W, valid=ok))
        return SimpleNamespace(object_type=otype, states=states, id=int(rng.integers(1e6)))

    v = np.full(n_frames, rng.uniform(6, 12))
    x = 10.0 + np.cumsum(v) * 0.1
    sdc = mk_track(x, np.zeros(n_frames), np.zeros(n_frames), v, 4.8, 2.0, 1)
    vl = np.full(n_frames, rng.uniform(5, 9))
    lead = mk_track(x[0] + 20 + np.cumsum(vl) * 0.1, np.zeros(n_frames), np.zeros(n_frames), vl, 4.5, 1.9, 1)
    ped_valid = np.arange(n_frames) > 20
    ped = mk_track(np.full(n_frames, 60.0), np.linspace(-5, 5, n_frames), np.full(n_frames, np.pi / 2),
                   np.full(n_frames, 1.2), 0.7, 0.7, 2, valid=ped_valid)
    lanes = []
    for k, y in enumerate((0.0, 3.5, -3.5)):
        xs = np.linspace(-50, 250, 60)
        lanes.append(_Feature(id=100 + k, which="lane",
                              lane=SimpleNamespace(speed_limit_mph=30.0,
                                                   polyline=[SimpleNamespace(x=float(a), y=y, z=0.0) for a in xs])))
    lanes.append(_Feature(id=200, which="road_edge",
                          road_edge=SimpleNamespace(polyline=[SimpleNamespace(x=float(a), y=-5.5, z=0.0)
                                                              for a in np.linspace(-50, 250, 30)])))
    lanes.append(_Feature(id=300, which="crosswalk",
                          crosswalk=SimpleNamespace(polygon=[SimpleNamespace(x=a, y=b, z=0.0) for a, b in
                                                             [(58, -6), (62, -6), (62, 6), (58, 6)]])))
    dyn = [SimpleNamespace(lane_states=[SimpleNamespace(lane=100, state=6 if i < 50 else 4)]) for i in range(n_frames)]
    return SimpleNamespace(scenario_id=f"fake{seed}", timestamps_seconds=ts.tolist(), current_time_index=current,
                           tracks=[sdc, lead, ped], sdc_track_index=0, map_features=lanes, dynamic_map_states=dyn)
