"""Waymo Open Motion Dataset (WOMD) -> raw scenes.

* TFRecord files are read with a small pure-Python reader, so TensorFlow is not
  needed. Parsing the Scenario proto needs only the protobuf message class from
  the official package:  pip install waymo-open-dataset-tf-2-12-0  (or any
  version that ships waymo_open_dataset/protos/scenario_pb2.py).
* WOMD scenarios are 9.1 s at 10 Hz: 11 history frames (current_time_index=10)
  and 80 future frames, i.e. exactly data.hist_steps=11 / data.fut_steps=80.
* WOMD has no navigation route. The route is reconstructed from the lanes the
  SDC's logged path traverses ("oracle" route, recorded in the manifest);
  disable with data.use_route=false for the no-route ablation.
"""
from __future__ import annotations

import struct
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..geometry_np import points_polyline_min_distance, polyline_tangent_angles, wrap_angle
from .scene import (SRC_WAYMO, TL_GREEN, TL_NONE, TL_RED, TL_YELLOW, ActorTrack, WorldPolyline,
                    assemble_scene)

WOMD_TYPE = {0: 3, 1: 0, 2: 1, 3: 2, 4: 3}
# TrafficSignalLaneState.State: 0 UNKNOWN, 1 ARROW_STOP, 2 ARROW_CAUTION, 3 ARROW_GO,
# 4 STOP, 5 CAUTION, 6 GO, 7 FLASHING_STOP, 8 FLASHING_CAUTION
WOMD_TL = {0: TL_NONE, 1: TL_RED, 2: TL_YELLOW, 3: TL_GREEN, 4: TL_RED, 5: TL_YELLOW, 6: TL_GREEN,
           7: TL_RED, 8: TL_YELLOW}
MPH_TO_MPS = 0.44704

# ---------------------------------------------------------------- TFRecord I/O


def _crc32c_table() -> List[int]:
    poly = 0x82F63B78
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        table.append(c)
    return table


_CRC_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_tfrecord(path: str, verify_crc: bool = False) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise IOError(f"{path}: truncated record header")
            (length,) = struct.unpack("<Q", header[:8])
            (len_crc,) = struct.unpack("<I", header[8:12])
            data = f.read(length)
            footer = f.read(4)
            if len(data) < length or len(footer) < 4:
                raise IOError(f"{path}: truncated record body")
            if verify_crc:
                if masked_crc(header[:8]) != len_crc:
                    raise IOError(f"{path}: length CRC mismatch")
                if masked_crc(data) != struct.unpack("<I", footer)[0]:
                    raise IOError(f"{path}: data CRC mismatch")
            yield data


def write_tfrecord(path: str, records: List[bytes]) -> None:
    with open(path, "wb") as f:
        for data in records:
            length = struct.pack("<Q", len(data))
            f.write(length)
            f.write(struct.pack("<I", masked_crc(length)))
            f.write(data)
            f.write(struct.pack("<I", masked_crc(data)))


def scenario_parser():
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise ImportError("Reading WOMD needs waymo_open_dataset.protos.scenario_pb2 "
                          "(pip install waymo-open-dataset-tf-2-12-0). TensorFlow itself is not used.") from exc

    def parse(blob: bytes):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(blob)
        return sc
    return parse


# ---------------------------------------------------------------- conversion


def _track_states(track, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    st = np.zeros((len(frames), 7))
    ok = np.zeros(len(frames), dtype=bool)
    n = len(track.states)
    for j, f in enumerate(frames):
        if f >= n:
            continue
        s = track.states[f]
        if not s.valid:
            continue
        st[j] = [s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y, s.length, s.width]
        ok[j] = True
    return st, ok


def _map_polylines(scenario) -> List[dict]:
    out = []
    for feat in scenario.map_features:
        which = feat.WhichOneof("feature_data")
        if which == "lane":
            pts = np.array([[p.x, p.y] for p in feat.lane.polyline], dtype=np.float64)
            spd = float(feat.lane.speed_limit_mph) * MPH_TO_MPS if feat.lane.speed_limit_mph > 0 else float("nan")
            out.append(dict(pts=pts, kind="lane", id=str(feat.id), speed_limit=spd))
        elif which in ("road_line", "road_edge"):
            line = getattr(feat, which)
            pts = np.array([[p.x, p.y] for p in line.polyline], dtype=np.float64)
            out.append(dict(pts=pts, kind="boundary", id=str(feat.id), speed_limit=float("nan")))
        elif which == "crosswalk":
            pts = np.array([[p.x, p.y] for p in feat.crosswalk.polygon], dtype=np.float64)
            out.append(dict(pts=pts, kind="crosswalk", id=str(feat.id), speed_limit=float("nan")))
    return [p for p in out if len(p["pts"]) >= 2]


def scenario_to_scene(cfg: Config, scenario) -> Optional[dict]:
    d = cfg.data
    K, H = d.hist_steps, d.fut_steps
    t0 = int(scenario.current_time_index)
    n_frames = len(scenario.timestamps_seconds)
    if t0 - (K - 1) < 0 or t0 + H >= n_frames:
        return None
    frames = np.arange(t0 - K + 1, t0 + H + 1)
    sdc_idx = int(scenario.sdc_track_index)
    sdc = scenario.tracks[sdc_idx]
    ego_st, ego_ok = _track_states(sdc, frames)
    if not ego_ok.all():
        return None
    ego = ActorTrack(states=ego_st, valid=ego_ok, atype=0, center_offset=0.0)
    others = []
    for i, tr in enumerate(scenario.tracks):
        if i == sdc_idx:
            continue
        st, ok = _track_states(tr, frames)
        if ok.any():
            others.append(ActorTrack(states=st, valid=ok, atype=WOMD_TYPE.get(int(tr.object_type), 3)))
    # Traffic lights per frame: lane id -> state.
    tl_per_frame = []
    for f in frames:
        m = {}
        if f < len(scenario.dynamic_map_states):
            for ls in scenario.dynamic_map_states[f].lane_states:
                m[str(ls.lane)] = WOMD_TL.get(int(ls.state), TL_NONE)
        tl_per_frame.append(m)
    raw_polys = _map_polylines(scenario)
    ego_xy = ego_st[:, :2]
    route_ok = bool(d.use_route)
    fut_xy = ego_xy[K - 1:]
    fut_yaw = ego_st[K - 1:, 2]
    polylines: List[WorldPolyline] = []
    for p in raw_polys:
        dmin_ego = float(np.min(np.linalg.norm(p["pts"] - ego_xy[K - 1], axis=1)))
        if dmin_ego > d.map_radius + d.poly_chunk_len:
            continue
        on_route = False
        if route_ok and p["kind"] == "lane":
            dist = points_polyline_min_distance(fut_xy, p["pts"])
            near = dist < 1.5
            if near.any():
                ang = polyline_tangent_angles(p["pts"])
                # heading of the lane at the closest vertex to each near ego point
                idx = np.argmin(np.linalg.norm(fut_xy[near][:, None, :] - p["pts"][None], axis=-1), axis=1)
                on_route = bool(np.any(np.abs(wrap_angle(ang[idx] - fut_yaw[near])) < 0.6))
        tl = None
        if p["kind"] == "lane":
            states_t = np.array([m.get(p["id"], TL_NONE) for m in tl_per_frame], dtype=np.int8)
            if states_t.any():
                tl = states_t
        polylines.append(WorldPolyline(pts=p["pts"], kind=p["kind"], on_route=on_route,
                                       speed_limit=p["speed_limit"], tl=tl))
    t0_s = float(scenario.timestamps_seconds[t0])
    return assemble_scene(cfg, ego, others, polylines, route_ok, SRC_WAYMO, t0_s)


def build_file_scenes(cfg: Config, path: str, parse=None, limit: int = 0) -> Tuple[List[dict], dict]:
    parse = parse or scenario_parser()
    scenes, n_read, n_skipped = [], 0, 0
    for blob in read_tfrecord(path):
        n_read += 1
        sc = parse(blob)
        scene = scenario_to_scene(cfg, sc)
        if scene is None:
            n_skipped += 1
        else:
            scenes.append(scene)
        if limit and len(scenes) >= limit:
            break
    meta = dict(log=path, scenarios=n_read, windows=len(scenes), skipped=n_skipped,
                route_source="oracle(lanes along the logged SDC path)" if cfg.data.use_route else "none")
    return scenes, meta
