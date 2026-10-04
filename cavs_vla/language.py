"""Route-derived navigation instructions (the policy's language channel) and intent labels.

The instruction is computed from the navigation route only (nuPlan's mission
roadblocks), never from the ego's future trajectory, so it is a legitimate
planner input. For WOMD, which ships no route, the route is reconstructed from
the lanes the logged ego traverses; this is disclosed in the dataset manifest
(route_source = "oracle") and can be ablated with data.use_route=false.

Intent labels are *targets* derived from the logged future and are only used
as an auxiliary supervision signal ("semantic reasoning" output).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .geometry_np import polyline_tangent_angles, wrap_angle

WORDS = ["<pad>", "<bos>", "<eos>", "follow", "the", "road", "turn", "left", "right", "at", "next",
         "intersection", "change", "lane", "to", "go", "straight", "speed", "limit", "mph", "unknown",
         "and", "keep"]
WORDS += [str(v) for v in range(5, 85, 5)]
VOCAB = {w: i for i, w in enumerate(WORDS)}
PAD, BOS, EOS = VOCAB["<pad>"], VOCAB["<bos>"], VOCAB["<eos>"]
VOCAB_SIZE = len(WORDS)

INTENTS = ["cruise", "accelerate", "decelerate", "stop", "turn_left", "turn_right",
           "lane_change_left", "lane_change_right"]

# Polyline attribute columns (shared with the raw scene format).
ATTR_LANE, ATTR_CONNECTOR, ATTR_CROSSWALK, ATTR_BOUNDARY, ATTR_ROUTE, ATTR_SPEED = range(6)
NUM_POLY_ATTR = 6


def tokenize(sentence: str, length: int) -> np.ndarray:
    ids = [BOS] + [VOCAB.get(w, VOCAB["unknown"]) for w in sentence.split()] + [EOS]
    ids = ids[:length]
    out = np.full(length, PAD, dtype=np.int16)
    out[: len(ids)] = ids
    return out


def detokenize(ids: np.ndarray) -> str:
    words = []
    for i in np.asarray(ids).tolist():
        if i in (PAD, BOS):
            continue
        if i == EOS:
            break
        words.append(WORDS[i])
    return " ".join(words)


def default_instruction(length: int) -> np.ndarray:
    return tokenize("follow the road", length)


def _speed_phrase(speed_mps: float) -> str:
    if not np.isfinite(speed_mps) or speed_mps <= 0.5:
        return ""
    mph = int(round(speed_mps * 2.23694 / 5.0) * 5)
    mph = int(np.clip(mph, 5, 80))
    return f" speed limit {mph} mph"


def instruction_from_route(poly: np.ndarray, poly_valid: np.ndarray, poly_attr: np.ndarray,
                           route_available: bool, length: int) -> str:
    """Build the instruction sentence in the scene frame (ego at origin, heading +x)."""
    lane_mask = (poly_attr[:, ATTR_LANE] > 0.5) | (poly_attr[:, ATTR_CONNECTOR] > 0.5)
    any_valid = poly_valid.any(axis=1)
    speed = np.nan
    # Speed limit of the lane closest to the ego.
    best = np.inf
    for i in np.nonzero(lane_mask & any_valid)[0]:
        pts = poly[i][poly_valid[i]]
        d = float(np.min(np.linalg.norm(pts, axis=1)))
        if d < best and poly_attr[i, ATTR_SPEED] > 0:
            best = d
            speed = float(poly_attr[i, ATTR_SPEED]) * 30.0
    speed_txt = _speed_phrase(speed) if best < 5.0 else ""
    if not route_available:
        return "follow the road" + speed_txt
    route_mask = lane_mask & any_valid & (poly_attr[:, ATTR_ROUTE] > 0.5)
    if not route_mask.any():
        return "follow the road" + speed_txt
    far_angles: List[float] = []
    mid_lat: List[float] = []
    for i in np.nonzero(route_mask)[0]:
        pts = poly[i][poly_valid[i]]
        ang = polyline_tangent_angles(pts)
        dist = np.linalg.norm(pts, axis=1)
        sel = (pts[:, 0] > 0) & (dist >= 25.0) & (dist <= 60.0)
        far_angles.extend(ang[sel].tolist())
        sel2 = (pts[:, 0] >= 15.0) & (pts[:, 0] <= 40.0)
        mid_lat.extend(pts[sel2, 1].tolist())
    if far_angles:
        dpsi = float(np.arctan2(np.mean(np.sin(far_angles)), np.mean(np.cos(far_angles))))
        if dpsi > 0.6:
            return "turn left at the next intersection" + speed_txt
        if dpsi < -0.6:
            return "turn right at the next intersection" + speed_txt
    if mid_lat:
        lat = float(np.median(mid_lat))
        # The route lane is laterally displaced and no route lane passes the ego.
        ego_on_route = False
        for i in np.nonzero(route_mask)[0]:
            pts = poly[i][poly_valid[i]]
            if float(np.min(np.linalg.norm(pts, axis=1))) < 1.5:
                ego_on_route = True
                break
        if not ego_on_route and abs(lat) > 2.0:
            return ("change lane to the left" if lat > 0 else "change lane to the right") + speed_txt
    return "go straight" + speed_txt


def intent_label(ego_traj: np.ndarray, hist_steps: int) -> int:
    """ego_traj: (T, 7) [x, y, psi, vx, vy, len, wid] in the scene frame (ego at t0 = origin)."""
    cur = ego_traj[hist_steps - 1]
    fin = ego_traj[-1]
    v0 = float(np.hypot(cur[3], cur[4]))
    vf = float(np.hypot(fin[3], fin[4]))
    disp = float(np.hypot(fin[0] - cur[0], fin[1] - cur[1]))
    dpsi = float(wrap_angle(fin[2] - cur[2]))
    # Lateral offset measured perpendicular to the final heading (the lane direction after
    # the manoeuvre), so a vehicle that starts slightly angled is not labelled a lane change.
    dx, dy = float(fin[0] - cur[0]), float(fin[1] - cur[1])
    lat = -dx * np.sin(fin[2]) + dy * np.cos(fin[2])
    if vf < 0.5 and (disp < 2.0 or v0 < 2.0):
        return INTENTS.index("stop")
    if dpsi > 0.6:
        return INTENTS.index("turn_left")
    if dpsi < -0.6:
        return INTENTS.index("turn_right")
    if abs(dpsi) < 0.3 and lat > 2.5:
        return INTENTS.index("lane_change_left")
    if abs(dpsi) < 0.3 and lat < -2.5:
        return INTENTS.index("lane_change_right")
    if vf - v0 < -2.0:
        return INTENTS.index("decelerate")
    if vf - v0 > 2.0:
        return INTENTS.index("accelerate")
    return INTENTS.index("cruise")
