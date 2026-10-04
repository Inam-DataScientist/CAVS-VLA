"""NumPy geometry used by the offline data builders (no torch dependency)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def to_frame(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """World -> frame whose origin is ``origin`` and whose +x axis points along ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    out = np.empty_like(d)
    out[..., 0] = d[..., 0] * c + d[..., 1] * s
    out[..., 1] = -d[..., 0] * s + d[..., 1] * c
    return out


def rotate_vec(v: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate world vectors into a frame with heading ``yaw`` (no translation)."""
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0] * c + v[..., 1] * s
    out[..., 1] = -v[..., 0] * s + v[..., 1] * c
    return out


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample to ``n`` points equally spaced in arclength (endpoints kept)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    pts = pts[keep]
    if len(pts) == 1:
        return np.repeat(pts[:1], n, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def chunk_polyline(pts: np.ndarray, max_len: float) -> List[np.ndarray]:
    """Split a polyline into consecutive pieces no longer than ``max_len`` metres."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return [pts] if len(pts) else []
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = s[-1]
    if total <= max_len:
        return [pts]
    n_chunks = int(np.ceil(total / max_len))
    bounds = np.linspace(0.0, total, n_chunks + 1)
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        inner = (s > a) & (s < b)
        xa = np.interp(a, s, pts[:, 0]); ya = np.interp(a, s, pts[:, 1])
        xb = np.interp(b, s, pts[:, 0]); yb = np.interp(b, s, pts[:, 1])
        piece = np.concatenate([[[xa, ya]], pts[inner], [[xb, yb]]], axis=0)
        pieces.append(piece)
    return pieces


def point_polyline_distance(p: np.ndarray, pts: np.ndarray) -> float:
    """Euclidean distance from point p to the polyline pts (segments)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return float(np.linalg.norm(pts[0] - p))
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


def points_polyline_min_distance(points: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each of many points (M,2) to a polyline (n,2) -> (M,)."""
    points = np.asarray(points, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return np.linalg.norm(points - pts[0], axis=1)
    a = pts[:-1][None]
    ab = (pts[1:] - pts[:-1])[None]
    ap = points[:, None, :] - a
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip((ap * ab).sum(-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.min(np.linalg.norm(points[:, None, :] - proj, axis=-1), axis=1)


def polyline_tangent_angles(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts))
    d = np.diff(pts, axis=0)
    d = np.concatenate([d, d[-1:]], axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


class PolylineIndex:
    """Bounding-box index over world-frame polylines for radius queries (no shapely needed)."""

    def __init__(self, polylines: List[dict]) -> None:
        self.polylines = [p for p in polylines if len(p["pts"]) >= 1]
        if self.polylines:
            boxes = np.array([[p["pts"][:, 0].min(), p["pts"][:, 1].min(),
                               p["pts"][:, 0].max(), p["pts"][:, 1].max()] for p in self.polylines])
        else:
            boxes = np.zeros((0, 4))
        self.boxes = boxes

    def query(self, x: float, y: float, radius: float) -> List[Tuple[dict, float]]:
        if len(self.boxes) == 0:
            return []
        dx = np.maximum(np.maximum(self.boxes[:, 0] - x, x - self.boxes[:, 2]), 0.0)
        dy = np.maximum(np.maximum(self.boxes[:, 1] - y, y - self.boxes[:, 3]), 0.0)
        cand = np.nonzero(dx * dx + dy * dy <= radius * radius)[0]
        out = []
        p = np.array([x, y])
        for i in cand:
            d = point_polyline_distance(p, self.polylines[i]["pts"])
            if d <= radius:
                out.append((self.polylines[i], d))
        return out
