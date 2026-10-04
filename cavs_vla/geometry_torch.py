"""Batched oriented-box geometry (collision, distance, time-to-collision) for the simulators."""
from __future__ import annotations

import torch


def box_corners(cx: torch.Tensor, cy: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor,
                width: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., 4, 2) corners in counter-clockwise order."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    sx = torch.tensor([1.0, -1.0, -1.0, 1.0], device=cx.device, dtype=cx.dtype)
    sy = torch.tensor([1.0, 1.0, -1.0, -1.0], device=cx.device, dtype=cx.dtype)
    lx = hl[..., None] * sx
    ly = hw[..., None] * sy
    x = cx[..., None] + lx * c[..., None] - ly * s[..., None]
    y = cy[..., None] + lx * s[..., None] + ly * c[..., None]
    return torch.stack([x, y], -1)


def boxes_overlap(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Separating-axis test for convex quads c1, c2 of shape (..., 4, 2) (broadcastable)."""
    c1, c2 = torch.broadcast_tensors(c1, c2)
    sep = torch.zeros(c1.shape[:-2], dtype=torch.bool, device=c1.device)
    for poly in (c1, c2):
        edges = torch.roll(poly, -1, dims=-2) - poly
        axes = torch.stack([-edges[..., 1], edges[..., 0]], -1)          # (..., 4, 2) normals
        p1 = (c1[..., None, :, :] * axes[..., :, None, :]).sum(-1)       # (..., 4 axes, 4 pts)
        p2 = (c2[..., None, :, :] * axes[..., :, None, :]).sum(-1)
        gap = (p1.amax(-1) < p2.amin(-1)) | (p2.amax(-1) < p1.amin(-1))
        sep = sep | gap.any(-1)
    return ~sep


def _point_segment_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    return torch.linalg.norm(p - (a + t[..., None] * ab), dim=-1)


def box_distance(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Minimum distance between convex quads (0 when overlapping)."""
    c1, c2 = torch.broadcast_tensors(c1, c2)
    a2, b2 = c2, torch.roll(c2, -1, dims=-2)
    a1, b1 = c1, torch.roll(c1, -1, dims=-2)
    d12 = _point_segment_dist(c1[..., :, None, :], a2[..., None, :, :], b2[..., None, :, :]).flatten(-2).amin(-1)
    d21 = _point_segment_dist(c2[..., :, None, :], a1[..., None, :, :], b1[..., None, :, :]).flatten(-2).amin(-1)
    d = torch.minimum(d12, d21)
    return torch.where(boxes_overlap(c1, c2), torch.zeros_like(d), d)


def point_polyline_distance(p: torch.Tensor, poly: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """p (B, 2), poly (B, P, Q, 2), valid (B, P, Q) -> (B,) distance to the nearest valid segment."""
    a = poly[:, :, :-1]
    b = poly[:, :, 1:]
    seg_ok = valid[:, :, :-1] & valid[:, :, 1:]
    d = _point_segment_dist(p[:, None, None, :], a, b)
    d = torch.where(seg_ok, d, torch.full_like(d, float("inf")))
    return d.flatten(1).amin(-1)
