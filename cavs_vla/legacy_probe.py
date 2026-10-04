"""Leakage probe for the locked BASELINE-01 (old repository format).

Run inside the OLD repository root (needs data/processed/{val,test}.npz,
data/processed/baseline_norm_stats.json and checkpoints/baseline_01/best.pt):

    python -m cavs_vla legacy-probe --legacy-root ~/Work/inam/Multi_Agent_VLA

Part 1 (numpy, no model): how close is the "map" M to the ground-truth future Y?
    If the centre polyline of M passes within centimetres of every future point,
    M was built from the future (unified_schema.build_sample passes all K+H ego
    states to _map_proxy).
Part 2 (torch): re-score the frozen checkpoint with
    (a) the original inputs,
    (b) M rebuilt from history only (the same _map_proxy on the K past positions),
    (c) M removed (all polylines absent),
    (d) C[0] = current speed / 20 instead of the mean FUTURE speed,
    (e) (b) + (d),
    and a constant-velocity baseline. A several-fold ADE increase under (b)-(e)
    confirms that the reported 0.062 m ADE depends on future information.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

LANE_W = 3.5


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.repeat(pts[:1], n, 0)
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
    if d[-1] < 1e-6:
        return np.repeat(pts[:1], n, 0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1)


def map_proxy(center_body: np.ndarray, P: int = 8, Q: int = 10) -> np.ndarray:
    """Exact copy of the old unified_schema._map_proxy."""
    cl = _resample(center_body, Q)
    tg = np.gradient(cl, axis=0)
    nl = np.stack([-tg[:, 1], tg[:, 0]], 1)
    ln = np.linalg.norm(nl, axis=1, keepdims=True)
    ln[ln < 1e-6] = 1
    nl /= ln
    M = np.zeros((P, Q, 3))
    for p in range(P):
        if p < 3:
            M[p, :, :2] = cl + nl * ((p - 1) * LANE_W)
            M[p, :, 2] = 1.0
    return M


def map_future_distance(M: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Median over samples of the mean distance from future points Y[:, :, :2] to M's centre polyline (p=1)."""
    out = np.zeros(len(M))
    for i in range(len(M)):
        cl = M[i, 1, :, :2]
        fut = Y[i, :, :2]
        a, b = cl[:-1], cl[1:]
        ab = b - a
        t = np.clip(((fut[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9)[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        out[i] = np.linalg.norm(fut[:, None] - proj, axis=-1).min(1).mean()
    return out


def leak_report_numpy(npz_path: str, max_samples: int = 20000) -> Dict:
    d = np.load(npz_path, allow_pickle=True)
    M, Y, X, C = d["M"], d["Y"], d["X"], d["C"]
    n = min(len(M), max_samples)
    dist = map_future_distance(M[:n], Y[:n])
    # Distance of the future to a history-only proxy, for comparison.
    hist_M = np.stack([map_proxy(X[i, :, :2]) for i in range(n)])
    dist_hist = map_future_distance(hist_M, Y[:n])
    v_future = np.linalg.norm(np.diff(Y[:n, :, :2], axis=1), axis=-1).mean(1) / 0.05
    c0 = C[:n, 0] * 20.0
    corr = float(np.corrcoef(c0, v_future)[0, 1]) if n > 2 else float("nan")
    return dict(samples=int(n), map_to_future_median_m=float(np.median(dist)),
                map_to_future_p90_m=float(np.quantile(dist, 0.9)),
                history_proxy_to_future_median_m=float(np.median(dist_hist)),
                corr_C0_vs_future_speed=corr,
                verdict=("LEAK: the map tensor contains the future path" if np.median(dist) < 0.05 else
                         "map tensor does not coincide with the future path"))


def legacy_rescore(legacy_root: str, split: str = "val", max_samples: int = 20000) -> Dict:
    import torch
    import torch.nn as nn

    class RouteFreeTrajectoryTransformer(nn.Module):
        """Verbatim copy of src/vla/vla_baseline_routefree.py (BASELINE-01)."""

        def __init__(self, K=8, N=15, P=8, Q=10, H=30, d_model=128, nhead=4, layers=3):
            super().__init__()
            self.H, self.N, self.P = H, N, P
            self.ego_enc = nn.Sequential(nn.Linear(K * 5, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.agent_enc = nn.Sequential(nn.Linear(K * 7, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.map_enc = nn.Sequential(nn.Linear(Q * 3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.ctx_enc = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.type_emb = nn.Embedding(5, d_model)
            self.pos_emb = nn.Embedding(N + P + 3, d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            self.dec = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, H * 4))

        def forward(self, X, A, A_mask, M, C):
            B, dev = X.shape[0], X.device
            q = self.query.expand(B, -1, -1)
            e = self.ego_enc(X.reshape(B, -1)).unsqueeze(1)
            a = self.agent_enc(A.reshape(B, self.N, -1))
            m = self.map_enc(M.reshape(B, self.P, -1))
            c = self.ctx_enc(C).unsqueeze(1)
            x = torch.cat([q, e, a, m, c], dim=1)
            agent_present = ~A_mask.all(dim=2)
            map_present = M[..., 2].sum(dim=2) > 0
            pm = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=dev), ~agent_present, ~map_present,
                            torch.zeros(B, 1, dtype=torch.bool, device=dev)], dim=1)
            T = x.shape[1]
            types = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                               torch.ones(B, 1, dtype=torch.long, device=dev),
                               torch.full((B, self.N), 2, dtype=torch.long, device=dev),
                               torch.full((B, self.P), 3, dtype=torch.long, device=dev),
                               torch.full((B, 1), 4, dtype=torch.long, device=dev)], dim=1)
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
            x = x + self.type_emb(types) + self.pos_emb(pos)
            out = self.enc(x, src_key_padding_mask=pm)
            return torch.tanh(self.dec(out[:, 0, :]).reshape(B, self.H, 4))

    root = os.path.expanduser(legacy_root)
    d = np.load(os.path.join(root, "data/processed", f"{split}.npz"), allow_pickle=True)
    stats = {k: np.asarray(v, np.float32) for k, v in json.load(open(os.path.join(root, "data/processed/baseline_norm_stats.json"))).items()}
    n = min(len(d["X"]), max_samples)
    X, A, Am, M, C, Y = (d[k][:n].astype(np.float32) for k in ("X", "A", "A_mask", "M", "C", "Y"))
    Am = Am.astype(bool)

    def norm(v, key):
        mn, mx = stats[f"{key}_min"], stats[f"{key}_max"]
        return 2.0 * (v - mn) / np.maximum(mx - mn, 1e-6) - 1.0

    def norm_agent(a):
        a = a.copy()
        a[..., 0:2] = np.clip(a[..., 0:2] / 50.0, -1.0, 1.0)
        mn, mx = stats["A_min"][2:7], stats["A_max"][2:7]
        a[..., 2:7] = 2.0 * (a[..., 2:7] - mn) / np.maximum(mx - mn, 1e-6) - 1.0
        return a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouteFreeTrajectoryTransformer().to(device)
    ck = torch.load(os.path.join(root, "checkpoints/baseline_01/best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    y_min, y_max = stats["Y_min"][:2], stats["Y_max"][:2]
    M_hist = np.stack([map_proxy(X[i, :, :2]) for i in range(n)]).astype(np.float32)
    M_none = np.zeros_like(M)
    C_now = C.copy()
    C_now[:, 0] = np.clip(X[:, -1, 2] / 20.0, 0.0, 1.0)
    variants = {"a_original": (M, C), "b_map_history_only": (M_hist, C), "c_map_removed": (M_none, C),
                "d_speed_now": (M, C_now), "e_history_map_and_speed_now": (M_hist, C_now)}
    res = {}
    with torch.no_grad():
        for name, (Mv, Cv) in variants.items():
            ade_s, fde_s = 0.0, 0.0
            for s0 in range(0, n, 1024):
                sl = slice(s0, min(n, s0 + 1024))
                t = lambda a: torch.as_tensor(a[sl], device=device)
                out = model(t(norm(X, "X")), t(norm_agent(A)), t(Am), t(norm(Mv, "M")), t(Cv)).cpu().numpy()
                pred = (out[..., :2] + 1) / 2 * (y_max - y_min) + y_min
                err = np.linalg.norm(pred - Y[sl, :, :2], axis=-1)
                ade_s += err.mean(1).sum()
                fde_s += err[:, -1].sum()
            res[name] = dict(ADE=ade_s / n, FDE=fde_s / n)
    v = X[:, -1, 2]
    tgrid = np.arange(1, Y.shape[1] + 1) * 0.05
    cv = np.stack([v[:, None] * tgrid, np.zeros((n, len(tgrid)))], -1)
    err = np.linalg.norm(cv - Y[:, :, :2], axis=-1)
    res["constant_velocity"] = dict(ADE=float(err.mean()), FDE=float(err[:, -1].mean()))
    res["numpy_leak_report"] = leak_report_numpy(os.path.join(root, "data/processed", f"{split}.npz"), max_samples)
    return res
