"""Batched GPU featurizer: raw scene tensors -> policy inputs for any observer at any time.

This single function is used by training (observer = ego, t = K-1), by the
closed-loop simulator (observer = each controlled vehicle, t = current step)
and by the CARLA bridge. Inputs at time t only read frames <= t (the leak test
in selftest.py perturbs every frame > t and asserts the inputs do not change).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .config import Config
from .language import ATTR_ROUTE, NUM_POLY_ATTR, default_instruction

EGO_FEAT = 7       # x, y, cos, sin, vx, vy, valid
AGENT_FEAT = 12    # x, y, cos, sin, vx, vy, length, width, vehicle, pedestrian, cyclist, static
MAP_PT_FEAT = 4    # x, y, tangent cos, tangent sin
MAP_ATTR_FEAT = NUM_POLY_ATTR + 3   # + traffic light red / yellow / green


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _local(x: torch.Tensor, y: torch.Tensor, ox: torch.Tensor, oy: torch.Tensor,
           c: torch.Tensor, s: torch.Tensor):
    dx, dy = x - ox, y - oy
    return dx * c + dy * s, -dx * s + dy * c


def _rot(vx: torch.Tensor, vy: torch.Tensor, c: torch.Tensor, s: torch.Tensor):
    return vx * c + vy * s, -vx * s + vy * c


def _view_b(v: torch.Tensor, ndim: int) -> torch.Tensor:
    return v.view(v.shape[0], *([1] * (ndim - 1)))


def acc_centers(cfg: Config, device=None) -> torch.Tensor:
    f = cfg.feat
    return torch.linspace(f.a_min, f.a_max, f.n_acc_bins, device=device)


def kappa_centers(cfg: Config, device=None) -> torch.Tensor:
    f = cfg.feat
    u = torch.linspace(-1.0, 1.0, f.n_kappa_bins, device=device)
    return f.kappa_max * torch.sign(u) * u * u


def to_token(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return torch.argmin((values.unsqueeze(-1) - centers).abs(), dim=-1)


def steps_per_action(cfg: Config) -> int:
    return int(round(cfg.feat.action_dt / cfg.data.dt))


def featurize(scene: Dict[str, torch.Tensor], t: int, actor: torch.Tensor, cfg: Config, train: bool = False,
              with_targets: bool = False, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    traj, valid = scene["traj"], scene["valid"]
    B, A, T, _ = traj.shape
    K, H = cfg.data.hist_steps, cfg.data.fut_steps
    fc = cfg.feat
    if t < K - 1 or t >= T:
        raise ValueError(f"featurize: t={t} outside [{K - 1}, {T - 1}]")
    dev = traj.device
    ps, vs, ss = fc.pos_scale, fc.vel_scale, fc.size_scale
    bidx = torch.arange(B, device=dev)
    own = traj[bidx, actor]                     # (B, T, 7)
    own_v = valid[bidx, actor]                  # (B, T)
    ref = own[:, t]                             # (B, 7)
    ox, oy, oyaw = ref[:, 0], ref[:, 1], ref[:, 2]
    c, s = torch.cos(oyaw), torch.sin(oyaw)

    # ------------------------------------------------------------- ego history
    eh = own[:, t - K + 1:t + 1]
    ev = own_v[:, t - K + 1:t + 1].clone()
    ex, ey = _local(eh[..., 0], eh[..., 1], ox[:, None], oy[:, None], c[:, None], s[:, None])
    epsi = eh[..., 2] - oyaw[:, None]
    evx, evy = _rot(eh[..., 3], eh[..., 4], c[:, None], s[:, None])
    if train and fc.ego_history_dropout > 0:
        drop = torch.rand(B, device=dev, generator=generator) < fc.ego_history_dropout
        ev[drop, : K - 1] = False
    evf = ev.float()
    ego = torch.stack([ex / ps, ey / ps, torch.cos(epsi), torch.sin(epsi), evx / vs, evy / vs, evf], -1)
    ego = ego * evf[..., None]

    # ------------------------------------------------------------- agents
    N = fc.num_agents
    atype_all = scene["atype"].long()
    ok_t = valid[:, :, t] & (atype_all >= 0)
    not_self = torch.arange(A, device=dev)[None, :] != actor[:, None]
    cand = ok_t & not_self
    pos_t = traj[:, :, t, :2]
    dist = torch.hypot(pos_t[..., 0] - ox[:, None], pos_t[..., 1] - oy[:, None])
    dist = torch.where(cand, dist, torch.full_like(dist, float("inf")))
    dsort, aidx = torch.topk(dist, k=N, dim=1, largest=False)
    aok = torch.isfinite(dsort)
    at = traj[bidx[:, None], aidx]              # (B, N, T, 7)
    avt = valid[bidx[:, None], aidx] & aok[..., None]
    ah, av = at[:, :, t - K + 1:t + 1], avt[:, :, t - K + 1:t + 1]
    cb, sb = c[:, None, None], s[:, None, None]
    ax, ay = _local(ah[..., 0], ah[..., 1], ox[:, None, None], oy[:, None, None], cb, sb)
    apsi = ah[..., 2] - oyaw[:, None, None]
    avx, avy = _rot(ah[..., 3], ah[..., 4], cb, sb)
    atype = atype_all[bidx[:, None], aidx].clamp(min=0)
    onehot = torch.nn.functional.one_hot(atype, 4).float()[:, :, None, :].expand(B, N, K, 4)
    avf = av.float()
    agents = torch.cat([torch.stack([ax / ps, ay / ps, torch.cos(apsi), torch.sin(apsi), avx / vs, avy / vs,
                                     ah[..., 5] / ss, ah[..., 6] / ss], -1), onehot], -1) * avf[..., None]
    # Physical state of each agent at t in the observer frame (for the safety layer).
    cur = at[:, :, t]
    cx, cy = _local(cur[..., 0], cur[..., 1], ox[:, None], oy[:, None], c[:, None], s[:, None])
    cvx, cvy = _rot(cur[..., 3], cur[..., 4], c[:, None], s[:, None])
    coff = scene["center_offset"][bidx[:, None], aidx]
    cpsi = wrap(cur[..., 2] - oyaw[:, None])
    agents_phys = torch.stack([cx + coff * torch.cos(cpsi), cy + coff * torch.sin(cpsi), cpsi, cvx, cvy,
                               cur[..., 5], cur[..., 6], atype.float()], -1) * aok[..., None].float()

    # ------------------------------------------------------------- map
    P = fc.num_polylines
    poly, pv = scene["poly"], scene["poly_valid"]
    px, py = _local(poly[..., 0], poly[..., 1], ox[:, None, None], oy[:, None, None], cb, sb)
    tdx = torch.diff(poly[..., 0], dim=-1)
    tdy = torch.diff(poly[..., 1], dim=-1)
    tdx = torch.cat([tdx, tdx[..., -1:]], -1)
    tdy = torch.cat([tdy, tdy[..., -1:]], -1)
    tlx, tly = _rot(tdx, tdy, cb, sb)
    tn = torch.hypot(tlx, tly).clamp(min=1e-6)
    pd = torch.where(pv, torch.hypot(px, py), torch.full_like(px, float("inf"))).amin(-1)
    psort, pidx = torch.topk(pd, k=P, dim=1, largest=False)
    pok = torch.isfinite(psort)
    g = (bidx[:, None], pidx)
    mvalid = pv[g] & pok[..., None]
    mvf = mvalid.float()
    mp = torch.stack([px[g] / ps, py[g] / ps, (tlx / tn)[g], (tly / tn)[g]], -1) * mvf[..., None]
    attr = scene["poly_attr"][g].clone()
    is_route_owner = (actor == 0).float()[:, None]
    attr[..., ATTR_ROUTE] = attr[..., ATTR_ROUTE] * is_route_owner
    tl_t = scene["poly_tl"][g][..., t].long().clamp(0, 3)
    tl1h = torch.nn.functional.one_hot(tl_t, 4)[..., 1:].float()
    map_attr = torch.cat([attr, tl1h], -1) * pok[..., None].float()

    # ------------------------------------------------------------- language
    instr = scene["instr"].long()
    default = torch.as_tensor(default_instruction(cfg.feat.instr_len), device=dev).long()[None].expand(B, -1)
    instr = torch.where((actor == 0)[:, None], instr, default)

    own_len = own[:, t, 5]
    own_wid = own[:, t, 6]
    feat = dict(ego=ego, ego_valid=ev, agents=agents, agents_valid=av, agents_idx=aidx, agents_ok=aok,
                agents_phys=agents_phys, map=mp, map_valid=mvalid, map_attr=map_attr, instr=instr,
                ego_speed=torch.hypot(ref[:, 3], ref[:, 4]),
                ego_dims=torch.stack([own_len, own_wid, scene["center_offset"][bidx, actor]], -1))

    if with_targets:
        if t + H > T - 1:
            raise ValueError("featurize: targets need t + H <= T - 1")
        fsl = slice(t + 1, t + H + 1)
        ef = own[:, fsl]
        fx, fy = _local(ef[..., 0], ef[..., 1], ox[:, None], oy[:, None], c[:, None], s[:, None])
        fpsi = wrap(ef[..., 2] - oyaw[:, None])
        fv = torch.hypot(ef[..., 3], ef[..., 4])
        feat["ego_fut"] = torch.stack([fx, fy, fpsi, fv], -1)
        feat["ego_fut_valid"] = own_v[:, fsl]
        af = at[:, :, fsl]
        afx, afy = _local(af[..., 0], af[..., 1], ox[:, None, None], oy[:, None, None], cb, sb)
        feat["agent_fut"] = torch.stack([afx, afy], -1)
        feat["agent_fut_valid"] = avt[:, :, fsl]
        r = steps_per_action(cfg)
        Ha = H // r
        frames = t + torch.arange(Ha + 1, device=dev) * r
        sj = own[:, frames]                                  # (B, Ha+1, 7)
        vj = torch.hypot(sj[..., 3], sj[..., 4])
        psij = wrap(sj[..., 2] - oyaw[:, None])
        okj = own_v[:, frames]
        a = (vj[:, 1:] - vj[:, :-1]) / fc.action_dt
        ds = 0.5 * (vj[:, 1:] + vj[:, :-1]) * fc.action_dt
        dpsi = wrap(psij[:, 1:] - psij[:, :-1])
        kap = torch.where(ds >= fc.kappa_min_ds, dpsi / ds.clamp(min=fc.kappa_min_ds), torch.zeros_like(ds))
        a = a.clamp(fc.a_min, fc.a_max)
        kap = kap.clamp(-fc.kappa_max, fc.kappa_max)
        feat["gt_ctrl"] = torch.stack([a, kap], -1)
        feat["gt_tok"] = torch.stack([to_token(a, acc_centers(cfg, dev)), to_token(kap, kappa_centers(cfg, dev))], -1)
        feat["gt_ctrl_valid"] = okj[:, 1:] & okj[:, :-1]
        feat["intent"] = scene["intent"].long()
    return feat


def scene_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        if v.dtype == torch.float64 and k != "t0":
            v = v.float()
        out[k] = v.to(device, non_blocking=True)
    return out


def heading_interval_cos_sin(psi: torch.Tensor, eps: float):
    """Exact bounds of cos and sin over [psi - eps, psi + eps] (eps < pi/2)."""
    lo, hi = psi - eps, psi + eps
    c_lo = torch.minimum(torch.cos(lo), torch.cos(hi))
    c_hi = torch.maximum(torch.cos(lo), torch.cos(hi))
    s_lo = torch.minimum(torch.sin(lo), torch.sin(hi))
    s_hi = torch.maximum(torch.sin(lo), torch.sin(hi))

    def contains(angle: float) -> torch.Tensor:
        # does [lo, hi] contain angle + 2k*pi for some integer k
        k = torch.ceil((lo - angle) / (2 * math.pi))
        return angle + 2 * math.pi * k <= hi

    c_hi = torch.where(contains(0.0), torch.ones_like(c_hi), c_hi)
    c_lo = torch.where(contains(math.pi), -torch.ones_like(c_lo), c_lo)
    s_hi = torch.where(contains(math.pi / 2), torch.ones_like(s_hi), s_hi)
    s_lo = torch.where(contains(-math.pi / 2), -torch.ones_like(s_lo), s_lo)
    return c_lo, c_hi, s_lo, s_hi
