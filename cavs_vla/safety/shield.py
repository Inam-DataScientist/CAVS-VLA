"""Predictive safety shield: certify the proposal, repair it with MPC, re-certify, or fall back.

Per scene and control tick:

1. Interactions. Agents are projected onto the proposal's path (the nominal
   rollout, extended straight). Each relevant agent becomes either
     * a LEAD (same direction, in the corridor, ahead): reduced state (gap, v_ego, v_other) -> HJ;
     * a REAR follower (same direction, behind): RSS-style soft cost only (rear blame);
     * a CONFLICT (everything else whose conformal prediction set meets the corridor):
       a path interval [z_lo, z_hi] occupied during [t_in, t_out].
2. Certificate check over the commit horizon (commit_s + actuator settling):
     ego tube  : exact interval kinematics for any command in U_cert (or the point command),
                 first-order actuator lag from the previous command, |w| <= w_bar;
     other tube: speed / displacement intervals under a_other in [a_o_min, a_o_max];
     LEAD      : gap stays >= d_min during the tube AND V_HJ(box corner) >= hj_margin at its end
                 (V >= 0 => the maximal-braking backup keeps the gap for the HJ horizon);
                 without HJ (B4/B5) a constant-velocity geometric check is used instead;
     CONFLICT  : no overlap inside the window during the tube, and at its end the ego can
                 still (A) stop before the zone, (B) clear it before t_in, or (C) cannot reach
                 it before t_out.
3. If the check fails: MPC (augmented Lagrangian, batched Adam) near the proposal, then the
   MPC output is re-checked as a point command. If that fails too: the HJ least-restrictive
   backup (the acceleration maximising the worst-case margin) or maximal braking.

Soundness of the executed command therefore never depends on MPC convergence; it depends on
the stated assumptions (bicycle + lag model, w_bar, X0, a_other bounds, conformal coverage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..config import Config
from ..dynamics import rollout
from ..features import steps_per_action, wrap

SRC_NOMINAL, SRC_MPC, SRC_FALLBACK, SRC_EMERGENCY = 0, 1, 2, 3
INF = 1e6


@dataclass
class Interactions:
    path: torch.Tensor          # (B, n, 2)
    path_s: torch.Tensor        # (B, n)
    lead_ok: torch.Tensor       # (B, N)
    lead_gap: torch.Tensor      # (B, N)  bumper-to-bumper, metres
    lead_v: torch.Tensor        # (B, N)  along-path speed of the other agent
    rear_ok: torch.Tensor
    rear_gap: torch.Tensor
    rear_v: torch.Tensor
    conf_ok: torch.Tensor       # (B, N)
    z_lo: torch.Tensor          # (B, N) path interval occupied by the agent's prediction set
    z_hi: torch.Tensor
    t_in: torch.Tensor          # (B, N) occupancy window (s)
    t_out: torch.Tensor
    geo_pred: torch.Tensor      # (B, N, H, 2) predicted centres
    geo_rad: torch.Tensor       # (B, N, H) agent covering radius + conformal radius
    geo_ok: torch.Tensor        # (B, N)
    overlap_now: torch.Tensor   # (B,) an agent box already intersects the ego corridor footprint


def _interp_polyline(pts: torch.Tensor, s_query: torch.Tensor) -> torch.Tensor:
    seg = pts[:, 1:] - pts[:, :-1]
    segl = torch.linalg.norm(seg, dim=-1)
    s_cum = torch.cat([torch.zeros_like(segl[:, :1]), torch.cumsum(segl, -1)], -1)
    idx = (torch.searchsorted(s_cum.contiguous(), s_query.contiguous(), right=True) - 1).clamp(0, seg.shape[1] - 1)
    s0 = torch.gather(s_cum, 1, idx)
    l = torch.gather(segl, 1, idx).clamp(min=1e-6)
    sg = torch.gather(seg, 1, idx[..., None].expand(-1, -1, 2))
    p0 = torch.gather(pts, 1, idx[..., None].expand(-1, -1, 2))
    return p0 + ((s_query - s0) / l)[..., None] * sg


def build_path(traj: torch.Tensor, v0: torch.Tensor, cfg: Config, n: int = 96) -> tuple:
    """Path through the nominal rollout, extended 150 m along its final heading, resampled to n points."""
    B = traj.shape[0]
    origin = torch.zeros(B, 1, 2, device=traj.device, dtype=traj.dtype)
    last = traj[:, -1]
    ext = last[:, :2] + 150.0 * torch.stack([torch.cos(last[:, 2]), torch.sin(last[:, 2])], -1)
    pts = torch.cat([origin, traj[..., :2], ext[:, None]], 1)
    length = (v0 * cfg.shield.check_horizon_s * 1.3 + 40.0).clamp(min=60.0, max=250.0)
    s = torch.linspace(0.0, 1.0, n, device=traj.device, dtype=traj.dtype)[None] * length[:, None]
    return _interp_polyline(pts, s), s


def _project(path: torch.Tensor, path_s: torch.Tensor, p: torch.Tensor):
    """Project points p (B, M, 2) on path (B, n, 2) -> (s, signed lateral, path heading)."""
    d = torch.cdist(p, path)                                  # (B, M, n)
    k = d.argmin(-1)                                          # (B, M)
    tang = torch.diff(path, dim=1)
    tang = torch.cat([tang, tang[:, -1:]], 1)
    th = torch.atan2(tang[..., 1], tang[..., 0])              # (B, n)
    pk = torch.gather(path, 1, k[..., None].expand(-1, -1, 2))
    thk = torch.gather(th, 1, k)
    rel = p - pk
    lat = -rel[..., 0] * torch.sin(thk) + rel[..., 1] * torch.cos(thk)
    lon = rel[..., 0] * torch.cos(thk) + rel[..., 1] * torch.sin(thk)
    s = torch.gather(path_s, 1, k) + lon
    return s, lat, thk


def build_interactions(cfg: Config, traj_nom: torch.Tensor, v0: torch.Tensor, agents_phys: torch.Tensor,
                       agents_ok: torch.Tensor, pred: torch.Tensor, conf_rad: Optional[torch.Tensor],
                       ego_dims: torch.Tensor) -> Interactions:
    sc = cfg.shield
    B, N = agents_ok.shape
    H = pred.shape[2]
    dt = cfg.data.dt
    path, path_s = build_path(traj_nom.float(), v0.float(), cfg)
    L_e, W_e, c_off = ego_dims[:, 0], ego_dims[:, 1], ego_dims[:, 2]
    front_e = (c_off + 0.5 * L_e)[:, None]
    rear_e = (c_off - 0.5 * L_e)[:, None]
    xy = agents_phys[..., :2]
    psi_a = agents_phys[..., 2]
    vx, vy = agents_phys[..., 3], agents_phys[..., 4]
    L_a, W_a = agents_phys[..., 5], agents_phys[..., 6]
    s_a, lat_a, th = _project(path, path_s, xy)
    cos_lead = math.cos(math.radians(sc.lead_heading_deg))
    same_dir = torch.cos(wrap(psi_a - th)) > cos_lead
    corridor = 0.5 * W_e[:, None] + 0.5 * W_a + sc.corridor_margin
    in_corr = lat_a.abs() < corridor
    v_along = (vx * torch.cos(th) + vy * torch.sin(th)).clamp(min=0.0)
    lead_gap = s_a - 0.5 * L_a - front_e
    ahead = (s_a + 0.5 * L_a) > front_e
    lead_ok = agents_ok & same_dir & in_corr & ahead
    # Rear follower in the ego frame (behind the reference point, same heading).
    rear_gap = rear_e - (xy[..., 0] + 0.5 * L_a)
    rear_ok = agents_ok & (torch.cos(psi_a) > cos_lead) & (xy[..., 1].abs() < corridor) & (xy[..., 0] < 0) & ~lead_ok
    rear_v = vx.clamp(min=0.0)
    # Conflicts from the prediction set (every second step to save compute), current position included.
    stride = 2
    steps = torch.arange(stride - 1, H, stride, device=pred.device)
    times = torch.cat([torch.zeros(1, device=pred.device), (steps + 1).float() * dt])
    pts = torch.cat([xy[:, :, None], pred[:, :, steps]], 2)                      # (B, N, Ts, 2)
    r_cov = 0.5 * torch.sqrt(L_a ** 2 + W_a ** 2)
    if conf_rad is not None:
        r_conf = torch.cat([torch.zeros_like(conf_rad[:, :, :1]), conf_rad[:, :, steps]], 2)
    else:
        r_conf = torch.zeros_like(pts[..., 0])
    Ts = pts.shape[2]
    s_p, lat_p, _ = _project(path, path_s, pts.reshape(B, N * Ts, 2))
    s_p, lat_p = s_p.view(B, N, Ts), lat_p.view(B, N, Ts)
    rad = r_cov[..., None] + r_conf
    hit = (lat_p.abs() < 0.5 * W_e[:, None, None] + rad + sc.corridor_margin) & (s_p > -5.0)
    hit = hit & agents_ok[..., None] & ~lead_ok[..., None] & ~rear_ok[..., None]
    conf_ok = hit.any(-1)
    big = torch.full_like(s_p, INF)
    z_lo = torch.where(hit, s_p - rad, big).amin(-1)
    z_hi = torch.where(hit, s_p + rad, -big).amax(-1)
    tt = times[None, None].expand(B, N, Ts)
    t_in = torch.where(hit, tt, big).amin(-1) - sc.time_margin_s
    last_hit = torch.where(hit, tt, -big).amax(-1)
    # If the set still meets the corridor at the end of the prediction, it occupies it indefinitely.
    t_out = torch.where(hit[..., -1], torch.full_like(last_hit, INF), last_hit + sc.time_margin_s)
    t_in = t_in.clamp(min=0.0)
    # Something already overlapping the ego footprint right now.
    now_overlap = agents_ok & (xy[..., 0] - r_cov < front_e) & (xy[..., 0] + r_cov > rear_e) & \
        (xy[..., 1].abs() < 0.5 * W_e[:, None] + 0.5 * W_a)
    geo_rad = r_cov[..., None] + (conf_rad if conf_rad is not None else torch.zeros_like(pred[..., 0]))
    return Interactions(path=path, path_s=path_s, lead_ok=lead_ok, lead_gap=lead_gap, lead_v=v_along,
                        rear_ok=rear_ok, rear_gap=rear_gap, rear_v=rear_v, conf_ok=conf_ok, z_lo=z_lo, z_hi=z_hi,
                        t_in=t_in, t_out=t_out, geo_pred=pred, geo_rad=geo_rad, geo_ok=agents_ok,
                        overlap_now=now_overlap.any(-1))


# ------------------------------------------------------------------------- interval kinematics


def _disp_bounds(v: torch.Tensor, a: torch.Tensor, dt: float):
    """Displacement and end speed over dt from speed v with constant accel a, stopping at 0."""
    v_end = v + a * dt
    stops = v_end < 0
    t_stop = torch.where(a < 0, v / (-a).clamp(min=1e-6), torch.full_like(v, dt))
    disp = torch.where(stops, 0.5 * v * t_stop.clamp(max=dt), v * dt + 0.5 * a * dt * dt)
    return disp.clamp(min=0.0), v_end.clamp(min=0.0)


def ego_tube(cfg: Config, v_lo, v_hi, a_cmd_lo, a_cmd_hi, a_prev, n_commit: int, n_tail: int):
    """Interval tube of (displacement, speed) under commanded accel intervals per sim step.

    a_cmd_lo/hi: (B, n_commit) commanded accelerations; the actuator follows with first-order lag
    from a_prev; after the commit horizon the backup command a_e_min is applied for n_tail steps.
    Returns s_lo, s_hi, v_lo, v_hi of shape (B, n_commit + n_tail) (state after each step).
    """
    dt = cfg.data.dt
    tau = cfg.sim.tau_a
    beta = 0.0 if tau <= 0 else max(0.0, 1.0 - dt / tau)
    w = cfg.hj.w_bar
    delay = cfg.sim.delay_steps
    s_lo = torch.zeros_like(v_lo)
    s_hi = torch.zeros_like(v_hi)
    act_lo = a_prev.clone()
    act_hi = a_prev.clone()
    out = []
    n = n_commit + n_tail
    for k in range(n):
        if k < delay:
            c_lo = c_hi = a_prev
        elif k < n_commit:
            c_lo, c_hi = a_cmd_lo[:, k], a_cmd_hi[:, k]
        else:
            c_lo = c_hi = torch.full_like(v_lo, cfg.hj.a_e_min)
        act_lo = c_lo + (act_lo - c_lo) * beta
        act_hi = c_hi + (act_hi - c_hi) * beta
        lo_a = torch.minimum(act_lo, act_hi) - w
        hi_a = torch.maximum(act_lo, act_hi) + w
        d_lo, v_lo = _disp_bounds(v_lo, lo_a, dt)
        d_hi, v_hi = _disp_bounds(v_hi, hi_a, dt)
        s_lo, s_hi = s_lo + d_lo, s_hi + d_hi
        out.append(torch.stack([s_lo, s_hi, v_lo, v_hi], -1))
    return torch.stack(out, 1)


def other_tube(cfg: Config, v_lo, v_hi, n: int):
    dt = cfg.data.dt
    s_lo = torch.zeros_like(v_lo)
    s_hi = torch.zeros_like(v_hi)
    out = []
    a_lo = torch.full_like(v_lo, cfg.hj.a_o_min)
    a_hi = torch.full_like(v_hi, cfg.hj.a_o_max)
    for _ in range(n):
        d_lo, v_lo = _disp_bounds(v_lo, a_lo, dt)
        d_hi, v_hi = _disp_bounds(v_hi, a_hi, dt)
        v_hi = v_hi.clamp(max=cfg.hj.v_max)
        s_lo, s_hi = s_lo + d_lo, s_hi + d_hi
        out.append(torch.stack([s_lo, s_hi, v_lo, v_hi], -1))
    return torch.stack(out, 1)


# ------------------------------------------------------------------------- shield


def _geom(dims: torch.Tensor):
    """dims (B, 3) = [length, width, reference->centre offset] -> front, rear (from reference), width."""
    front = dims[:, 2] + 0.5 * dims[:, 0]
    rear = dims[:, 2] - 0.5 * dims[:, 0]
    return front, rear, dims[:, 1]


def _subset(it: Interactions, idx: torch.Tensor) -> Interactions:
    return Interactions(**{k: getattr(it, k)[idx] for k in it.__dataclass_fields__})


def _repeat(it: Interactions, n: int) -> Interactions:
    return Interactions(**{k: getattr(it, k).repeat_interleave(n, 0) for k in it.__dataclass_fields__})


class Shield:
    def __init__(self, cfg: Config, hj_table=None, conformal_table: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.hj = hj_table
        self.conf = conformal_table
        self.r = steps_per_action(cfg)
        self.n_commit = max(1, int(round(cfg.shield.commit_s / cfg.data.dt)))
        tau = cfg.sim.tau_a
        self.n_tail = (int(math.ceil(4.0 * tau / cfg.data.dt)) if tau > 0 else 0) + cfg.sim.delay_steps + 1

    # ---------------------------------------------------------------- certificate check
    def check(self, it: Interactions, dims: torch.Tensor, v_lo, v_hi, u_lo, u_hi, a_prev, use_hj: bool,
              eps_pos: float, eps_vel: float, plan_lo=None, plan_hi=None) -> Dict[str, torch.Tensor]:
        """Certificate check of a control interval. Returns safe (B,) and the decisive margin (B,) in metres."""
        cfg = self.cfg
        dt = cfg.data.dt
        B = v_lo.shape[0]
        dev = v_lo.device
        front, rear, _ = _geom(dims)
        nc = self.n_commit
        idx = torch.clamp(torch.arange(nc, device=dev) // self.r, max=u_lo.shape[1] - 1)
        tube = ego_tube(cfg, v_lo, v_hi, u_lo[:, idx, 0], u_hi[:, idx, 0], a_prev, nc, self.n_tail)
        n = tube.shape[1]
        T_end = n * dt
        se_lo, se_hi, ve_lo, ve_hi = tube[..., 0], tube[..., 1], tube[..., 2], tube[..., 3]
        safe = ~it.overlap_now
        margin = torch.full((B,), INF, device=dev)
        N = it.lead_ok.shape[1]
        if bool(it.lead_ok.any()):
            vo_lo = (it.lead_v - eps_vel).clamp(min=0.0)
            vo_hi = it.lead_v + eps_vel
            if use_hj and self.hj is not None:
                ot = other_tube(cfg, vo_lo.reshape(-1), vo_hi.reshape(-1), n).view(B, N, n, 4)
                gap_lo = (it.lead_gap - eps_pos)[..., None] + ot[..., 0] - se_hi[:, None]
                v_end = self.hj.box_min(gap_lo[..., -1], ve_hi[:, -1:].expand(B, N), ot[..., -1, 2])
                path_m = (gap_lo - cfg.hj.d_min).amin(-1)
                m_lead = torch.minimum(v_end - cfg.shield.hj_margin, path_m)
            else:
                # Geometric baseline (no HJ): constant-velocity lead, plan tube over the check horizon.
                H_chk = int(round(cfg.shield.check_horizon_s / dt))
                if plan_lo is not None:
                    idx2 = torch.clamp(torch.arange(H_chk, device=dev) // self.r, max=plan_lo.shape[1] - 1)
                    full = ego_tube(cfg, v_lo, v_hi, plan_lo[:, idx2, 0], plan_hi[:, idx2, 0], a_prev, H_chk, 0)
                else:
                    full = tube
                tg = torch.arange(1, full.shape[1] + 1, device=dev).float() * dt
                g = (it.lead_gap - eps_pos)[..., None] + vo_lo[..., None] * tg - full[:, None, :, 1]
                m_lead = (g - cfg.hj.d_min).amin(-1)
            ok = (m_lead >= 0) | ~it.lead_ok
            safe = safe & ok.all(-1)
            margin = torch.minimum(margin, torch.where(it.lead_ok, m_lead, torch.full_like(m_lead, INF)).amin(-1))
        if bool(it.conf_ok.any()):
            tk = torch.arange(1, n + 1, device=dev).float() * dt
            in_win = (tk[None, None] >= it.t_in[..., None]) & (tk[None, None] <= it.t_out[..., None])
            occ_lo = se_lo[:, None] + rear[:, None, None] - eps_pos
            occ_hi = se_hi[:, None] + front[:, None, None] + eps_pos
            overlap = (occ_hi > it.z_lo[..., None]) & (occ_lo < it.z_hi[..., None])
            during = (in_win & overlap).any(-1)
            # (A) stop before the zone with the settled backup braking
            a_brk = max(-(cfg.hj.a_e_min + cfg.hj.w_bar), 0.1)
            front_end = (se_hi[:, -1] + front + ve_hi[:, -1] ** 2 / (2.0 * a_brk))[:, None]
            opt_a = front_end <= it.z_lo - cfg.hj.d_min
            # (B) clear the zone before it becomes occupied, with the slowest admissible progress
            a_go = cfg.hj.a_e_max - cfg.hj.w_bar
            need = ((it.z_hi + cfg.hj.d_min) - (se_lo[:, -1] + rear)[:, None]).clamp(min=0)
            vgo = ve_lo[:, -1:]
            if a_go > 1e-3:
                t_clr = (-vgo + torch.sqrt(vgo * vgo + 2 * a_go * need)) / a_go
            else:
                t_clr = need / vgo.clamp(min=1e-3)
            opt_b = (T_end + t_clr) <= it.t_in
            # (C) cannot reach the zone before it is vacated, even accelerating maximally
            a_max = cfg.hj.a_e_max + cfg.hj.w_bar
            dist_in = it.z_lo - (se_hi[:, -1] + front)[:, None]
            vh = ve_hi[:, -1:]
            t_arr = torch.where(dist_in <= 0, torch.zeros_like(dist_in),
                                (-vh + torch.sqrt(vh * vh + 2 * a_max * dist_in.clamp(min=0))) / a_max)
            opt_c = (T_end + t_arr) > it.t_out
            passed = it.t_out < T_end
            ok = ~during & (opt_a | opt_b | opt_c | passed)
            m_conf = torch.where(opt_a, it.z_lo - cfg.hj.d_min - front_end, torch.zeros_like(front_end))
            m_conf = torch.where(ok, m_conf.clamp(min=0), -torch.ones_like(m_conf))
            ok = ok | ~it.conf_ok
            safe = safe & ok.all(-1)
            margin = torch.minimum(margin, torch.where(it.conf_ok, m_conf, torch.full_like(m_conf, INF)).amin(-1))
        return dict(safe=safe, margin=margin)

    # ---------------------------------------------------------------- MPC
    def mpc(self, it: Interactions, dims: torch.Tensor, v0, ctrl_ref, traj_ref, use_hj: bool, geometric: bool,
            a_prev) -> torch.Tensor:
        """Batched augmented-Lagrangian MPC around the proposal. Returns (B, Ha, 2) controls."""
        cfg, sc, fc = self.cfg, self.cfg.shield, self.cfg.feat
        front, rear, width = _geom(dims)
        Hm = min(sc.mpc_steps, ctrl_ref.shape[1])
        ref_c = ctrl_ref[:, :Hm].detach().float()
        n_sim = Hm * self.r
        ref_p = traj_ref[:, :n_sim, :2].detach().float()
        a_rng = fc.a_max - fc.a_min
        za = torch.logit(((ref_c[..., 0] - fc.a_min) / a_rng).clamp(1e-3, 1 - 1e-3))
        zk = torch.atanh((ref_c[..., 1] / fc.kappa_max).clamp(-0.999, 0.999))
        z = torch.stack([za, zk], -1).clone().requires_grad_(True)
        dt = cfg.data.dt
        tk = torch.arange(1, n_sim + 1, device=v0.device).float() * dt
        lam = None
        rho = sc.rho0
        with torch.enable_grad():
            opt = torch.optim.Adam([z], lr=sc.mpc_lr)
            for _ in range(sc.mpc_outer):
                g_all = None
                for _ in range(sc.mpc_inner):
                    a = fc.a_min + a_rng * torch.sigmoid(z[..., 0])
                    k = fc.kappa_max * torch.tanh(z[..., 1])
                    tr = rollout(v0.float(), torch.stack([a, k], -1), cfg, n_steps=n_sim)
                    cost = sc.w_track * ((tr[..., :2] - ref_p) ** 2).sum(-1).mean(-1)
                    cost = cost + sc.w_ctrl_a * ((a - ref_c[..., 0]) ** 2).mean(-1)
                    cost = cost + sc.w_ctrl_k * ((k - ref_c[..., 1]) ** 2).mean(-1)
                    da = torch.diff(torch.cat([a_prev[:, None].float(), a], 1), dim=1)
                    cost = cost + sc.w_jerk * (da ** 2).mean(-1)
                    s_e = torch.cumsum(tr[..., 3] * dt, -1)
                    g_list = []
                    if bool(it.lead_ok.any()):
                        gap = it.lead_gap[..., None] + it.lead_v[..., None] * tk - s_e[:, None]
                        if use_hj and self.hj is not None:
                            val = self.hj.value_torch(gap, tr[..., 3][:, None].expand_as(gap),
                                                      it.lead_v[..., None].expand_as(gap))
                            g = sc.hj_margin - val
                        else:
                            g = cfg.hj.d_min - gap
                        g_list.append(torch.where(it.lead_ok[..., None], g, torch.full_like(g, -1.0)).flatten(1))
                    if bool(it.conf_ok.any()):
                        ins_a = (s_e[:, None] + front[:, None, None]) - (it.z_lo[..., None] - cfg.hj.d_min)
                        ins_b = (it.z_hi[..., None] + cfg.hj.d_min) - (s_e[:, None] + rear[:, None, None])
                        g = torch.minimum(ins_a, ins_b)
                        win = (tk >= it.t_in[..., None]) & (tk <= it.t_out[..., None]) & it.conf_ok[..., None]
                        g_list.append(torch.where(win, g, torch.full_like(g, -1.0)).flatten(1))
                    if geometric:
                        Hg = min(n_sim, it.geo_pred.shape[2])
                        dist = torch.linalg.norm(tr[:, None, :Hg, :2] - it.geo_pred[:, :, :Hg], dim=-1)
                        need = it.geo_rad[:, :, :Hg] + 0.5 * width[:, None, None] + 0.5
                        g = need - dist
                        g_list.append(torch.where(it.geo_ok[..., None], g, torch.full_like(g, -1.0)).flatten(1))
                    if bool(it.rear_ok.any()):
                        close = (it.rear_ok & (it.rear_gap < 10.0)).any(-1).float()
                        cost = cost + sc.rear_soft_weight * (close[:, None] * F.relu(-a - 3.0) ** 2).mean(-1)
                    loss = cost
                    if g_list:
                        g_all = torch.cat(g_list, 1)
                        if lam is None:
                            lam = torch.zeros_like(g_all)
                        psi = torch.where(lam + rho * g_all > 0, lam * g_all + 0.5 * rho * g_all ** 2,
                                          -lam ** 2 / (2 * rho))
                        loss = cost + psi.sum(-1)
                    opt.zero_grad(set_to_none=True)
                    loss.sum().backward()
                    opt.step()
                if g_all is not None:
                    lam = (lam + rho * g_all.detach()).clamp(min=0.0)
                    rho = rho * 2.0
        with torch.no_grad():
            a = fc.a_min + a_rng * torch.sigmoid(z[..., 0])
            k = fc.kappa_max * torch.tanh(z[..., 1])
            u = torch.stack([a, k], -1).detach()
        if Hm < ctrl_ref.shape[1]:
            u = torch.cat([u, ctrl_ref[:, Hm:].float()], 1)
        return u

    # ---------------------------------------------------------------- fallback
    def fallback(self, it: Interactions, dims, v_lo, v_hi, a_nom, kappa_nom, a_prev, use_hj: bool, eps_pos, eps_vel):
        """HJ least-restrictive backup: the admissible acceleration closest to the proposal; if none is
        admissible, the one with the largest worst-case margin (reported as an emergency)."""
        cfg = self.cfg
        B = v_lo.shape[0]
        n_c = cfg.shield.fallback_candidates
        cands = torch.linspace(cfg.hj.a_e_min, cfg.hj.a_e_max, n_c, device=v_lo.device)
        rep = lambda x: x.repeat_interleave(n_c, 0)
        u = torch.stack([cands.repeat(B), rep(kappa_nom)], -1)[:, None]
        res = self.check(_repeat(it, n_c), rep(dims), rep(v_lo), rep(v_hi), u, u, rep(a_prev), use_hj,
                         eps_pos, eps_vel)
        safe = res["safe"].view(B, n_c)
        margin = res["margin"].view(B, n_c)
        any_safe = safe.any(-1)
        dist = (cands[None] - a_nom[:, None]).abs()
        best_safe = torch.where(safe, dist, torch.full_like(dist, INF)).argmin(-1)
        best_margin = margin.argmax(-1)
        pick = torch.where(any_safe, best_safe, best_margin)
        return torch.stack([cands[pick], kappa_nom], -1), any_safe

    # ---------------------------------------------------------------- one decision
    @torch.no_grad()
    def step(self, method: Dict, v0: torch.Tensor, ctrl_nom: torch.Tensor, traj_nom: torch.Tensor,
             u_lo: Optional[torch.Tensor], u_hi: Optional[torch.Tensor], agents_phys: torch.Tensor,
             agents_ok: torch.Tensor, pred: torch.Tensor, conf_rad: Optional[torch.Tensor], ego_dims: torch.Tensor,
             a_prev: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B = v0.shape[0]
        dev = v0.device
        dims = ego_dims.float()
        use_cert = bool(method.get("cert")) and u_lo is not None
        use_hj = bool(method.get("hj"))
        use_mpc = bool(method.get("mpc"))
        geometric = bool(method.get("geometric"))
        eps_pos = cfg.cert.eps_agent_pos if use_cert else 0.0
        eps_vel = cfg.cert.eps_agent_vel if use_cert else 0.0
        eps_v = cfg.cert.eps_ego_speed if use_cert else 0.0
        v0 = v0.float()
        ctrl_nom = ctrl_nom.float()
        v_lo = (v0 - eps_v).clamp(min=0.0)
        v_hi = v0 + eps_v
        lo = u_lo.float() if use_cert else ctrl_nom
        hi = u_hi.float() if use_cert else ctrl_nom
        it = build_interactions(cfg, traj_nom.float(), v0, agents_phys.float(), agents_ok, pred.float(),
                                None if conf_rad is None else conf_rad.float(), dims)
        cmd = ctrl_nom[:, 0].clone()
        source = torch.zeros(B, dtype=torch.long, device=dev)
        certified = torch.zeros(B, dtype=torch.bool, device=dev)
        margin = torch.full((B,), INF, device=dev)
        base = dict(n_lead=it.lead_ok.sum(-1), n_conf=it.conf_ok.sum(-1), n_rear=it.rear_ok.sum(-1))
        if not (use_hj or use_cert or use_mpc):
            # Unshielded policy (B1); still report whether its proposal would have passed the HJ check.
            if self.hj is not None:
                margin = self.check(it, dims, v_lo, v_hi, ctrl_nom, ctrl_nom, a_prev, True, 0.0, 0.0)["margin"]
            return dict(cmd=cmd, source=source, certified=certified, margin=margin, **base)
        if method.get("mpc_always"):
            u = self.mpc(it, dims, v0, ctrl_nom, traj_nom.float(), use_hj, True, a_prev)
            return dict(cmd=u[:, 0], source=torch.full_like(source, SRC_MPC), certified=certified, margin=margin,
                        **base)
        res = self.check(it, dims, v_lo, v_hi, lo, hi, a_prev, use_hj, eps_pos, eps_vel, plan_lo=lo, plan_hi=hi)
        certified = res["safe"]
        margin = res["margin"].clone()
        need = ~certified
        if bool(need.any()) and use_mpc:
            idx = need.nonzero().squeeze(-1)
            sub = _subset(it, idx)
            u_sub = self.mpc(sub, dims[idx], v0[idx], ctrl_nom[idx], traj_nom[idx].float(), use_hj, geometric,
                             a_prev[idx])
            res2 = self.check(sub, dims[idx], v_lo[idx], v_hi[idx], u_sub, u_sub, a_prev[idx], use_hj,
                              eps_pos, eps_vel)
            ok2 = res2["safe"]
            take = idx[ok2]
            cmd[take] = u_sub[ok2, 0]
            source[take] = SRC_MPC
            margin[idx] = torch.where(ok2, res2["margin"], margin[idx])
            need = need.clone()
            need[take] = False
        if bool(need.any()):
            idx = need.nonzero().squeeze(-1)
            if method.get("fallback") == "hj":
                u_fb, any_safe = self.fallback(_subset(it, idx), dims[idx], v_lo[idx], v_hi[idx], ctrl_nom[idx, 0, 0],
                                               ctrl_nom[idx, 0, 1], a_prev[idx], use_hj, eps_pos, eps_vel)
                cmd[idx] = u_fb
                source[idx] = torch.where(any_safe, torch.full_like(idx, SRC_FALLBACK),
                                          torch.full_like(idx, SRC_EMERGENCY))
            else:
                cmd[idx, 0] = cfg.hj.a_e_min
                source[idx] = SRC_FALLBACK
        return dict(cmd=cmd, source=source, certified=certified, margin=margin, **base)
