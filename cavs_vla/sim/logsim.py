"""Batched closed-loop log-replay simulator (nuPlan / WOMD scenes) with an explicit plant.

* Controlled vehicles (the ego, plus num_controlled-1 logged vehicles for the
  multi-CAV experiment) are driven by policy -> shield -> plant (actuator lag,
  bounded noise, optional delay) at 10 Hz.
* Other agents either replay the log (non-reactive) or follow their logged path
  with IDM speed control w.r.t. any actor ahead of them, including controlled
  vehicles (reactive, CLS-R style). Pedestrians and static objects replay.
* Planning runs every ``replan_interval`` steps and its output becomes active
  ``latency_steps`` later; prediction and the shield run every step on the
  measured state, so plan staleness is handled explicitly.
* Multi-CAV coordination: controlled vehicles broadcast their active plan.
  Receivers replace their *prediction* of a controlled vehicle with the received
  plan (time-shifted by the message age, inflated by v*age), and a
  first-come-first-served rule decides who yields at a shared conflict point.
* Every step records collisions (oriented boxes, at-fault classification),
  box distances, time-to-collision, drivable-area violations, progress,
  comfort, interventions, certificate statistics, HJ margins and latency.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import METHOD_TABLE, Config
from ..dynamics import Plant
from ..features import featurize, steps_per_action, wrap
from ..geometry_torch import box_corners, box_distance, boxes_overlap, point_polyline_distance
from ..language import ATTR_CONNECTOR, ATTR_LANE
from ..safety.conformal import radius_torch
from ..safety.shield import SRC_EMERGENCY, SRC_FALLBACK, SRC_MPC, Shield
from ..safety.verifier import certify
from ..utils import LatencyMeter
from ..train import autocast_ctx


def _to_torch(scene_np: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in scene_np.items():
        t = torch.as_tensor(np.ascontiguousarray(v))
        if t.dtype == torch.float64 and k != "t0":
            t = t.float()
        out[k] = t.to(device)
    return out


class LogSim:
    def __init__(self, cfg: Config, model, hj_table=None, conformal_table: Optional[dict] = None,
                 device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hj = hj_table
        self.conf = conformal_table
        self.shield = Shield(cfg, hj_table, conformal_table)
        self.latency = LatencyMeter()

    # ------------------------------------------------------------------ helpers
    def _choose_controlled(self, sc: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        K = cfg.data.hist_steps
        B, A = sc["atype"].shape
        C = cfg.sim.num_controlled
        idx = torch.full((B, C), -1, dtype=torch.long, device=self.device)
        idx[:, 0] = 0
        if C > 1:
            full = sc["valid"][:, :, K - 1:].all(-1) & (sc["atype"] == 0)
            full[:, 0] = False
            d = torch.hypot(sc["traj"][:, :, K - 1, 0], sc["traj"][:, :, K - 1, 1])
            d = torch.where(full, d, torch.full_like(d, float("inf")))
            k = min(C - 1, A - 1)
            dv, di = torch.topk(d, k=k, dim=1, largest=False)
            idx[:, 1:1 + k] = torch.where(torch.isfinite(dv), di, torch.full_like(di, -1))
        return idx

    def _idm_paths(self, sc):
        """Logged path of every actor: positions over valid frames (held when invalid) + arclength."""
        traj, valid = sc["traj"], sc["valid"]
        T = traj.shape[2]
        pos = traj[..., :2].clone()
        # forward-fill invalid frames, back-fill leading invalid frames
        for t in range(1, T):
            pos[:, :, t] = torch.where(valid[:, :, t, None], pos[:, :, t], pos[:, :, t - 1])
        for t in range(T - 2, -1, -1):
            first_valid_later = ~valid[:, :, t] & valid[:, :, t + 1]
            pos[:, :, t] = torch.where(first_valid_later[..., None], pos[:, :, t + 1], pos[:, :, t])
        seg = torch.linalg.norm(torch.diff(pos, dim=2), dim=-1)
        s = torch.cat([torch.zeros_like(seg[..., :1]), torch.cumsum(seg, -1)], -1)
        return pos, s

    @staticmethod
    def _interp_path(pos, s_cum, s_q):
        """pos (B,A,T,2), s_cum (B,A,T), s_q (B,A) -> point and heading along the path (extrapolated)."""
        T = pos.shape[2]
        idx = (torch.searchsorted(s_cum.contiguous(), s_q[..., None].contiguous(), right=True) - 1).clamp(0, T - 2)
        s0 = torch.gather(s_cum, 2, idx).squeeze(-1)
        s1 = torch.gather(s_cum, 2, idx + 1).squeeze(-1)
        p0 = torch.gather(pos, 2, idx[..., None].expand(-1, -1, -1, 2)).squeeze(2)
        p1 = torch.gather(pos, 2, (idx + 1)[..., None].expand(-1, -1, -1, 2)).squeeze(2)
        seg = p1 - p0
        l = (s1 - s0).clamp(min=1e-6)
        point = p0 + ((s_q - s0) / l)[..., None] * seg
        heading = torch.atan2(seg[..., 1], seg[..., 0])
        return point, heading, (s1 - s0) > 1e-3

    # ------------------------------------------------------------------ main loop
    @torch.no_grad()
    def run(self, scene_np: Dict[str, np.ndarray], method_name: str, seed: int = 0) -> List[Dict]:
        cfg = self.cfg
        dev = self.device
        method = METHOD_TABLE[method_name]
        d, s_cfg = cfg.data, cfg.sim
        K, H = d.hist_steps, d.fut_steps
        T = K + H
        dt = d.dt
        r = steps_per_action(cfg)
        gen = torch.Generator(device=dev).manual_seed(seed)
        sc = _to_torch(scene_np, dev)
        log_traj = sc["traj"].clone()
        log_valid = sc["valid"].clone()
        traj = sc["traj"].clone()
        valid = sc["valid"].clone()
        B, A = sc["atype"].shape
        C = s_cfg.num_controlled
        cidx = self._choose_controlled(sc)                      # (B, C)
        cok = cidx >= 0
        cflat = cidx.clamp(min=0).reshape(-1)
        bflat = torch.arange(B, device=dev).repeat_interleave(C)
        is_ctrl = torch.zeros(B, A, dtype=torch.bool, device=dev)
        is_ctrl[bflat[cok.reshape(-1)], cflat[cok.reshape(-1)]] = True
        expert = bool(method.get("expert"))

        st = traj[bflat, cflat, K - 1]
        state = torch.stack([st[:, 0], st[:, 1], st[:, 2], torch.hypot(st[:, 3], st[:, 4])], -1)
        v_next_log = torch.hypot(traj[bflat, cflat, K, 3], traj[bflat, cflat, K, 4])
        a0 = ((v_next_log - state[:, 3]) / dt).clamp(cfg.feat.a_min, cfg.feat.a_max)
        plant = Plant(cfg, B * C, dev, gen)
        plant.reset(a0, torch.zeros_like(a0))
        a_prev = a0.clone()
        idm_pos, idm_s = self._idm_paths(sc)
        s_agent = idm_s[:, :, K - 1].clone()
        v_agent = torch.hypot(traj[:, :, K - 1, 3], traj[:, :, K - 1, 4])
        reactive = (sc["atype"] <= 2) & (sc["atype"] >= 0) & ~is_ctrl & bool(s_cfg.reactive)

        n_sel = B * C
        active = None
        queue: List[Dict] = []
        Ha = H // r
        rec = _Recorder(cfg, B, C, dev)
        broadcast = None
        for t in range(K - 1, T - 1):
            rep = {k: v.repeat_interleave(C, 0) for k, v in sc.items() if k not in ("traj", "valid")}
            rep["traj"] = traj.repeat_interleave(C, 0)
            rep["valid"] = valid.repeat_interleave(C, 0)
            with self.latency.measure("featurize"):
                feat = featurize(rep, t, cflat, cfg)
            with self.latency.measure("policy"), autocast_ctx(dev, torch.bfloat16):
                out = self.model(feat)
            out = {k: v.float() if torch.is_floating_point(v) else v for k, v in out.items()}
            sel = out["mode_logits"].argmax(-1)
            ii = torch.arange(n_sel, device=dev)
            plan = dict(ctrl=out["ctrl"][ii, sel], traj=out["traj"][ii, sel], origin=state[:, :3].clone(), t=t)
            if method.get("cert"):
                with self.latency.measure("certify"):
                    cert = certify(self.model, feat, cfg)
                plan["u_lo"], plan["u_hi"] = cert["u_lo"], cert["u_hi"]
                rec.cert_width.append((cert["u_hi"][:, 0] - cert["u_lo"][:, 0]).cpu())
            step_idx = t - (K - 1)
            if step_idx % s_cfg.replan_interval == 0:
                queue.append(dict(plan=plan, activate=t + s_cfg.latency_steps))
            ready = [q for q in queue if q["activate"] <= t]
            if ready:
                active = ready[-1]["plan"]
                queue = [q for q in queue if q["activate"] > t]
            if active is None:
                active = plan
            age = t - active["t"]
            o = min(age // r, Ha - 1)
            ctrl_s = torch.cat([active["ctrl"][:, o:], active["ctrl"][:, -1:].expand(-1, o, -1)], 1)
            tr = active["traj"]
            tr_s = torch.cat([tr[:, age:], tr[:, -1:].expand(-1, age, -1)], 1) if age > 0 else tr
            tr_s = _reframe(tr_s, active["origin"], state[:, :3])
            u_lo = u_hi = None
            if method.get("cert"):
                u_lo = torch.cat([active["u_lo"][:, o:], active["u_lo"][:, -1:].expand(-1, o, -1)], 1)
                u_hi = torch.cat([active["u_hi"][:, o:], active["u_hi"][:, -1:].expand(-1, o, -1)], 1)
            pred = out["agent_pred"]
            agents_ok = feat["agents_ok"].clone()
            rad = radius_torch(self.conf, out["agent_logscale"], feat["agents_phys"][..., 7]) if self.conf else None
            if C > 1 and broadcast is not None:
                pred, rad, agents_ok = self._coordinate(feat, pred, rad, agents_ok, broadcast, cidx, state, tr_s, t)
            if expert:
                cmd = None
                res = dict(source=torch.zeros(n_sel, dtype=torch.long, device=dev),
                           certified=torch.zeros(n_sel, dtype=torch.bool, device=dev),
                           margin=torch.full((n_sel,), float("nan"), device=dev))
            else:
                with self.latency.measure("shield"):
                    res = self.shield.step(method, state[:, 3], ctrl_s, tr_s, u_lo, u_hi, feat["agents_phys"],
                                           agents_ok, pred, rad, feat["ego_dims"], a_prev)
                cmd = res["cmd"]
            if t % 10 == 0:
                rec.store_prediction(t, feat, out, rad, state)
            # ---- advance controlled vehicles
            if expert:
                nxt = log_traj[bflat, cflat, t + 1]
                new_state = torch.stack([nxt[:, 0], nxt[:, 1], nxt[:, 2], torch.hypot(nxt[:, 3], nxt[:, 4])], -1)
                cmd_exec = torch.stack([(new_state[:, 3] - state[:, 3]) / dt, torch.zeros_like(state[:, 3])], -1)
            else:
                new_state = plant.step(state, cmd)
                cmd_exec = cmd
            ok_flat = cok.reshape(-1)
            new_state = torch.where(ok_flat[:, None], new_state, state)
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 0] = new_state[ok_flat, 0]
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 1] = new_state[ok_flat, 1]
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 2] = new_state[ok_flat, 2]
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 3] = new_state[ok_flat, 3] * torch.cos(new_state[ok_flat, 2])
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 4] = new_state[ok_flat, 3] * torch.sin(new_state[ok_flat, 2])
            traj[bflat[ok_flat], cflat[ok_flat], t + 1, 5:7] = traj[bflat[ok_flat], cflat[ok_flat], t, 5:7]
            valid[bflat[ok_flat], cflat[ok_flat], t + 1] = True
            # ---- other agents
            if s_cfg.reactive:
                s_agent, v_agent = self._idm_step(traj, valid, log_valid, idm_pos, idm_s, s_agent, v_agent,
                                                  reactive, t, sc)
            rec.step(t, traj, valid, sc, cidx, cok, state, new_state, cmd_exec, ctrl_s, res, log_traj)
            a_prev = cmd_exec[:, 0] if not expert else a_prev
            # Broadcast the plan that was just used, expressed in the frame of the pose it was reframed to (time t).
            broadcast = dict(traj=tr_s, origin=state[:, :3].clone(), t=t,
                             spread=(None if u_lo is None else (u_hi[:, :, 0] - u_lo[:, :, 0])))
            state = new_state
        return rec.finalize(traj, valid, sc, cidx, cok, log_traj, method_name, seed, self.conf)

    # ------------------------------------------------------------------ IDM agents
    def _idm_step(self, traj, valid, log_valid, pos, s_cum, s_agent, v_agent, reactive, t, sc):
        cfg = self.cfg.sim
        dt = self.cfg.data.dt
        B, A = s_agent.shape
        x, y, yaw = traj[:, :, t, 0], traj[:, :, t, 1], traj[:, :, t, 2]
        L = traj[:, :, t, 5].clamp(min=0.5)
        present = valid[:, :, t]
        # gap to the closest present actor ahead in a 1.8 m lateral band of each agent's heading
        dx = x[:, None, :] - x[:, :, None]
        dy = y[:, None, :] - y[:, :, None]
        c, s = torch.cos(yaw)[:, :, None], torch.sin(yaw)[:, :, None]
        lon = dx * c + dy * s
        lat = -dx * s + dy * c
        cand = present[:, None, :] & (lon > 0) & (lat.abs() < 1.8)
        cand = cand & ~torch.eye(A, dtype=torch.bool, device=x.device)[None]
        gap = lon - 0.5 * (L[:, :, None] + L[:, None, :])
        gap = torch.where(cand, gap, torch.full_like(gap, 1e4))
        g, j = gap.min(-1)
        v_all = torch.hypot(traj[:, :, t, 3], traj[:, :, t, 4])
        v_lead = torch.gather(v_all, 1, j)
        v_log = torch.hypot(sc["traj"][:, :, t + 1, 3], sc["traj"][:, :, t + 1, 4])
        v0 = v_log + 0.5
        dv = v_agent - v_lead
        s_star = cfg.idm_s0 + v_agent * cfg.idm_T + v_agent * dv / (2 * (cfg.idm_a * cfg.idm_b) ** 0.5)
        acc = cfg.idm_a * (1 - (v_agent / v0.clamp(min=0.1)) ** cfg.idm_delta
                           - (s_star.clamp(min=0) / g.clamp(min=0.1)) ** 2)
        acc = acc.clamp(-8.0, cfg.idm_a * 2)
        v_new = (v_agent + acc * dt).clamp(min=0.0)
        s_new = s_agent + 0.5 * (v_agent + v_new) * dt
        point, heading, moving = self._interp_path(pos, s_cum, s_new)
        upd = reactive & log_valid[:, :, t + 1]
        heading = torch.where(moving, heading, traj[:, :, t, 2])
        traj[:, :, t + 1, 0] = torch.where(upd, point[..., 0], traj[:, :, t + 1, 0])
        traj[:, :, t + 1, 1] = torch.where(upd, point[..., 1], traj[:, :, t + 1, 1])
        traj[:, :, t + 1, 2] = torch.where(upd, heading, traj[:, :, t + 1, 2])
        traj[:, :, t + 1, 3] = torch.where(upd, v_new * torch.cos(heading), traj[:, :, t + 1, 3])
        traj[:, :, t + 1, 4] = torch.where(upd, v_new * torch.sin(heading), traj[:, :, t + 1, 4])
        # agents that just appeared start from their logged state
        appear = reactive & log_valid[:, :, t + 1] & ~log_valid[:, :, t]
        s_new = torch.where(appear, s_cum[:, :, t + 1], s_new)
        v_new = torch.where(appear, v_log, v_new)
        return torch.where(reactive, s_new, s_agent), torch.where(reactive, v_new, v_agent)

    # ------------------------------------------------------------------ multi-CAV coordination
    def _coordinate(self, feat, pred, rad, agents_ok, broadcast, cidx, state, my_traj, t):
        """Replace predictions of other controlled vehicles by their broadcast plans (FCFS priority)."""
        cfg = self.cfg
        dt = cfg.data.dt
        B, C = cidx.shape
        n = B * C
        H = pred.shape[2]
        age = t - broadcast["t"]
        aidx = feat["agents_idx"]                                   # (n, N) raw actor ids in the receiver's scene
        recv_b = torch.arange(n, device=pred.device) // C
        # Broadcast plans in world frame: (n, H, 2)
        bt = broadcast["traj"]
        world = _to_world(bt, broadcast["origin"])
        world = torch.cat([world[:, age:], world[:, -1:].expand(-1, age, -1)], 1) if age > 0 else world
        # receiver frame
        ox, oy, oyaw = state[:, 0], state[:, 1], state[:, 2]
        c, s = torch.cos(oyaw), torch.sin(oyaw)
        my_world = _to_world(my_traj, state[:, :3])
        pred = pred.clone()
        rad = rad.clone() if rad is not None else torch.zeros_like(pred[..., 0])
        for k in range(C):
            sender_actor = cidx[:, k]                               # (B,)
            sender_flat = torch.arange(B, device=pred.device) * C + k
            match = (aidx == sender_actor[recv_b][:, None]) & (sender_actor[recv_b] >= 0)[:, None]
            if not bool(match.any()):
                continue
            sw = world[sender_flat][recv_b]                         # (n, H, 2)
            dx = sw[..., 0] - ox[:, None]
            dy = sw[..., 1] - oy[:, None]
            loc = torch.stack([dx * c[:, None] + dy * s[:, None], -dx * s[:, None] + dy * c[:, None]], -1)
            v_s = state[sender_flat][recv_b][:, 3]
            infl = 0.5 + v_s * age * dt
            if broadcast["spread"] is not None:
                spread = broadcast["spread"][sender_flat][recv_b]   # (n, Ha) accel interval width
                tt = torch.arange(1, H + 1, device=pred.device).float() * dt
                infl = infl[:, None] + 0.25 * spread.mean(-1, keepdim=True) * tt ** 2
            else:
                infl = infl[:, None].expand(-1, H)
            # FCFS: who reaches the closest approach point first keeps priority.
            dmat = torch.cdist(my_world, sw)                        # (n, H, H)
            dmin, flat = dmat.flatten(1).min(-1)
            i_me, i_other = flat // H, flat % H
            conflict = dmin < 4.0
            i_have_priority = conflict & ((i_me < i_other - int(0.5 / dt)) |
                                          ((i_me - i_other).abs() <= int(0.5 / dt)) &
                                          (torch.arange(n, device=pred.device) % C < k))
            for slot in range(aidx.shape[1]):
                m = match[:, slot]
                if not bool(m.any()):
                    continue
                pred[m, slot] = loc[m]
                rad[m, slot] = infl[m]
                agents_ok[m, slot] = agents_ok[m, slot] & ~i_have_priority[m]
        return pred, rad, agents_ok


def _to_world(traj_local: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(origin[:, 2])[:, None], torch.sin(origin[:, 2])[:, None]
    x = origin[:, 0:1] + traj_local[..., 0] * c - traj_local[..., 1] * s
    y = origin[:, 1:2] + traj_local[..., 0] * s + traj_local[..., 1] * c
    return torch.stack([x, y], -1)


def _reframe(traj_local: torch.Tensor, origin: torch.Tensor, new_origin: torch.Tensor) -> torch.Tensor:
    """Express a plan stored in the frame ``origin`` in the frame ``new_origin`` (x, y, psi, v)."""
    w = _to_world(traj_local, origin)
    c, s = torch.cos(new_origin[:, 2])[:, None], torch.sin(new_origin[:, 2])[:, None]
    dx = w[..., 0] - new_origin[:, 0:1]
    dy = w[..., 1] - new_origin[:, 1:2]
    x = dx * c + dy * s
    y = -dx * s + dy * c
    psi = wrap(traj_local[..., 2] + origin[:, 2:3] - new_origin[:, 2:3])
    return torch.stack([x, y, psi, traj_local[..., 3]], -1)


class _Recorder:
    """Per-step metric accumulation; ``finalize`` returns one dict per (scenario, controlled vehicle)."""

    def __init__(self, cfg: Config, B: int, C: int, device) -> None:
        self.cfg = cfg
        self.B, self.C = B, C
        n = B * C
        z = lambda: torch.zeros(n, device=device)
        self.collided = torch.zeros(n, dtype=torch.bool, device=device)
        self.at_fault = torch.zeros(n, dtype=torch.bool, device=device)
        self.coll_time = torch.full((n,), float("nan"), device=device)
        self.coll_speed = torch.full((n,), float("nan"), device=device)
        self.min_dist = torch.full((n,), float("inf"), device=device)
        self.min_ttc = torch.full((n,), float("inf"), device=device)
        self.offroad_steps = z()
        self.max_lane_dev = z()
        self.steps = 0
        self.speed_sum = z()
        self.max_acc = torch.full((n,), -float("inf"), device=device)
        self.min_acc = torch.full((n,), float("inf"), device=device)
        self.max_jerk = z()
        self.max_lat_acc = z()
        self.comfort_viol = z()
        self.interv = z()
        self.mpc = z()
        self.fallback = z()
        self.emergency = z()
        self.da_sum = z()
        self.dk_sum = z()
        self.certified = z()
        self.min_margin = torch.full((n,), float("inf"), device=device)
        self.stationary = z()
        self.deadlock = torch.zeros(n, dtype=torch.bool, device=device)
        self.prev_a = None
        self.cert_width: List[torch.Tensor] = []
        self.preds: List = []
        self.dist_travel = z()

    def store_prediction(self, t, feat, out, rad, observer_pose):
        self.preds.append((t, feat["agents_idx"].clone(), out["agent_pred"].clone(),
                           None if rad is None else rad.clone(), feat["agents_ok"].clone(),
                           observer_pose[:, :3].clone()))

    def step(self, t, traj, valid, sc, cidx, cok, state, new_state, cmd, ctrl_s, res, log_traj):
        cfg = self.cfg
        dt = cfg.data.dt
        B, C = self.B, self.C
        dev = traj.device
        n = B * C
        bflat = torch.arange(B, device=dev).repeat_interleave(C)
        cflat = cidx.clamp(min=0).reshape(-1)
        okf = cok.reshape(-1)
        nxt = traj[:, :, t + 1]                                   # (B, A, 7)
        off = sc["center_offset"]
        cx = nxt[..., 0] + off * torch.cos(nxt[..., 2])
        cy = nxt[..., 1] + off * torch.sin(nxt[..., 2])
        corners = box_corners(cx, cy, nxt[..., 2], nxt[..., 5], nxt[..., 6])   # (B, A, 4, 2)
        mine = corners[bflat, cflat]                               # (n, 4, 2)
        others = corners[bflat]                                    # (n, A, 4, 2)
        present = valid[bflat, :, t + 1] & (sc["atype"][bflat] >= 0)
        present[torch.arange(n, device=dev), cflat] = False
        ov = boxes_overlap(mine[:, None], others) & present
        dist = box_distance(mine[:, None], others)
        dist = torch.where(present, dist, torch.full_like(dist, float("inf")))
        self.min_dist = torch.minimum(self.min_dist, dist.amin(-1))
        hit = ov.any(-1) & okf
        new_hit = hit & ~self.collided
        if bool(new_hit.any()):
            # At fault unless the other vehicle came from behind while the ego was slower, or the ego was stopped.
            yaw = new_state[:, 2]
            rel_x = (cx[bflat] - (new_state[:, 0] + off[bflat, cflat] * torch.cos(yaw))[:, None]) * torch.cos(yaw)[:, None] \
                + (cy[bflat] - (new_state[:, 1] + off[bflat, cflat] * torch.sin(yaw))[:, None]) * torch.sin(yaw)[:, None]
            v_o = torch.hypot(nxt[bflat, :, 3], nxt[bflat, :, 4])
            from_behind = (rel_x < -0.5 * nxt[bflat, cflat, 5][:, None]) & (v_o > new_state[:, 3:4])
            stopped = new_state[:, 3] < 0.2
            fault = (ov & ~from_behind).any(-1) & ~stopped
            self.at_fault = self.at_fault | (new_hit & fault)
            self.coll_time = torch.where(new_hit, torch.full_like(self.coll_time, (t + 1) * dt), self.coll_time)
            self.coll_speed = torch.where(new_hit, new_state[:, 3], self.coll_speed)
        self.collided = self.collided | hit
        # time to collision (constant velocity, 0..3 s)
        vx = nxt[..., 3]
        vy = nxt[..., 4]
        ttc = torch.full((n,), float("inf"), device=dev)
        for k in range(1, 11):
            tau = 0.3 * k
            cxk = cx + vx * tau
            cyk = cy + vy * tau
            ck = box_corners(cxk, cyk, nxt[..., 2], nxt[..., 5], nxt[..., 6])
            ovk = (boxes_overlap(ck[bflat, cflat][:, None], ck[bflat]) & present).any(-1)
            ttc = torch.where(ovk & torch.isinf(ttc), torch.full_like(ttc, tau), ttc)
        self.min_ttc = torch.minimum(self.min_ttc, ttc)
        # drivable area: distance to the nearest lane / connector centreline
        poly = sc["poly"][bflat]
        lane = (sc["poly_attr"][bflat][..., ATTR_LANE] + sc["poly_attr"][bflat][..., ATTR_CONNECTOR]) > 0
        pv = sc["poly_valid"][bflat] & lane[..., None]
        has_map = pv.any(-1).any(-1)
        dev_lane = point_polyline_distance(new_state[:, :2], poly, pv)
        dev_lane = torch.where(has_map, dev_lane, torch.zeros_like(dev_lane))
        self.max_lane_dev = torch.maximum(self.max_lane_dev, dev_lane)
        thr = cfg.sim.lane_half_width + cfg.sim.offroad_tolerance
        self.offroad_steps += (dev_lane > thr).float()
        # kinematics / comfort
        v = new_state[:, 3]
        self.speed_sum += v
        self.dist_travel += 0.5 * (state[:, 3] + v) * dt
        a = cmd[:, 0]
        self.max_acc = torch.maximum(self.max_acc, a)
        self.min_acc = torch.minimum(self.min_acc, a)
        if self.prev_a is not None:
            jerk = (a - self.prev_a).abs() / dt
            self.max_jerk = torch.maximum(self.max_jerk, jerk)
            self.comfort_viol += ((a.abs() > 4.0) | (jerk > 4.0)).float()
        self.prev_a = a
        lat = v * v * cmd[:, 1].abs()
        self.max_lat_acc = torch.maximum(self.max_lat_acc, lat)
        src = res["source"]
        self.interv += (src > 0).float()
        self.mpc += (src == SRC_MPC).float()
        self.fallback += (src == SRC_FALLBACK).float()
        self.emergency += (src == SRC_EMERGENCY).float()
        self.da_sum += (cmd[:, 0] - ctrl_s[:, 0, 0]).abs()
        self.dk_sum += (cmd[:, 1] - ctrl_s[:, 0, 1]).abs()
        self.certified += res["certified"].float()
        m = res["margin"]
        self.min_margin = torch.minimum(self.min_margin, torch.where(torch.isfinite(m), m, self.min_margin))
        # deadlock: stopped while the log was moving and nothing within 10 m ahead
        log_v = torch.hypot(log_traj[bflat, cflat, t + 1, 3], log_traj[bflat, cflat, t + 1, 4])
        ahead_free = ~((dist < 10.0) & present).any(-1)
        stuck = (v < 0.1) & (log_v > 1.0) & ahead_free
        self.stationary = torch.where(stuck, self.stationary + dt, torch.zeros_like(self.stationary))
        self.deadlock = self.deadlock | (self.stationary >= cfg.sim.deadlock_s)
        self.steps += 1

    def finalize(self, traj, valid, sc, cidx, cok, log_traj, method_name, seed, conf_table) -> List[Dict]:
        cfg = self.cfg
        K = cfg.data.hist_steps
        dt = cfg.data.dt
        B, C = self.B, self.C
        dev = traj.device
        n = B * C
        bflat = torch.arange(B, device=dev).repeat_interleave(C)
        cflat = cidx.clamp(min=0).reshape(-1)
        # progress along the expert (logged) path
        exp = log_traj[bflat, cflat, K - 1:, :2]
        seg = torch.linalg.norm(torch.diff(exp, dim=1), dim=-1)
        s_exp = torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg, -1)], -1)
        final = traj[bflat, cflat, -1, :2]
        k = torch.cdist(final[:, None], exp).squeeze(1).argmin(-1)
        s_ego = torch.gather(s_exp, 1, k[:, None]).squeeze(1)
        progress = torch.where(s_exp[:, -1] > 1.0, s_ego / s_exp[:, -1], torch.ones_like(s_ego))
        # STL robustness: always (dist > d_min) and (v < v_max) and (lane deviation < max)
        stl = cfg.stl
        v_peak = self._max_speed(traj, bflat, cflat)
        rho = torch.minimum(torch.minimum(self.min_dist - stl.d_min, stl.v_max - v_peak),
                            stl.lane_dev_max - self.max_lane_dev)
        cov = self._closed_loop_coverage(traj, valid, conf_table) if conf_table else {}
        steps = max(self.steps, 1)
        out = []
        okf = cok.reshape(-1)
        for i in range(n):
            if not bool(okf[i]):
                continue
            rec = dict(
                method=method_name, seed=seed, scene=int(i // C), vehicle=int(i % C),
                collision=bool(self.collided[i]), at_fault_collision=bool(self.at_fault[i]),
                collision_time=float(self.coll_time[i]), collision_speed=float(self.coll_speed[i]),
                min_distance=float(self.min_dist[i]), min_ttc=float(self.min_ttc[i]),
                offroad_fraction=float(self.offroad_steps[i] / steps), max_lane_dev=float(self.max_lane_dev[i]),
                progress=float(progress[i]), distance_m=float(self.dist_travel[i]),
                mean_speed=float(self.speed_sum[i] / steps), max_accel=float(self.max_acc[i]),
                min_accel=float(self.min_acc[i]), max_jerk=float(self.max_jerk[i]),
                max_lat_acc=float(self.max_lat_acc[i]), comfort_violation_rate=float(self.comfort_viol[i] / steps),
                intervention_rate=float(self.interv[i] / steps), mpc_rate=float(self.mpc[i] / steps),
                fallback_rate=float(self.fallback[i] / steps), emergency_rate=float(self.emergency[i] / steps),
                intervention_mag_a=float(self.da_sum[i] / steps), intervention_mag_kappa=float(self.dk_sum[i] / steps),
                certified_fraction=float(self.certified[i] / steps), min_safety_margin=float(self.min_margin[i]),
                deadlock=bool(self.deadlock[i]), stl_robustness=float(rho[i]),
                src=int(sc["src"][i // C]),
            )
            if self.cert_width:
                w = torch.stack(self.cert_width, 0)[:, i]
                rec["cert_width_accel"] = float(w[:, 0].mean())
                rec["cert_width_kappa"] = float(w[:, 1].mean())
            rec.update({f"cl_coverage_{k}": v for k, v in cov.items()})
            out.append(rec)
        return out

    def _max_speed(self, traj, bflat, cflat):
        K = self.cfg.data.hist_steps
        v = torch.hypot(traj[bflat, cflat, K:, 3], traj[bflat, cflat, K:, 4])
        return v.amax(-1)

    def _closed_loop_coverage(self, traj, valid, conf_table) -> Dict[str, float]:
        """Fraction of agents whose realised closed-loop path stayed inside the conformal discs."""
        B, C = self.B, self.C
        dev = traj.device
        T = traj.shape[2]
        hits, total = 0, 0
        bflat = torch.arange(B, device=dev).repeat_interleave(C)
        for (t, aidx, pred, rad, ok, ref) in self.preds:
            if rad is None:
                continue
            H = pred.shape[2]
            h = min(H, T - 1 - t)
            if h <= 0:
                continue
            fut = traj[bflat[:, None], aidx, t + 1:t + 1 + h, :2]          # (n, N, h, 2) scene frame
            fv = valid[bflat[:, None], aidx, t + 1:t + 1 + h] & ok[..., None]
            ox, oy, oyaw = ref[:, 0], ref[:, 1], ref[:, 2]
            c, s = torch.cos(oyaw)[:, None, None], torch.sin(oyaw)[:, None, None]
            dx = fut[..., 0] - ox[:, None, None]
            dy = fut[..., 1] - oy[:, None, None]
            loc = torch.stack([dx * c + dy * s, -dx * s + dy * c], -1)
            err = torch.linalg.norm(loc - pred[:, :, :h], dim=-1)
            inside = (err <= rad[:, :, :h]) | ~fv
            agent_has = fv.any(-1)
            hits += int((inside.all(-1) & agent_has).sum())
            total += int(agent_has.sum())
        return {"overall": hits / total} if total else {}
