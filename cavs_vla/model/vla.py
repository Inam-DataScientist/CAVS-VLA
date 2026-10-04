"""State-language-action driving policy with multi-modal action tokens and interval bounds.

Inputs (from features.featurize): ego history, N agents, P map polylines,
instruction tokens. Tokens are encoded by per-entity MLPs, mixed by a
Transformer encoder, and decoded by M mode queries into

  * action-token logits over (acceleration, curvature) for Ha control steps,
  * the expected control under those logits (continuous, used for execution),
  * the kinematic rollout of that control (x, y, psi, v) for H steps,
  * a score per mode,
  * per-agent future positions with a Laplace scale (for conformal sets),
  * an intent class (interpretable "reasoning" output; never trusted for safety).

``bounds`` mirrors ``forward`` with interval arithmetic and returns sound
bounds on the mode scores and on the expected controls of every mode.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..dynamics import rollout
from ..features import AGENT_FEAT, EGO_FEAT, MAP_ATTR_FEAT, MAP_PT_FEAT, acc_centers, kappa_centers, steps_per_action
from ..language import PAD, VOCAB_SIZE
from .layers import FLOAT_PAD, IDecoderBlock, IEncoderBlock, IMLP, expectation_bounds, precise_matmul

ENTITY_EGO, ENTITY_AGENT, ENTITY_MAP, ENTITY_TEXT = range(4)


class VLAPolicy(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        m, d_cfg, f = cfg.model, cfg.data, cfg.feat
        d = m.d_model
        K, H = d_cfg.hist_steps, d_cfg.fut_steps
        self.K, self.H = K, H
        self.N, self.P, self.Q, self.L = f.num_agents, f.num_polylines, d_cfg.poly_points, f.instr_len
        self.M = m.n_modes
        self.Ha = H // steps_per_action(cfg)
        self.nA, self.nK = f.n_acc_bins, f.n_kappa_bins
        self.ego_enc = IMLP([K * EGO_FEAT, d, d])
        self.agent_enc = IMLP([K * AGENT_FEAT + K, d, d])
        self.map_enc = IMLP([self.Q * MAP_PT_FEAT + self.Q + MAP_ATTR_FEAT, d, d])
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.tok_pos = nn.Parameter(torch.zeros(self.L, d))
        self.type_emb = nn.Parameter(torch.zeros(4, d))
        self.encoder = nn.ModuleList([IEncoderBlock(d, m.n_heads, m.ff_mult, m.ln_eps) for _ in range(m.enc_layers)])
        self.mode_query = nn.Parameter(torch.randn(self.M, d) * 0.02)
        self.decoder = nn.ModuleList([IDecoderBlock(d, m.n_heads, m.ff_mult, m.ln_eps)
                                      for _ in range(m.dec_layers)])
        self.score_head = IMLP([d, d, 1])
        self.act_head = IMLP([d, 2 * d, self.Ha * (self.nA + self.nK)], final_scale=0.1)
        self.agent_head = IMLP([d, d, H * 4], final_scale=0.1)
        self.intent_head = IMLP([d, d, m.num_intents])
        self.register_buffer("acc_c", acc_centers(cfg), persistent=False)
        self.register_buffer("kap_c", kappa_centers(cfg), persistent=False)
        nn.init.normal_(self.tok_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        with torch.no_grad():
            # Bias the acceleration head toward "keep speed" and curvature toward straight.
            last = self.act_head.layers[-1].bias.view(self.Ha, self.nA + self.nK)
            last[:, : self.nA] = -0.05 * (self.acc_c - 0.0).abs()
            last[:, self.nA:] = -2.0 * (self.kap_c / cfg.feat.kappa_max).abs()

    # ----------------------------------------------------------------- inputs
    def _inputs(self, feat: Dict[str, torch.Tensor]):
        B = feat["ego"].shape[0]
        ego_in = feat["ego"].reshape(B, -1)
        agent_in = torch.cat([feat["agents"].reshape(B, self.N, -1), feat["agents_valid"].float()], -1)
        map_in = torch.cat([feat["map"].reshape(B, self.P, -1), feat["map_valid"].float(), feat["map_attr"]], -1)
        return ego_in, agent_in, map_in

    def _masks(self, feat):
        B = feat["ego"].shape[0]
        dev = feat["ego"].device
        ego_ok = torch.ones(B, 1, dtype=torch.bool, device=dev)
        agent_ok = feat["agents_valid"].any(-1)
        map_ok = feat["map_valid"].any(-1)
        text_ok = feat["instr"] != PAD
        return torch.cat([ego_ok, agent_ok, map_ok, text_ok], 1)

    def _text(self, instr: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(instr) + self.tok_pos[None] + self.type_emb[ENTITY_TEXT]

    # ----------------------------------------------------------------- forward
    def encode(self, feat: Dict[str, torch.Tensor]):
        ego_in, agent_in, map_in = self._inputs(feat)
        e = self.ego_enc(ego_in)[:, None] + self.type_emb[ENTITY_EGO]
        a = self.agent_enc(agent_in) + self.type_emb[ENTITY_AGENT]
        mp = self.map_enc(map_in) + self.type_emb[ENTITY_MAP]
        tx = self._text(feat["instr"])
        x = torch.cat([e, a, mp, tx], 1)
        valid = self._masks(feat)
        for blk in self.encoder:
            x = blk(x, valid)
        return x, valid

    def forward(self, feat: Dict[str, torch.Tensor], rollout_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        B = feat["ego"].shape[0]
        x, valid = self.encode(feat)
        q = self.mode_query[None].expand(B, -1, -1) + x[:, :1]
        for blk in self.decoder:
            q = blk(q, x, valid)
        mode_logits = self.score_head(q).squeeze(-1)
        logits = self.act_head(q).view(B, self.M, self.Ha, self.nA + self.nK)
        acc_logits, kap_logits = logits[..., : self.nA], logits[..., self.nA:]
        acc = (torch.softmax(acc_logits.float(), -1) * self.acc_c).sum(-1)
        kap = (torch.softmax(kap_logits.float(), -1) * self.kap_c).sum(-1)
        ctrl = torch.stack([acc, kap], -1)
        v0 = feat["ego_speed"].float()[:, None].expand(B, self.M)
        traj = rollout(v0, ctrl, self.cfg, n_steps=rollout_steps)
        agent_tok = x[:, 1:1 + self.N]
        ag = self.agent_head(agent_tok).view(B, self.N, self.H, 4).float()
        cur = feat["agents"][:, :, -1, :2].float() * self.cfg.feat.pos_scale
        agent_pred = cur[:, :, None, :] + ag[..., :2] * self.cfg.feat.pos_scale
        agent_logscale = ag[..., 2:].clamp(-3.0, 4.0)
        intent_logits = self.intent_head(x[:, 0])
        tokens = torch.stack([acc_logits.argmax(-1), kap_logits.argmax(-1)], -1)
        return dict(mode_logits=mode_logits, acc_logits=acc_logits, kap_logits=kap_logits, ctrl=ctrl, traj=traj,
                    agent_pred=agent_pred, agent_logscale=agent_logscale, intent_logits=intent_logits,
                    tokens=tokens)

    # ----------------------------------------------------------------- interval bounds
    @torch.no_grad()
    def bounds(self, feat: Dict[str, torch.Tensor], lo: Dict[str, torch.Tensor],
               hi: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sound bounds for inputs in the box lo[k] <= feat[k] <= hi[k] for k in (ego, agents, map, map_attr).

        Validity masks and instruction tokens are treated as exact (they are not perceptual quantities).
        """
        with precise_matmul():
            return self._bounds(feat, lo, hi)

    def _bounds(self, feat, lo, hi):
        B = feat["ego"].shape[0]
        ego_lo, ego_hi = lo["ego"].reshape(B, -1), hi["ego"].reshape(B, -1)
        av = feat["agents_valid"].float()
        ag_lo = torch.cat([lo["agents"].reshape(B, self.N, -1), av], -1)
        ag_hi = torch.cat([hi["agents"].reshape(B, self.N, -1), av], -1)
        mv = feat["map_valid"].float()
        mp_lo = torch.cat([lo["map"].reshape(B, self.P, -1), mv, lo["map_attr"]], -1)
        mp_hi = torch.cat([hi["map"].reshape(B, self.P, -1), mv, hi["map_attr"]], -1)
        el, eh = self.ego_enc.ibp(ego_lo, ego_hi)
        al, ah = self.agent_enc.ibp(ag_lo, ag_hi)
        ml, mh = self.map_enc.ibp(mp_lo, mp_hi)
        te = self._text(feat["instr"])
        x_lo = torch.cat([el[:, None] + self.type_emb[ENTITY_EGO], al + self.type_emb[ENTITY_AGENT],
                          ml + self.type_emb[ENTITY_MAP], te], 1)
        x_hi = torch.cat([eh[:, None] + self.type_emb[ENTITY_EGO], ah + self.type_emb[ENTITY_AGENT],
                          mh + self.type_emb[ENTITY_MAP], te], 1)
        valid = self._masks(feat)
        for blk in self.encoder:
            x_lo, x_hi = blk.ibp(x_lo, x_hi, valid)
        q_lo = self.mode_query[None].expand(B, -1, -1) + x_lo[:, :1]
        q_hi = self.mode_query[None].expand(B, -1, -1) + x_hi[:, :1]
        for blk in self.decoder:
            q_lo, q_hi = blk.ibp(q_lo, q_hi, x_lo, x_hi, valid)
        s_lo, s_hi = self.score_head.ibp(q_lo, q_hi)
        l_lo, l_hi = self.act_head.ibp(q_lo, q_hi)
        l_lo = l_lo.view(B, self.M, self.Ha, self.nA + self.nK)
        l_hi = l_hi.view(B, self.M, self.Ha, self.nA + self.nK)
        a_lo, a_hi = expectation_bounds(l_lo[..., : self.nA], l_hi[..., : self.nA], self.acc_c)
        k_lo, k_hi = expectation_bounds(l_lo[..., self.nA:], l_hi[..., self.nA:], self.kap_c)
        pad = FLOAT_PAD
        return dict(mode_lo=s_lo.squeeze(-1) - pad, mode_hi=s_hi.squeeze(-1) + pad,
                    ctrl_lo=torch.stack([a_lo, k_lo], -1) - pad, ctrl_hi=torch.stack([a_hi, k_hi], -1) + pad)


def certified_control_set(bnd: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hull of the control bounds over every mode that can win the argmax inside the input box.

    Returns (u_lo, u_hi) of shape (B, Ha, 2) and the feasible-mode mask (B, M).
    """
    best_lo = bnd["mode_lo"].amax(-1, keepdim=True)
    feasible = bnd["mode_hi"] >= best_lo
    big = torch.finfo(bnd["ctrl_lo"].dtype).max
    lo = torch.where(feasible[..., None, None], bnd["ctrl_lo"], torch.full_like(bnd["ctrl_lo"], big)).amin(1)
    hi = torch.where(feasible[..., None, None], bnd["ctrl_hi"], torch.full_like(bnd["ctrl_hi"], -big)).amax(1)
    return lo, hi, feasible


# --------------------------------------------------------------------- losses


def policy_losses(out: Dict[str, torch.Tensor], feat: Dict[str, torch.Tensor], cfg: Config,
                  safety_penalty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    tc = cfg.train
    gt = feat["ego_fut"].float()
    gv = feat["ego_fut_valid"].float()
    traj = out["traj"]
    B, M, H, _ = traj.shape
    pos_err = torch.linalg.norm(traj[..., :2] - gt[:, None, :, :2], dim=-1)          # (B, M, H)
    denom = gv.sum(-1).clamp(min=1.0)
    ade = (pos_err * gv[:, None]).sum(-1) / denom[:, None]
    best = ade.argmin(1)
    bi = torch.arange(B, device=traj.device)
    tb = traj[bi, best]
    reg = F.smooth_l1_loss(tb[..., :2], gt[..., :2], reduction="none").sum(-1)
    head = 1.0 - torch.cos(tb[..., 2] - gt[..., 2])
    spd = F.smooth_l1_loss(tb[..., 3], gt[..., 3], reduction="none")
    l_reg = ((reg + 2.0 * head + 0.5 * spd) * gv).sum() / gv.sum().clamp(min=1.0)
    l_cls = F.cross_entropy(out["mode_logits"].float(), best)
    cv = feat["gt_ctrl_valid"].float()
    acc_l = out["acc_logits"][bi, best].float()
    kap_l = out["kap_logits"][bi, best].float()
    ce_a = F.cross_entropy(acc_l.reshape(-1, acc_l.shape[-1]), feat["gt_tok"][..., 0].reshape(-1), reduction="none")
    ce_k = F.cross_entropy(kap_l.reshape(-1, kap_l.shape[-1]), feat["gt_tok"][..., 1].reshape(-1), reduction="none")
    l_tok = ((ce_a + ce_k) * cv.reshape(-1)).sum() / cv.sum().clamp(min=1.0)
    av = feat["agent_fut_valid"].float()
    scale = torch.exp(out["agent_logscale"])
    nll = (torch.abs(out["agent_pred"] - feat["agent_fut"].float()) / scale + torch.log(2 * scale)).sum(-1)
    l_agent = (nll * av).sum() / av.sum().clamp(min=1.0)
    l_intent = F.cross_entropy(out["intent_logits"].float(), feat["intent"])
    loss = tc.w_reg * l_reg + tc.w_cls * l_cls + tc.w_tok * l_tok + tc.w_agent * l_agent + tc.w_intent * l_intent
    logs = dict(l_reg=float(l_reg), l_cls=float(l_cls), l_tok=float(l_tok), l_agent=float(l_agent),
                l_intent=float(l_intent))
    if safety_penalty is not None:
        loss = loss + safety_penalty
        logs["l_safe"] = float(safety_penalty)
    with torch.no_grad():
        top = out["mode_logits"].argmax(1)
        fde_all = pos_err[..., -1]
        logs.update(minADE=float(ade.min(1).values.mean()), minFDE=float(fde_all.min(1).values.mean()),
                    ADE=float(ade[bi, top].mean()), FDE=float(fde_all[bi, top].mean()))
    return loss, logs


def proximity_penalty(out: Dict[str, torch.Tensor], feat: Dict[str, torch.Tensor], cfg: Config,
                      hj=None) -> torch.Tensor:
    """Differentiable safety term for counterexample-guided refinement.

    For every mode, penalise (i) disc-overlap with the logged agent futures and
    (ii) negative HJ value w.r.t. agents in the ego's path, weighted by the mode
    probability, so the policy learns both to avoid and to rank safe modes higher.
    """
    traj = out["traj"]
    B, M, H, _ = traj.shape
    probs = torch.softmax(out["mode_logits"].float(), -1)
    ef = traj[..., :2]                                                     # (B, M, H, 2)
    af = feat["agent_fut"].float()                                         # (B, N, H, 2)
    av = feat["agent_fut_valid"].float()
    dims = feat["agents_phys"][..., 5:7]
    r_ag = 0.5 * dims.amax(-1).clamp(min=0.5)                              # (B, N)
    r_eg = 0.5 * feat["ego_dims"][:, 1].float()
    d = torch.linalg.norm(ef[:, :, None] - af[:, None], dim=-1)            # (B, M, N, H)
    need = (r_ag[:, None, :, None] + r_eg[:, None, None, None] + cfg.train.safe_margin)
    pen = (F.relu(need - d) * av[:, None]).sum((-1, -2)) / av.sum((-1, -2)).clamp(min=1.0)[:, None]
    total = (pen * probs).sum(-1).mean()
    if hj is not None:
        # Same-direction agents ahead in the ego's lane: (gap, v_ego, v_other) along the plan.
        aphys = feat["agents_phys"]
        same_dir = (torch.cos(aphys[..., 2]) > 0.8) & (aphys[..., 0] > 0) & (aphys[..., 1].abs() < 2.0)
        ahead = feat["agents_ok"] & same_dir                                # (B, N)
        if ahead.any():
            gap = af[:, None, :, :, 0] - ef[:, :, None, :, 0] - 0.5 * (dims[..., 0][:, None, :, None]
                                                                       + feat["ego_dims"][:, 0][:, None, None, None])
            v_e = traj[..., 3][:, :, None, :].expand_as(gap)
            v_o = (torch.diff(af[..., 0], dim=-1, prepend=af[..., :1, 0]) / cfg.data.dt).clamp(min=0)
            v_o = v_o[:, None].expand_as(gap)
            val = hj.value_torch(gap, v_e, v_o)
            mask = (ahead[:, None, :, None] & (av[:, None] > 0)).float()
            hpen = (F.relu(cfg.shield.hj_margin - val) * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp(min=1.0)
            total = total + (hpen * probs).sum(-1).mean()
    return total
