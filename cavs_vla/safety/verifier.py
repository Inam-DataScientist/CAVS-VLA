"""Neural output certificate: physical perception box X0 -> certified control set U_cert.

X0 (per scene, in physical units, applied to every valid history step):
    agent position +- eps_agent_pos (m), velocity +- eps_agent_vel (m/s),
    heading +- eps_agent_heading_deg (cos/sin bounded exactly),
    ego velocity +- eps_ego_speed (m/s), map point positions +- eps_map_pos (m).
Validity masks, sizes, types, traffic-light states and instruction tokens are exact.

``certify`` returns U_cert = hull of the expected controls over every mode that
can win the argmax for some input in X0 (VLAPolicy.bounds + certified_control_set).
``export_head_onnx`` writes the action head and a VNNLIB property so the same
claim can be re-checked independently with alpha-beta-CROWN.
"""
from __future__ import annotations

import copy
import math
import os
from typing import Dict, Tuple

import torch

from ..config import Config
from ..features import heading_interval_cos_sin
from ..model.vla import VLAPolicy, certified_control_set


def input_box(feat: Dict[str, torch.Tensor], cfg: Config) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    c, f = cfg.cert, cfg.feat
    lo, hi = {}, {}
    ego = feat["ego"].float()
    ev = feat["ego_valid"].float()[..., None]
    e_rad = torch.zeros_like(ego)
    e_rad[..., 4:6] = c.eps_ego_speed / f.vel_scale
    e_rad = e_rad * ev
    lo["ego"], hi["ego"] = ego - e_rad, ego + e_rad

    ag = feat["agents"].float()
    av = feat["agents_valid"].float()[..., None]
    a_rad = torch.zeros_like(ag)
    a_rad[..., 0:2] = c.eps_agent_pos / f.pos_scale
    a_rad[..., 4:6] = c.eps_agent_vel / f.vel_scale
    a_rad = a_rad * av
    a_lo, a_hi = ag - a_rad, ag + a_rad
    psi = torch.atan2(ag[..., 3], ag[..., 2])
    c_lo, c_hi, s_lo, s_hi = heading_interval_cos_sin(psi, math.radians(c.eps_agent_heading_deg))
    valid = av[..., 0] > 0
    a_lo[..., 2] = torch.where(valid, c_lo, ag[..., 2])
    a_hi[..., 2] = torch.where(valid, c_hi, ag[..., 2])
    a_lo[..., 3] = torch.where(valid, s_lo, ag[..., 3])
    a_hi[..., 3] = torch.where(valid, s_hi, ag[..., 3])
    lo["agents"], hi["agents"] = a_lo, a_hi

    mp = feat["map"].float()
    mv = feat["map_valid"].float()[..., None]
    m_rad = torch.zeros_like(mp)
    m_rad[..., 0:2] = c.eps_map_pos / f.pos_scale
    m_rad = m_rad * mv
    lo["map"], hi["map"] = mp - m_rad, mp + m_rad
    lo["map_attr"] = hi["map_attr"] = feat["map_attr"].float()
    return lo, hi


@torch.no_grad()
def certify(model: VLAPolicy, feat: Dict[str, torch.Tensor], cfg: Config,
            out: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    lo, hi = input_box(feat, cfg)
    with torch.autocast(device_type=feat["ego"].device.type, enabled=False):
        bnd = model.bounds({k: (v.float() if torch.is_floating_point(v) else v) for k, v in feat.items()}, lo, hi)
    u_lo, u_hi, feasible = certified_control_set(bnd)
    res = dict(u_lo=u_lo, u_hi=u_hi, feasible_modes=feasible, mode_lo=bnd["mode_lo"], mode_hi=bnd["mode_hi"],
               ctrl_lo=bnd["ctrl_lo"], ctrl_hi=bnd["ctrl_hi"])
    if out is not None:
        # Soundness self-check at the nominal input: the executed control must lie inside its own bounds.
        tol = 1e-3
        ctrl = out["ctrl"].float()
        inside = ((ctrl >= bnd["ctrl_lo"] - tol) & (ctrl <= bnd["ctrl_hi"] + tol)).all(-1).all(-1).all(-1)
        res["nominal_inside"] = inside
    res["width"] = (u_hi - u_lo)
    return res


class ActionHead(torch.nn.Module):
    """Mode embedding q (1, d) -> expected controls (1, Ha*2); exported for alpha-beta-CROWN."""

    def __init__(self, model: VLAPolicy) -> None:
        super().__init__()
        self.head = copy.deepcopy(model.act_head)   # a copy, so moving it to CPU leaves the policy untouched
        self.Ha, self.nA, self.nK = model.Ha, model.nA, model.nK
        self.register_buffer("acc_c", model.acc_c.clone())
        self.register_buffer("kap_c", model.kap_c.clone())

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        logits = self.head(q).view(q.shape[0], self.Ha, self.nA + self.nK)
        a = (torch.softmax(logits[..., : self.nA], -1) * self.acc_c).sum(-1)
        k = (torch.softmax(logits[..., self.nA:], -1) * self.kap_c).sum(-1)
        return torch.stack([a, k], -1).reshape(q.shape[0], -1)


@torch.no_grad()
def export_head_onnx(model: VLAPolicy, feat: Dict[str, torch.Tensor], cfg: Config, out_dir: str,
                     scene_index: int = 0, mode: int = -1, claim_steps: int = 3) -> Dict[str, str]:
    """Write head.onnx + property.vnnlib for one scene/mode.

    The input box is the IBP box of that mode's decoder embedding over X0. The property states that the
    first ``claim_steps`` controls stay inside the IBP control bounds; alpha-beta-CROWN proving it
    (UNSAT of the negation) independently confirms that part of the certificate, and a tighter
    bound can be searched by bisection on the thresholds.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    lo, hi = input_box(feat, cfg)
    sub = {k: v[scene_index:scene_index + 1] for k, v in feat.items() if torch.is_tensor(v) and v.shape[:1] == feat["ego"].shape[:1]}
    lo1 = {k: v[scene_index:scene_index + 1] for k, v in lo.items()}
    hi1 = {k: v[scene_index:scene_index + 1] for k, v in hi.items()}
    q_lo, q_hi = _decoder_box(model, sub, lo1, hi1)
    out = model(sub)
    m = int(out["mode_logits"].argmax(-1)[0]) if mode < 0 else mode
    ql, qh = q_lo[0, m:m + 1], q_hi[0, m:m + 1]
    head = ActionHead(model).cpu().eval()
    onnx_path = os.path.join(out_dir, "action_head.onnx")
    torch.onnx.export(head, ((ql + qh) / 2).cpu(), onnx_path, input_names=["q"], output_names=["u"],
                      opset_version=17, dynamic_axes=None)
    # Bounds claimed for the outputs, from our own IBP on the head.
    from ..model.layers import expectation_bounds
    l_lo, l_hi = model.act_head.ibp(ql, qh)
    l_lo = l_lo.view(1, model.Ha, model.nA + model.nK)
    l_hi = l_hi.view(1, model.Ha, model.nA + model.nK)
    a_lo, a_hi = expectation_bounds(l_lo[..., : model.nA], l_hi[..., : model.nA], model.acc_c)
    k_lo, k_hi = expectation_bounds(l_lo[..., model.nA:], l_hi[..., model.nA:], model.kap_c)
    y_lo = torch.stack([a_lo, k_lo], -1).reshape(-1)
    y_hi = torch.stack([a_hi, k_hi], -1).reshape(-1)
    vnn = os.path.join(out_dir, "property.vnnlib")
    d = ql.shape[-1]
    n_out = model.Ha * 2
    with open(vnn, "w") as f:
        f.write(f"; CAVS-VLA action-head certificate, scene {scene_index}, mode {m}\n")
        for i in range(d):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(n_out):
            f.write(f"(declare-const Y_{j} Real)\n")
        for i in range(d):
            f.write(f"(assert (>= X_{i} {float(ql[0, i]):.8f}))\n(assert (<= X_{i} {float(qh[0, i]):.8f}))\n")
        terms = []
        for j in range(min(n_out, 2 * claim_steps)):
            terms.append(f"(and (<= Y_{j} {float(y_lo[j]) - 1e-6:.8f}))")
            terms.append(f"(and (>= Y_{j} {float(y_hi[j]) + 1e-6:.8f}))")
        f.write("(assert (or " + " ".join(terms) + "))\n")
    return dict(onnx=onnx_path, vnnlib=vnn, mode=str(m))


@torch.no_grad()
def _decoder_box(model: VLAPolicy, feat, lo, hi):
    """IBP box of the decoder output (mode embeddings), reusing the model's bound pass step by step."""
    B = feat["ego"].shape[0]
    av = feat["agents_valid"].float()
    mv = feat["map_valid"].float()
    el, eh = model.ego_enc.ibp(lo["ego"].reshape(B, -1), hi["ego"].reshape(B, -1))
    al, ah = model.agent_enc.ibp(torch.cat([lo["agents"].reshape(B, model.N, -1), av], -1),
                                 torch.cat([hi["agents"].reshape(B, model.N, -1), av], -1))
    ml, mh = model.map_enc.ibp(torch.cat([lo["map"].reshape(B, model.P, -1), mv, lo["map_attr"]], -1),
                               torch.cat([hi["map"].reshape(B, model.P, -1), mv, hi["map_attr"]], -1))
    te = model._text(feat["instr"])
    from ..model.vla import ENTITY_AGENT, ENTITY_EGO, ENTITY_MAP
    x_lo = torch.cat([el[:, None] + model.type_emb[ENTITY_EGO], al + model.type_emb[ENTITY_AGENT],
                      ml + model.type_emb[ENTITY_MAP], te], 1)
    x_hi = torch.cat([eh[:, None] + model.type_emb[ENTITY_EGO], ah + model.type_emb[ENTITY_AGENT],
                      mh + model.type_emb[ENTITY_MAP], te], 1)
    valid = model._masks(feat)
    for blk in model.encoder:
        x_lo, x_hi = blk.ibp(x_lo, x_hi, valid)
    q_lo = model.mode_query[None].expand(B, -1, -1) + x_lo[:, :1]
    q_hi = model.mode_query[None].expand(B, -1, -1) + x_hi[:, :1]
    for blk in model.decoder:
        q_lo, q_hi = blk.ibp(q_lo, q_hi, x_lo, x_hi, valid)
    return q_lo, q_hi
