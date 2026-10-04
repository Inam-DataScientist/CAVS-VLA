"""Verification-friendly building blocks: every module has ``forward`` and a sound ``ibp``.

Interval bound propagation (IBP) maps an input box [lo, hi] to an output box
that contains f(x) for every x in the input box. The modules are written with
explicit tensor ops (no fused attention kernels) so that the forward pass and
its bound pass are mirror images and can be tested against each other
(selftest: zero-width boxes reproduce forward exactly; random samples inside a
box always land inside the propagated bounds).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Interval = Tuple[torch.Tensor, torch.Tensor]
NEG = -1e9
FLOAT_PAD = 1e-4   # outward padding of certified bounds to absorb float32 rounding


class precise_matmul:
    """Context manager: disable TF32 so bound computations use full float32 matmuls."""

    def __enter__(self):
        self.prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.prev
        return False


def _mid_rad(lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (lo + hi) * 0.5, (hi - lo).clamp(min=0) * 0.5


def interval_matmul(a_lo, a_hi, b_lo, b_hi) -> Interval:
    """Sound bounds of A @ B for interval matrices (mid-radius arithmetic)."""
    am, ar = _mid_rad(a_lo, a_hi)
    bm, br = _mid_rad(b_lo, b_hi)
    mid = am @ bm
    rad = am.abs() @ br + ar @ bm.abs() + ar @ br
    return mid - rad, mid + rad


def softmax_bounds(lo: torch.Tensor, hi: torch.Tensor) -> Interval:
    """Elementwise bounds of softmax over the last dim for logits in [lo, hi]."""
    m = hi.amax(-1, keepdim=True)
    el = torch.exp(lo - m)
    eu = torch.exp(hi - m)
    sl = el.sum(-1, keepdim=True)
    su = eu.sum(-1, keepdim=True)
    p_lo = el / (el + (su - eu).clamp(min=0) + 1e-30)
    p_hi = eu / (eu + (sl - el).clamp(min=0) + 1e-30)
    return p_lo.clamp(0.0, 1.0), p_hi.clamp(0.0, 1.0)


def expectation_bounds(logit_lo: torch.Tensor, logit_hi: torch.Tensor, centers: torch.Tensor) -> Interval:
    """Exact bounds of sum_b p_b c_b over {p : p_lo <= p <= p_hi, sum p = 1}, centers ascending.

    This is the linear program min/max c^T p over a box intersected with the
    simplex, solved greedily (fill the cheapest / most expensive bins first).
    """
    p_lo, p_hi = softmax_bounds(logit_lo, logit_hi)
    rem = (1.0 - p_lo.sum(-1, keepdim=True)).clamp(min=0)
    cap = (p_hi - p_lo).clamp(min=0)
    before = torch.cumsum(cap, -1) - cap
    add_min = torch.minimum((rem - before).clamp(min=0), cap)
    e_lo = ((p_lo + add_min) * centers).sum(-1)
    cap_r = torch.flip(cap, [-1])
    before_r = torch.cumsum(cap_r, -1) - cap_r
    add_max = torch.flip(torch.minimum((rem - before_r).clamp(min=0), cap_r), [-1])
    e_hi = ((p_lo + add_max) * centers).sum(-1)
    return torch.minimum(e_lo, e_hi), torch.maximum(e_lo, e_hi)


class ILinear(nn.Linear):
    def ibp(self, lo: torch.Tensor, hi: torch.Tensor) -> Interval:
        mid, rad = _mid_rad(lo, hi)
        out_mid = F.linear(mid, self.weight, self.bias)
        out_rad = F.linear(rad, self.weight.abs())
        return out_mid - out_rad, out_mid + out_rad


class IMLP(nn.Module):
    """Linear -> ReLU -> ... -> Linear (no activation after the last layer)."""

    def __init__(self, dims, final_scale: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ILinear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        if final_scale != 1.0:
            with torch.no_grad():
                self.layers[-1].weight.mul_(final_scale)
                self.layers[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def ibp(self, lo: torch.Tensor, hi: torch.Tensor) -> Interval:
        for i, layer in enumerate(self.layers):
            lo, hi = layer.ibp(lo, hi)
            if i < len(self.layers) - 1:
                lo, hi = F.relu(lo), F.relu(hi)
        return lo, hi


class ILayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xc = xf - xf.mean(-1, keepdim=True)
        var = (xc * xc).mean(-1, keepdim=True)
        out = xc * torch.rsqrt(var + self.eps) * self.weight.float() + self.bias.float()
        return out.to(x.dtype)

    def ibp(self, lo: torch.Tensor, hi: torch.Tensor) -> Interval:
        d = lo.shape[-1]
        mid, rad = _mid_rad(lo, hi)
        c_mid = mid - mid.mean(-1, keepdim=True)
        c_rad = rad * (1.0 - 2.0 / d) + rad.sum(-1, keepdim=True) / d
        c_lo, c_hi = c_mid - c_rad, c_mid + c_rad
        sq_hi = torch.maximum(c_lo * c_lo, c_hi * c_hi)
        straddle = (c_lo <= 0) & (c_hi >= 0)
        sq_lo = torch.where(straddle, torch.zeros_like(sq_hi), torch.minimum(c_lo * c_lo, c_hi * c_hi))
        inv_hi = torch.rsqrt(sq_lo.mean(-1, keepdim=True) + self.eps)
        inv_lo = torch.rsqrt(sq_hi.mean(-1, keepdim=True) + self.eps)
        cands = torch.stack([c_lo * inv_lo, c_lo * inv_hi, c_hi * inv_lo, c_hi * inv_hi], 0)
        y_lo, y_hi = cands.amin(0), cands.amax(0)
        ym, yr = _mid_rad(y_lo, y_hi)
        om = ym * self.weight + self.bias
        orad = yr * self.weight.abs()
        return om - orad, om + orad


class IAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.q = ILinear(d, d)
        self.k = ILinear(d, d)
        self.v = ILinear(d, d)
        self.o = ILinear(d, d)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, h, L, dh = x.shape
        return x.transpose(1, 2).reshape(B, L, h * dh)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(self.q(xq)), self._split(self.k(xkv)), self._split(self.v(xkv))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        scores = scores.masked_fill(~key_valid[:, None, None, :], NEG)
        attn = torch.softmax(scores, dim=-1)
        return self.o(self._merge(attn @ v))

    def ibp(self, q_lo, q_hi, kv_lo, kv_hi, key_valid: torch.Tensor) -> Interval:
        ql, qh = self.q.ibp(q_lo, q_hi)
        kl, kh = self.k.ibp(kv_lo, kv_hi)
        vl, vh = self.v.ibp(kv_lo, kv_hi)
        ql, qh, kl, kh, vl, vh = (self._split(t) for t in (ql, qh, kl, kh, vl, vh))
        s_lo, s_hi = interval_matmul(ql, qh, kl.transpose(-1, -2), kh.transpose(-1, -2))
        scale = 1.0 / math.sqrt(self.dh)
        s_lo, s_hi = s_lo * scale, s_hi * scale
        mask = ~key_valid[:, None, None, :]
        s_lo = s_lo.masked_fill(mask, NEG)
        s_hi = s_hi.masked_fill(mask, NEG)
        p_lo, p_hi = softmax_bounds(s_lo, s_hi)
        o_lo, o_hi = interval_matmul(p_lo, p_hi, vl, vh)
        return self.o.ibp(self._merge(o_lo), self._merge(o_hi))


class IFeedForward(nn.Module):
    def __init__(self, d: int, mult: int) -> None:
        super().__init__()
        self.mlp = IMLP([d, d * mult, d])

    def forward(self, x):
        return self.mlp(x)

    def ibp(self, lo, hi):
        return self.mlp.ibp(lo, hi)


class IEncoderBlock(nn.Module):
    """Pre-LN self-attention block."""

    def __init__(self, d: int, n_heads: int, mult: int, eps: float) -> None:
        super().__init__()
        self.ln1 = ILayerNorm(d, eps)
        self.attn = IAttention(d, n_heads)
        self.ln2 = ILayerNorm(d, eps)
        self.ff = IFeedForward(d, mult)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, valid)
        return x + self.ff(self.ln2(x))

    def ibp(self, lo: torch.Tensor, hi: torch.Tensor, valid: torch.Tensor) -> Interval:
        hl, hh = self.ln1.ibp(lo, hi)
        al, ah = self.attn.ibp(hl, hh, hl, hh, valid)
        lo, hi = lo + al, hi + ah
        fl, fh = self.ln2.ibp(lo, hi)
        fl, fh = self.ff.ibp(fl, fh)
        return lo + fl, hi + fh


class IDecoderBlock(nn.Module):
    """Mode queries: cross-attention to the scene, self-attention among modes, feed-forward."""

    def __init__(self, d: int, n_heads: int, mult: int, eps: float) -> None:
        super().__init__()
        self.ln_q = ILayerNorm(d, eps)
        self.ln_m = ILayerNorm(d, eps)
        self.cross = IAttention(d, n_heads)
        self.ln_s = ILayerNorm(d, eps)
        self.self_attn = IAttention(d, n_heads)
        self.ln_f = ILayerNorm(d, eps)
        self.ff = IFeedForward(d, mult)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, mem_valid: torch.Tensor) -> torch.Tensor:
        q = q + self.cross(self.ln_q(q), self.ln_m(mem), mem_valid)
        h = self.ln_s(q)
        all_valid = torch.ones(q.shape[0], q.shape[1], dtype=torch.bool, device=q.device)
        q = q + self.self_attn(h, h, all_valid)
        return q + self.ff(self.ln_f(q))

    def ibp(self, q_lo, q_hi, m_lo, m_hi, mem_valid) -> Interval:
        a, b = self.ln_q.ibp(q_lo, q_hi)
        c, d = self.ln_m.ibp(m_lo, m_hi)
        xl, xh = self.cross.ibp(a, b, c, d, mem_valid)
        q_lo, q_hi = q_lo + xl, q_hi + xh
        a, b = self.ln_s.ibp(q_lo, q_hi)
        all_valid = torch.ones(q_lo.shape[0], q_lo.shape[1], dtype=torch.bool, device=q_lo.device)
        xl, xh = self.self_attn.ibp(a, b, a, b, all_valid)
        q_lo, q_hi = q_lo + xl, q_hi + xh
        a, b = self.ln_f.ibp(q_lo, q_hi)
        xl, xh = self.ff.ibp(a, b)
        return q_lo + xl, q_hi + xh
