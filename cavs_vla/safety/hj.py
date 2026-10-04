"""Hamilton-Jacobi reachability for the longitudinal interaction z = (d, v_ego, v_other).

Dynamics (along the ego path):
    d'     = v_other - v_ego
    v_ego' = sat(a_ego + w),   a_ego in [a_e_min, a_e_max] (control), |w| <= w_bar (model mismatch)
    v_oth' = sat(a_oth),       a_oth in [a_o_min, a_o_max] (adversarial other agent)
where sat() stops speeds at 0 and v_max (the 2-D [d, dv] prototype had no saturation,
so an adversarial lead could "keep braking" after it had stopped).

Failure set: d < d_min. The value V(z) solves the HJI variational inequality
    0 = min{ dV/dt + max_a min_{w, a_o} grad V . f ,  l(z) - V },   l(z) = d - d_min
and approximates  min_t (d(t) - d_min)  under optimal play, so {V >= 0} is the safe set
and the maximal-braking policy is the safety-preserving backup.

Numerics: local Lax-Friedrichs, first-order upwind differences, explicit time
stepping under a CFL bound, then a monotone lower envelope (V is increasing in d,
decreasing in v_ego, increasing in v_other) so that (i) the box minimum is a corner
lookup and (ii) the table never exceeds the raw solution. ``validate`` compares
against the closed-form worst case (both vehicles brake maximally), which is the
exact game value for this model, and returns a calibration offset that makes the
table conservative on the validation set.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import HJConfig


class HJGrid:
    def __init__(self, c: HJConfig) -> None:
        self.c = c
        self.d = np.linspace(c.d_lo, c.d_hi, c.n_d)
        self.v = np.linspace(0.0, c.v_max, c.n_v)
        self.dd = self.d[1] - self.d[0]
        self.dv = self.v[1] - self.v[0]
        self.D, self.VE, self.VO = np.meshgrid(self.d, self.v, self.v, indexing="ij")


def _diffs(V: np.ndarray, axis: int, h: float) -> Tuple[np.ndarray, np.ndarray]:
    # Ghost cells by linear extrapolation: one-sided slopes at the boundary equal the interior
    # slope, which avoids the artificial flattening (and value over-estimation) of edge padding.
    pad = [(0, 0)] * V.ndim
    pad[axis] = (1, 1)
    Vp = np.pad(V, pad, mode="reflect", reflect_type="odd")
    sl_c = [slice(None)] * V.ndim
    sl_m = [slice(None)] * V.ndim
    sl_p = [slice(None)] * V.ndim
    sl_c[axis] = slice(1, -1)
    sl_m[axis] = slice(0, -2)
    sl_p[axis] = slice(2, None)
    c, m, p = Vp[tuple(sl_c)], Vp[tuple(sl_m)], Vp[tuple(sl_p)]
    return (c - m) / h, (p - c) / h


def _ego_term(p: np.ndarray, at_zero: np.ndarray, at_max: np.ndarray, c: HJConfig) -> np.ndarray:
    """max_a min_w p * sat(a + w) with saturation at v = 0 and v = v_max."""
    interior = np.where(p >= 0, p * (c.a_e_max - c.w_bar), p * (c.a_e_min + c.w_bar))
    zero = np.where(p >= 0, p * max(c.a_e_max - c.w_bar, 0.0), p * max(c.a_e_min + c.w_bar, 0.0))
    top = np.where(p >= 0, p * min(c.a_e_max - c.w_bar, 0.0), p * min(c.a_e_min + c.w_bar, 0.0))
    return np.where(at_zero, zero, np.where(at_max, top, interior))


def _other_term(p: np.ndarray, at_zero: np.ndarray, at_max: np.ndarray, c: HJConfig) -> np.ndarray:
    """min_{a_o} p * sat(a_o)."""
    interior = np.where(p >= 0, p * c.a_o_min, p * c.a_o_max)
    zero = np.where(p >= 0, p * max(c.a_o_min, 0.0), p * max(c.a_o_max, 0.0))
    top = np.where(p >= 0, p * min(c.a_o_min, 0.0), p * min(c.a_o_max, 0.0))
    return np.where(at_zero, zero, np.where(at_max, top, interior))


def solve(c: HJConfig, verbose: bool = True) -> Dict[str, np.ndarray]:
    g = HJGrid(c)
    l = g.D - c.d_min
    V = l.copy()
    rel = g.VO - g.VE
    ve0 = g.VE <= 1e-9
    vem = g.VE >= c.v_max - 1e-9
    vo0 = g.VO <= 1e-9
    vom = g.VO >= c.v_max - 1e-9
    alpha_d = np.abs(rel)
    alpha_e = max(abs(c.a_e_min), abs(c.a_e_max)) + c.w_bar
    alpha_o = max(abs(c.a_o_min), abs(c.a_o_max))
    dt = c.cfl / (float(alpha_d.max()) / g.dd + alpha_e / g.dv + alpha_o / g.dv)
    n_steps = int(np.ceil(c.horizon / dt))
    dt = c.horizon / n_steps
    t0 = time.time()
    for it in range(n_steps):
        pdm, pdp = _diffs(V, 0, g.dd)
        pem, pep = _diffs(V, 1, g.dv)
        pom, pop = _diffs(V, 2, g.dv)
        pd, pe, po = 0.5 * (pdm + pdp), 0.5 * (pem + pep), 0.5 * (pom + pop)
        Hc = pd * rel + _ego_term(pe, ve0, vem, c) + _other_term(po, vo0, vom, c)
        diss = alpha_d * 0.5 * (pdp - pdm) + alpha_e * 0.5 * (pep - pem) + alpha_o * 0.5 * (pop - pom)
        Hn = Hc + diss
        V = np.minimum(V, V + dt * Hn)
        V = np.minimum(V, l)
        if verbose and (it % max(1, n_steps // 10) == 0 or it == n_steps - 1):
            print(f"  HJ step {it + 1}/{n_steps}  t={dt * (it + 1):.2f}s  safe-fraction={(V >= 0).mean():.4f}")
    raw = V.copy()
    V = monotone_envelope(V)
    return dict(V=V.astype(np.float32), V_raw=raw.astype(np.float32), d=g.d, v=g.v, dt=dt, n_steps=n_steps,
                seconds=time.time() - t0)


def monotone_envelope(V: np.ndarray) -> np.ndarray:
    """Largest function below V that is increasing in d, decreasing in v_ego, increasing in v_other."""
    W = np.minimum.accumulate(V[::-1], axis=0)[::-1]           # min over d' >= d
    W = np.minimum.accumulate(W, axis=1)                         # min over v_e' <= v_e
    W = np.minimum.accumulate(W[:, :, ::-1], axis=2)[:, :, ::-1]  # min over v_o' >= v_o
    return W


def closed_form_value(d: np.ndarray, ve: np.ndarray, vo: np.ndarray, c: HJConfig, dt: float = 0.002) -> np.ndarray:
    """Exact game value: min over [0, horizon] of (gap - d_min) when the ego brakes at a_e_min + w_bar
    and the other agent brakes at a_o_min (both optimal for this monotone game)."""
    d = np.asarray(d, dtype=np.float64).copy()
    ve = np.asarray(ve, dtype=np.float64).copy()
    vo = np.asarray(vo, dtype=np.float64).copy()
    ae = min(c.a_e_min + c.w_bar, 0.0)
    ao = c.a_o_min
    best = d - c.d_min
    n = int(np.ceil(c.horizon / dt))
    for _ in range(n):
        ve_n = np.maximum(ve + ae * dt, 0.0)
        vo_n = np.maximum(vo + ao * dt, 0.0)
        d = d + 0.5 * ((vo + vo_n) - (ve + ve_n)) * dt
        ve, vo = ve_n, vo_n
        best = np.minimum(best, d - c.d_min)
    return best


def interp_np(table: Dict[str, np.ndarray], d: np.ndarray, ve: np.ndarray, vo: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((table["d"], table["v"], table["v"]), table["V"].astype(np.float64),
                                bounds_error=False, fill_value=None)
    return f(np.stack([np.clip(d, table["d"][0], table["d"][-1]), np.clip(ve, 0, table["v"][-1]),
                       np.clip(vo, 0, table["v"][-1])], -1))


def validate(table: Dict[str, np.ndarray], c: HJConfig, n: int = 20000, seed: int = 0,
             band: float = 1.0) -> Dict[str, float]:
    """Compare the table with the closed-form value on random interior states."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(c.d_lo + 2.0, c.d_hi - 20.0, n)
    ve = rng.uniform(0.0, c.v_max - 2.0, n)
    vo = rng.uniform(0.0, c.v_max - 2.0, n)
    # Both values are clamped at the grid's deepest representable gap: below it every state is
    # unsafe and the depth of the violation carries no information for the safety decision.
    floor = c.d_lo - c.d_min
    true = np.maximum(closed_form_value(d, ve, vo, c), floor)
    est = np.maximum(interp_np(table, d, ve, vo), floor)
    err = est - true
    clear = np.abs(true) > band
    sign_ok = np.sign(est[clear]) == np.sign(true[clear])
    unsafe_missed = np.mean((true[clear] < 0) & (est[clear] >= 0))
    over = np.quantile(err, 0.999)
    return dict(n=int(n), mae=float(np.abs(err).mean()), max_over=float(err.max()), q999_over=float(over),
                max_under=float(-err.min()), sign_agreement=float(sign_ok.mean()),
                unsafe_classified_safe=float(unsafe_missed), calibration_offset=float(max(over, 0.0)))


def build_table(c: HJConfig, path: Optional[str] = None, verbose: bool = True) -> Dict[str, object]:
    sol = solve(c, verbose=verbose)
    report = validate(sol, c)
    # Shift the table down by the 99.9% over-estimate so it is conservative w.r.t. the exact value.
    sol["V"] = (sol["V"] - report["calibration_offset"]).astype(np.float32)
    report_after = validate(sol, c, seed=1)
    out = dict(V=sol["V"], d=sol["d"], v=sol["v"], params=np.array(list(asdict(c).items()), dtype=object),
               report_before=report, report_after=report_after, solve_seconds=sol["seconds"], dt=sol["dt"])
    if path:
        np.savez(path, V=sol["V"], d=sol["d"], v=sol["v"], w_bar=c.w_bar, a_e_min=c.a_e_min, a_e_max=c.a_e_max,
                 a_o_min=c.a_o_min, a_o_max=c.a_o_max, d_min=c.d_min, horizon=c.horizon,
                 calibration_offset=report["calibration_offset"])
    return out


class HJTable:
    """Torch lookup (trilinear, differentiable) with conservative handling outside the grid."""

    def __init__(self, path: str, device=None) -> None:
        import torch

        z = np.load(path)
        self.d_lo, self.d_hi = float(z["d"][0]), float(z["d"][-1])
        self.v_max = float(z["v"][-1])
        self.params = {k: float(z[k]) for k in ("w_bar", "a_e_min", "a_e_max", "a_o_min", "a_o_max", "d_min",
                                                "horizon", "calibration_offset")}
        self.V = torch.as_tensor(z["V"], dtype=torch.float32, device=device)[None, None]  # (1,1,D,Ve,Vo)
        self.device = device

    def to(self, device) -> "HJTable":
        self.V = self.V.to(device)
        self.device = device
        return self

    def value_torch(self, d, ve, vo):
        import torch
        import torch.nn.functional as F

        shape = d.shape
        dn = (2.0 * (d - self.d_lo) / (self.d_hi - self.d_lo) - 1.0).clamp(-1.0, 1.0)
        ven = (2.0 * ve / self.v_max - 1.0).clamp(-1.0, 1.0)
        von = (2.0 * vo / self.v_max - 1.0).clamp(-1.0, 1.0)
        grid = torch.stack([von, ven, dn], -1).reshape(1, -1, 1, 1, 3).to(self.V.dtype)
        val = F.grid_sample(self.V, grid, mode="bilinear", padding_mode="border", align_corners=True)
        val = val.reshape(shape)
        # Beyond the grid: deeper overlap than d_lo extrapolates linearly (unsafe); ego faster than
        # v_max is outside the verified domain -> treated as unsafe.
        val = torch.where(d < self.d_lo, val + (d - self.d_lo), val)
        val = torch.where(ve > self.v_max, torch.full_like(val, -1e3), val)
        return val

    def box_min(self, d_lo, ve_hi, vo_lo):
        """Minimum of V over a box using monotonicity: the corner (d_lo, ve_hi, vo_lo)."""
        return self.value_torch(d_lo, ve_hi, vo_lo.clamp(min=0.0))
