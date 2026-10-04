"""Kinematic bicycle (rear-axle reference) used by the policy rollout, MPC, HJ and the plant.

State (x, y, psi, v); control (a, kappa) = (longitudinal acceleration, path curvature).
Controls are piecewise constant over ``action_dt``; integration uses ``dt`` with
the midpoint rule on heading, which is exact for constant (a, kappa) when v
does not hit zero inside the step.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .config import Config
from .features import steps_per_action


def rollout(v0: torch.Tensor, ctrl: torch.Tensor, cfg: Config, n_steps: Optional[int] = None,
            x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Roll controls (..., Ha, 2) forward from (0, 0, 0, v0) (or x0 = (..., 4)) -> (..., n_steps, 4)."""
    r = steps_per_action(cfg)
    dt = cfg.data.dt
    a = ctrl[..., 0].repeat_interleave(r, dim=-1)
    k = ctrl[..., 1].repeat_interleave(r, dim=-1)
    n = a.shape[-1] if n_steps is None else min(n_steps, a.shape[-1])
    if x0 is None:
        x = torch.zeros_like(v0)
        y = torch.zeros_like(v0)
        psi = torch.zeros_like(v0)
        v = v0
    else:
        x, y, psi, v = x0.unbind(-1)
    out = []
    for i in range(n):
        v_next = torch.relu(v + a[..., i] * dt)
        vm = 0.5 * (v + v_next)
        dpsi = k[..., i] * vm * dt
        pm = psi + 0.5 * dpsi
        x = x + vm * torch.cos(pm) * dt
        y = y + vm * torch.sin(pm) * dt
        psi = psi + dpsi
        v = v_next
        out.append(torch.stack([x, y, psi, v], -1))
    return torch.stack(out, -2)


def rollout_np(v0: float, ctrl: np.ndarray, dt: float, steps_per_act: int) -> np.ndarray:
    a = np.repeat(ctrl[:, 0], steps_per_act)
    k = np.repeat(ctrl[:, 1], steps_per_act)
    x = y = psi = 0.0
    v = v0
    out = np.zeros((len(a), 4))
    for i in range(len(a)):
        v_next = max(v + a[i] * dt, 0.0)
        vm = 0.5 * (v + v_next)
        dpsi = k[i] * vm * dt
        pm = psi + 0.5 * dpsi
        x += vm * np.cos(pm) * dt
        y += vm * np.sin(pm) * dt
        psi += dpsi
        v = v_next
        out[i] = (x, y, psi, v)
    return out


class Plant:
    """Executed vehicle: kinematic bicycle with first-order actuator lag, bounded noise and delay.

    This is deliberately *not* the planner's model: the difference is the model
    mismatch that the HJ disturbance bound w_bar must cover. ``estimate_disturbance``
    measures |a_executed - a_commanded| under random excitation.
    """

    def __init__(self, cfg: Config, n: int, device: torch.device, generator: Optional[torch.Generator] = None):
        s = cfg.sim
        self.dt = cfg.data.dt
        self.tau_a, self.tau_k = s.tau_a, s.tau_k
        self.noise = s.accel_noise
        self.delay = s.delay_steps
        self.gen = generator
        self.device = device
        self.a_act = torch.zeros(n, device=device)
        self.k_act = torch.zeros(n, device=device)
        self.queue = [torch.zeros(n, 2, device=device) for _ in range(self.delay)]
        self.a_min, self.a_max = cfg.feat.a_min, cfg.feat.a_max
        self.k_max = cfg.feat.kappa_max

    def reset(self, a0: torch.Tensor, k0: torch.Tensor) -> None:
        self.a_act = a0.clone()
        self.k_act = k0.clone()
        self.queue = [torch.stack([a0, k0], -1) for _ in range(self.delay)]

    def step(self, state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """state (n, 4) [x, y, psi, v] world; cmd (n, 2) [a, kappa] -> next state."""
        if self.delay:
            self.queue.append(cmd)
            cmd = self.queue.pop(0)
        a_c = cmd[:, 0].clamp(self.a_min, self.a_max)
        k_c = cmd[:, 1].clamp(-self.k_max, self.k_max)
        alpha_a = 1.0 if self.tau_a <= 0 else min(1.0, self.dt / self.tau_a)
        alpha_k = 1.0 if self.tau_k <= 0 else min(1.0, self.dt / self.tau_k)
        self.a_act = self.a_act + alpha_a * (a_c - self.a_act)
        self.k_act = self.k_act + alpha_k * (k_c - self.k_act)
        noise = (torch.rand(self.a_act.shape, device=self.device, generator=self.gen) * 2 - 1) * self.noise
        a = self.a_act + noise
        x, y, psi, v = state.unbind(-1)
        v_next = torch.relu(v + a * self.dt)
        vm = 0.5 * (v + v_next)
        dpsi = self.k_act * vm * self.dt
        pm = psi + 0.5 * dpsi
        return torch.stack([x + vm * torch.cos(pm) * self.dt, y + vm * torch.sin(pm) * self.dt,
                            psi + dpsi, v_next], -1)


@torch.no_grad()
def estimate_disturbance(cfg: Config, n: int = 4096, seconds: float = 60.0, quantile: float = 0.999,
                         device: Optional[torch.device] = None) -> dict:
    """Excite the plant with random held commands and measure the *steady-state* acceleration mismatch.

    Actuator lag is handled explicitly by the shield (the commit tube allows the
    executed acceleration to lie between the previous and the new command for
    3 time constants), so w_bar only has to cover what remains after the
    actuator has settled: noise, delay residue and integration error. Each
    command is held for 3*tau + one control period; the mismatch is measured
    over the final control period.
    """
    device = device or torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(1234)
    plant = Plant(cfg, n, device, gen)
    plant.reset(torch.zeros(n, device=device), torch.zeros(n, device=device))
    r = steps_per_action(cfg)
    settle = int(np.ceil(3.0 * max(cfg.sim.tau_a, 0.0) / cfg.data.dt)) + cfg.sim.delay_steps
    state = torch.zeros(n, 4, device=device)
    state[:, 3] = 5.0 + torch.rand(n, device=device, generator=gen) * 15.0
    steps = max(1, int(seconds / ((settle + r) * cfg.data.dt)))
    errs = []
    lo, hi = cfg.hj.a_e_min, cfg.hj.a_e_max
    for _ in range(steps):
        cmd_a = lo + (hi - lo) * torch.rand(n, device=device, generator=gen)
        cmd_k = (torch.rand(n, device=device, generator=gen) * 2 - 1) * 0.05
        cmd = torch.stack([cmd_a, cmd_k], -1)
        for _ in range(settle):
            state = plant.step(state, cmd)
        v_before = state[:, 3].clone()
        for _ in range(r):
            state = plant.step(state, cmd)
        a_eff = (state[:, 3] - v_before) / cfg.feat.action_dt
        moving = (v_before > 0.5) & (state[:, 3] > 0.5)
        errs.append((a_eff - cmd_a)[moving])
        # re-randomise speeds that hit the bounds so every sample is informative
        bad = (state[:, 3] < 1.0) | (state[:, 3] > cfg.hj.v_max - 1.0)
        state[bad, 3] = 5.0 + torch.rand(int(bad.sum()), device=device, generator=gen) * 15.0
    e = torch.cat(errs).abs().cpu().numpy()
    return dict(w_bar_quantile=float(np.quantile(e, quantile)), w_bar_max=float(e.max()),
                quantile=quantile, samples=int(e.size), tau_a=cfg.sim.tau_a, noise=cfg.sim.accel_noise,
                delay_steps=cfg.sim.delay_steps)
