"""Product of the world MDP with the LTL monitor.

A product state is ``v = s * nQ + q``: world state ``s`` together with monitor
state ``q``, where ``q`` already accounts for the label of ``s``. Transitions:

    (s, q) --a--> (s', delta(q, L(s')))      with probability  P(s' | s, a)

Product states whose monitor state is ``false`` are *bad*: the LTL safety
specification has been violated on every continuation. Safety of the LTL
specification therefore reduces to never visiting a bad product state.
"""

from __future__ import annotations

import numpy as np

from .logic.ltl import Monitor
from .world import World


class Product:
    def __init__(self, world: World, monitor: Monitor):
        if tuple(monitor.alphabet) != tuple(world.alphabet):
            raise ValueError("monitor must be built over the world's label alphabet")
        self.world = world
        self.monitor = monitor
        nq = monitor.n_states
        S, A = world.n_states, world.n_actions
        self.nq = nq
        self.n_states = S * nq
        self.n_actions = A
        label_next = world.label_id[world.next]  # (S, A, 2)
        q_next = np.moveaxis(monitor.delta[:, label_next], 0, 1)  # (S, nq, A, 2)
        nxt = world.next[:, None, :, :] * nq + q_next
        self.next = np.ascontiguousarray(nxt.reshape(S * nq, A, 2))
        prob = np.broadcast_to(world.prob[:, None, :, :], (S, nq, A, 2))
        self.prob = np.ascontiguousarray(prob.reshape(S * nq, A, 2))
        self.world_state = np.repeat(np.arange(S), nq)
        self.q = np.tile(np.arange(nq), S)
        self.bad = monitor.is_bad[self.q]
        self.world_sink = world.is_sink[self.world_state]
        self.landed = self.world_state == world.sink["landed"]
        self.terminal = self.world_sink | self.bad

    def initial(self, s: int | np.ndarray) -> np.ndarray | int:
        """Product state for starting in world state ``s`` (label of ``s`` consumed)."""
        q0 = self.monitor.delta[0, self.world.label_id[s]]
        out = np.asarray(s) * self.nq + q0
        return int(out) if np.ndim(out) == 0 else out

    def initial_states(self) -> np.ndarray:
        return np.asarray(self.initial(self.world.start_states), dtype=np.int64)

    def atoms(self) -> dict[str, np.ndarray]:
        """Atomic propositions over product states (world atoms + spec atoms)."""
        out = {k: v[self.world_state] for k, v in self.world.atoms().items()}
        out["spec_violated"] = self.bad.copy()
        out["spec_satisfied"] = self.monitor.is_true[self.q]
        out["terminal"] = self.terminal.copy()
        return out

    def describe(self, v: int) -> str:
        s, q = divmod(int(v), self.nq)
        return f"{self.world.describe(s)} | monitor q{q}: {self.monitor.states[q]}"
