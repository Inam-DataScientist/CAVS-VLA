import numpy as np

from _helpers import small_cfg
from safeland.config import Config
from safeland.world import SINKS, World


def _world():
    return World(small_cfg().world)


def test_probabilities_are_distributions():
    w = _world()
    assert np.all(w.prob >= 0)
    assert np.allclose(w.prob.sum(axis=2), 1.0)


def test_sinks_absorbing():
    w = _world()
    for name in SINKS:
        s = w.sink[name]
        assert np.all(w.next[s] == s)


def test_tables_match_direct_dynamics():
    w = _world()
    for s in range(w.n_states):
        for a in range(w.n_actions):
            direct = {}
            for p, t in w.outcomes(s, a):
                direct[t] = direct.get(t, 0.0) + p
            table = {}
            for k in range(2):
                if w.prob[s, a, k] > 0:
                    t = int(w.next[s, a, k])
                    table[t] = table.get(t, 0.0) + w.prob[s, a, k]
            assert direct.keys() == table.keys()
            for t in direct:
                assert abs(direct[t] - table[t]) < 1e-12


def test_battery_strictly_decreases_so_runs_terminate():
    w = _world()
    for s in range(w.n_fly):
        b = w.decode(s)[3]
        for a in range(w.n_actions):
            for k in range(2):
                t = int(w.next[s, a, k])
                if w.prob[s, a, k] > 0 and t < w.n_fly:
                    assert w.decode(t)[3] < b


def test_wind_only_at_altitude():
    w = _world()
    for s in range(w.n_fly):
        for a in range(w.n_actions):
            outs = w.outcomes(s, a)
            if len(outs) == 2:
                _, _, z, _ = w.decode(s)
                dz = {5: 1, 6: -1}.get(a, 0)
                assert z + dz >= w.cfg.wind_min_alt


def test_events():
    w = World(Config().world)
    assert w.resolve(6, 6, 0, 5) == w.sink["landed"]
    assert w.resolve(0, 0, 0, 5) == w.sink["crash"]
    assert w.resolve(3, 3, 2, 5) == w.sink["nfz"]
    assert w.resolve(5, 2, 2, 5) == w.sink["collision"]
    assert w.resolve(5, 2, 3, 5) < w.n_fly  # above the building
    assert w.resolve(-1, 0, 2, 5) == w.sink["geofence"]
    assert w.resolve(0, 0, 2, 0) == w.sink["empty"]
    assert w.resolve(6, 6, 0, 0) == w.sink["landed"]  # touching down on the last unit is fine


def test_start_states_are_flying():
    w = _world()
    assert np.all(w.start_states < w.n_fly)
    assert all(w.decode(int(s))[2] == w.cfg.start_alt for s in w.start_states)
