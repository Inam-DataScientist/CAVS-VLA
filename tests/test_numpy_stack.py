"""Tests for every component that does not need torch (runs anywhere: `python tests/test_numpy_stack.py`).

The torch components are covered by `python -m cavs_vla selftest`, which must pass on the GPU machine.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cavs_vla.config import HJConfig, load_config, parse_overrides  # noqa: E402


def test_config_overrides():
    cfg = load_config(None, ["train.lr=1e-4", "data.sources=[nuplan,waymo]", "sim.reactive=false"])
    assert cfg.train.lr == 1e-4 and cfg.data.sources == ["nuplan", "waymo"] and cfg.sim.reactive is False
    try:
        load_config(None, ["train.nope=1"])
        raise AssertionError("unknown key accepted")
    except KeyError:
        pass
    assert parse_overrides(["a.b=3"]) == {"a": {"b": 3}}


def test_nuplan_build_and_splits():
    from cavs_vla.data.build import RawSceneStore, build_dataset
    from cavs_vla.language import INTENTS, detokenize
    from cavs_vla.synthetic import make_synthetic_nuplan
    with tempfile.TemporaryDirectory() as tmp:
        info = make_synthetic_nuplan(os.path.join(tmp, "raw"), n_logs=4, duration_s=25.0)
        logs = [os.path.splitext(os.path.basename(p))[0] for p in json.loads(info["logs"])]
        with open(os.path.join(tmp, "splits.json"), "w") as f:
            json.dump({n: ["train", "train", "val", "test"][i] for i, n in enumerate(logs)}, f)
        cfg = load_config(None, [f"data.nuplan_db_root={info['db_root']}", "data.map_backend=json",
                                 f"data.json_map_path={info['map_json']}", f"data.out_dir={tmp}/ds",
                                 "data.num_workers=1", f"data.split_file={tmp}/splits.json"])
        m = build_dataset(cfg, overwrite=True)
        assert m["counts"]["train"] > 0 and m["counts"]["val"] > 0 and m["counts"]["test"] > 0
        assert not any(m["split_log_intersections"].values())
        st = RawSceneStore(cfg, f"{tmp}/ds/train")
        s = st.get(0)
        K = cfg.data.hist_steps
        assert np.allclose(s["traj"][0, K - 1, :3], 0, atol=1e-4)
        assert s["valid"][0].all()
        assert detokenize(s["instr"]).startswith(("go straight", "turn", "change", "follow"))
        assert 0 <= int(s["intent"]) < len(INTENTS)
        # the map comes from the map file: centrelines are lanes at y = 0 / 3.5 in world, never the ego path
        lane = s["poly_attr"][:, 0] > 0.5
        assert lane.any()
        # 10 Hz frames
        assert abs(m["logs"][0]["frames_10hz"] - 25.0 / 0.1) < 3


def test_map_is_not_the_future():
    """The stored map must not coincide with the ego's future path (the BASELINE-01 leak)."""
    from cavs_vla.data.build import RawSceneStore, build_dataset
    from cavs_vla.geometry_np import points_polyline_min_distance
    from cavs_vla.synthetic import make_synthetic_nuplan
    with tempfile.TemporaryDirectory() as tmp:
        info = make_synthetic_nuplan(os.path.join(tmp, "raw"), n_logs=2, duration_s=25.0)
        cfg = load_config(None, [f"data.nuplan_db_root={info['db_root']}", "data.map_backend=json",
                                 f"data.json_map_path={info['map_json']}", f"data.out_dir={tmp}/ds",
                                 "data.num_workers=1", "data.split_ratios=[1.0,0.0,0.0]"])
        build_dataset(cfg, overwrite=True)
        st = RawSceneStore(cfg, f"{tmp}/ds/train")
        K = cfg.data.hist_steps
        changed_lane = 0
        for i in range(len(st)):
            s = st.get(i)
            fut = s["traj"][0, K:, :2]
            if abs(fut[-1, 1]) > 2.0:          # windows in which the ego changes lanes
                changed_lane += 1
                d = min(points_polyline_min_distance(fut, s["poly"][p]).max()
                        for p in np.nonzero(s["poly_valid"].any(1))[0])
                assert d > 0.5, "a single map polyline follows the whole future path: future leakage"
        assert changed_lane > 0


def test_waymo_conversion_and_tfrecord():
    from cavs_vla.data.waymo import crc32c, read_tfrecord, scenario_to_scene, write_tfrecord
    from cavs_vla.synthetic import make_fake_womd_scenario
    cfg = load_config(None, [])
    s = scenario_to_scene(cfg, make_fake_womd_scenario(0))
    assert s is not None and s["valid"][0].all()
    assert set(np.unique(s["poly_tl"])) >= {0}
    assert crc32c(b"123456789") == 0xE3069283
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.tfrecord")
        write_tfrecord(p, [b"a" * 10, b"b" * 1000])
        assert [len(r) for r in read_tfrecord(p, verify_crc=True)] == [10, 1000]


def test_hj_conservative():
    from cavs_vla.safety.hj import build_table
    c = HJConfig(n_d=103, n_v=31)
    with tempfile.TemporaryDirectory() as tmp:
        out = build_table(c, os.path.join(tmp, "hj.npz"), verbose=False)
    rep = out["report_after"]
    assert rep["unsafe_classified_safe"] == 0.0, rep
    assert rep["sign_agreement"] > 0.98, rep
    assert rep["max_over"] < 1e-6, rep


def test_conformal_coverage():
    from cavs_vla.safety.conformal import calibrate, coverage, scores
    rng = np.random.default_rng(0)
    n, H = 4000, 40
    gt = rng.normal(size=(n, H, 2)).cumsum(1)
    scale = np.abs(rng.normal(1, 0.2, size=(n, H, 2))) * np.linspace(0.2, 3, H)[None, :, None]
    pred = gt + rng.laplace(size=(n, H, 2)) * scale
    valid = rng.random((n, H)) > 0.1
    types = rng.integers(0, 2, n)
    s = scores(pred, scale, gt, valid, 0.2)
    cal = calibrate([s[:2000]], [types[:2000]], 0.1, 0.2)
    cov = coverage([s[2000:]], [types[2000:]], cal)
    assert 0.87 < cov["overall"] < 0.93, cov


def test_legacy_probe_flags_leak():
    from cavs_vla.legacy_probe import leak_report_numpy, map_proxy
    K, H = 8, 30
    rng = np.random.default_rng(1)
    Xs, Ys, Ms, Cs = [], [], [], []
    for _ in range(100):
        v, a = rng.uniform(3, 14), rng.normal(0, 0.8)
        t = np.arange(-(K - 1), H + 1) * 0.05
        x = v * t + 0.5 * a * t ** 2
        y = 0.002 * x ** 2
        x, y = x - x[K - 1], y - y[K - 1]
        X = np.zeros((K, 5)); X[:, 0], X[:, 1], X[:, 2] = x[:K], y[:K], v + a * t[:K]
        Y = np.zeros((H, 5)); Y[:, 0], Y[:, 1], Y[:, 2] = x[K:], y[K:], v + a * t[K:]
        Xs.append(X); Ys.append(Y); Ms.append(map_proxy(np.stack([x, y], 1)))
        Cs.append([np.clip(Y[:, 2].mean() / 20, 0, 1), 1.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "val.npz")
        np.savez(p, X=np.array(Xs), Y=np.array(Ys), M=np.array(Ms), C=np.array(Cs))
        rep = leak_report_numpy(p)
    assert rep["verdict"].startswith("LEAK"), rep


def test_carla_conversions():
    src = open(os.path.join(os.path.dirname(__file__), "..", "cavs_vla", "sim", "carla_env.py")).read()
    ns: dict = {}
    code = src[src.index("def lh_to_rh_xy"):src.index("class SpeedController")]
    exec("import math\nimport numpy as np\nfrom typing import Tuple\n" + code, ns)
    yaw = ns["lh_to_rh_yaw"](90.0)
    x, y = ns["lh_to_rh_xy"](0.0, 10.0)
    assert np.allclose([10 * math.cos(yaw), 10 * math.sin(yaw)], [x, y])
    assert ns["curvature_to_steer"](0.05, 2.9, math.radians(70)) < 0     # left curvature -> negative CARLA steer


def test_metrics():
    from cavs_vla.sim.metrics import paired_tests, summarize, wilson
    p, lo, hi = wilson(5, 100)
    assert lo < p < hi
    recs = [dict(method=m, seed=0, scene=i, vehicle=0, collision=(m == "B1" and i < 10), at_fault_collision=False,
                 deadlock=False, distance_m=50.0, progress=1.0) for m in ("B1", "B7") for i in range(50)]
    s = summarize(recs)
    assert abs(s["B1"]["collision"]["rate"] - 0.2) < 1e-9
    t = paired_tests(recs, "B1", ["B7"])
    assert t["B7"]["collisions_ref_only"] == 10 and t["B7"]["mcnemar_p"] < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
