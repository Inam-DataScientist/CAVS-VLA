"""End-to-end self-test on synthetic nuPlan-format logs. Run this first on every new machine:

    python -m cavs_vla selftest --workdir /tmp/cavs_selftest

Every gate prints PASS/FAIL; the process exits non-zero if any gate fails.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from .config import Config, dump_json, load_config
from .data.build import RawSceneStore, build_dataset
from .data.waymo import scenario_to_scene
from .features import featurize, scene_to_device
from .synthetic import make_fake_womd_scenario, make_synthetic_nuplan


def _cfg(work: str, db_root: str, map_json: str) -> Config:
    return load_config(None, [
        f"data.nuplan_db_root={db_root}", "data.map_backend=json", f"data.json_map_path={map_json}",
        f"data.out_dir={work}/data", "data.num_workers=2", f"data.split_file={work}/splits.json",
        "data.window_stride_s=0.5", "data.max_polylines=96",
        "model.d_model=64", "model.n_heads=4", "model.enc_layers=2", "model.dec_layers=1",
        "train.batch_size=16", "train.epochs=2", "train.num_workers=0", "train.warmup_steps=5",
        "train.log_every=10", f"train.ckpt_dir={work}/ckpt", "train.amp=bf16",
        "hj.n_d=103", "hj.n_v=31", f"hj.table_path={work}/hj.npz",
        f"conformal.table_path={work}/conformal.json", "conformal.max_batches=20",
        "shield.mpc_outer=2", "shield.mpc_inner=8",
        "sim.num_scenarios=8", "sim.batch_size=4", "sim.seeds=[0]", f"sim.out_dir={work}/closed_loop",
    ])


def run(workdir: str = "/tmp/cavs_selftest", keep: bool = False) -> bool:
    if os.path.exists(workdir) and not keep:
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)
    results: List[Tuple[str, bool, str]] = []
    state: Dict = {}

    def gate(name: str, fn: Callable[[], str]) -> None:
        t0 = time.time()
        try:
            msg = fn()
            results.append((name, True, f"{msg} ({time.time() - t0:.1f}s)"))
            print(f"[PASS] {name}: {msg}")
        except Exception as exc:  # every gate reports its own failure and the run continues
            results.append((name, False, repr(exc)))
            print(f"[FAIL] {name}: {exc!r}")
            traceback.print_exc()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def g_data():
        info = make_synthetic_nuplan(os.path.join(workdir, "raw"), n_logs=10, duration_s=30.0)
        logs = [os.path.splitext(os.path.basename(p))[0] for p in json.loads(info["logs"])]
        splits = {n: ("train" if i < 6 else "val" if i < 8 else "test") for i, n in enumerate(logs)}
        with open(os.path.join(workdir, "splits.json"), "w") as f:
            json.dump(splits, f)
        cfg = _cfg(workdir, info["db_root"], info["map_json"])
        state["cfg"] = cfg
        m = build_dataset(cfg, overwrite=True)
        assert all(m["counts"][s] > 0 for s in ("train", "val", "test")), m["counts"]
        assert not any(m["split_log_intersections"].values())
        sc = scenario_to_scene(cfg, make_fake_womd_scenario(1))
        assert sc is not None and sc["traj"].shape[0] == cfg.data.max_actors
        return f"counts={m['counts']} route={m['route_sources']}"

    def g_leak():
        cfg = state["cfg"]
        st = RawSceneStore(cfg, os.path.join(cfg.data.out_dir, "train"))
        batch = st.get_batch(list(range(min(8, len(st)))))
        sc = scene_to_device({k: torch.as_tensor(v) for k, v in batch.items()}, device)
        K = cfg.data.hist_steps
        B = sc["traj"].shape[0]
        actor = torch.zeros(B, dtype=torch.long, device=device)
        f1 = featurize(sc, K - 1, actor, cfg)
        sc2 = {k: v.clone() for k, v in sc.items()}
        g = torch.Generator(device=device).manual_seed(0)
        sc2["traj"][:, :, K:] += torch.randn(sc2["traj"][:, :, K:].shape, device=device, generator=g) * 50
        sc2["valid"][:, :, K:] = ~sc2["valid"][:, :, K:]
        sc2["poly_tl"][:, :, K:] = 1
        f2 = featurize(sc2, K - 1, actor, cfg)
        keys = ["ego", "ego_valid", "agents", "agents_valid", "map", "map_valid", "map_attr", "instr", "ego_speed",
                "agents_phys"]
        for k in keys:
            assert torch.equal(f1[k], f2[k]), f"input '{k}' changed when only future frames changed"
        return "no input depends on frames after t0"

    def g_model():
        from .model.layers import precise_matmul
        from .model.vla import VLAPolicy
        from .safety.verifier import input_box
        cfg = state["cfg"]
        model = VLAPolicy(cfg).to(device).eval()
        st = RawSceneStore(cfg, os.path.join(cfg.data.out_dir, "train"))
        sc = scene_to_device({k: torch.as_tensor(v) for k, v in st.get_batch(list(range(4))).items()}, device)
        K = cfg.data.hist_steps
        feat = featurize(sc, K - 1, torch.zeros(4, dtype=torch.long, device=device), cfg, with_targets=True)
        out = model(feat)
        assert out["traj"].shape == (4, cfg.model.n_modes, cfg.data.fut_steps, 4)
        assert torch.isfinite(out["traj"]).all()
        zero = {k: feat[k].float() for k in ("ego", "agents", "map", "map_attr")}
        b0 = model.bounds(feat, zero, zero)
        gap = float(torch.maximum((b0["ctrl_lo"] - out["ctrl"]).abs(), (b0["ctrl_hi"] - out["ctrl"]).abs()).max())
        assert gap < 1e-3, f"zero-width IBP differs from forward by {gap}"
        lo, hi = input_box(feat, cfg)
        bnd = model.bounds(feat, lo, hi)
        viol = 0
        for _ in range(32):
            f = dict(feat)
            for k in ("ego", "agents", "map"):
                f[k] = lo[k] + (hi[k] - lo[k]) * torch.rand_like(lo[k])
            with precise_matmul():
                o = model(f)
            viol += int(((o["ctrl"] < bnd["ctrl_lo"] - 1e-4) | (o["ctrl"] > bnd["ctrl_hi"] + 1e-4)).sum())
        assert viol == 0, f"{viol} sampled outputs escaped the IBP bounds"
        state["model_params"] = sum(p.numel() for p in model.parameters())
        width = float((bnd["ctrl_hi"] - bnd["ctrl_lo"])[..., 0, 0].mean())
        return f"params={state['model_params']:,} zero-width gap={gap:.2e} sampled-violations=0 mean accel width={width:.3f}"

    def g_train():
        from .train import train
        from .evaluate import kinematic_fit
        cfg = state["cfg"]
        ck = train(cfg, run_name="selftest")
        assert os.path.exists(ck)
        hist = json.load(open(os.path.join(os.path.dirname(ck), "training_history.json")))
        l0, l1 = hist[0]["train_loss"], hist[-1]["train_loss"]
        assert all(math.isfinite(h["train_loss"]) for h in hist)
        state["ckpt"] = ck
        kf = kinematic_fit(cfg, "train", max_batches=5)
        return f"loss {l0:.3f} -> {l1:.3f}; kinematic fit ADE={kf['ADE_mean']:.3f} m"

    def g_hj():
        from .safety.hj import HJTable, build_table, interp_np
        cfg = state["cfg"]
        out = build_table(cfg.hj, cfg.hj.table_path, verbose=False)
        rep = out["report_after"]
        assert rep["unsafe_classified_safe"] < 0.005, rep
        assert rep["sign_agreement"] > 0.97, rep
        tab = HJTable(cfg.hj.table_path, device)
        z = np.load(cfg.hj.table_path)
        d = np.array([5.0, 20.0, 40.0]); ve = np.array([10.0, 15.0, 5.0]); vo = np.array([0.0, 10.0, 3.0])
        ref = interp_np(dict(V=z["V"], d=z["d"], v=z["v"]), d, ve, vo)
        got = tab.value_torch(torch.tensor(d, device=device).float(), torch.tensor(ve, device=device).float(),
                              torch.tensor(vo, device=device).float()).cpu().numpy()
        assert np.allclose(ref, got, atol=1e-3), (ref, got)
        state["hj"] = tab
        return f"MAE={rep['mae']:.2f} m sign={rep['sign_agreement']:.4f} unsafe->safe={rep['unsafe_classified_safe']}"

    def g_conformal():
        from .evaluate import calibrate_conformal
        cfg = state["cfg"]
        table = calibrate_conformal(cfg, state["ckpt"])
        assert all(np.isfinite(v) for v in table["q"].values() if v is not None)
        state["conf"] = table
        return f"q={ {k: round(v, 2) for k, v in table['q'].items()} } coverage={table['test_coverage']}"

    def g_shield():
        from .safety.shield import SRC_NOMINAL, Shield
        from .config import METHOD_TABLE
        cfg = state["cfg"]
        sh = Shield(cfg, state["hj"], None)
        Ha = cfg.data.fut_steps // int(round(cfg.feat.action_dt / cfg.data.dt))
        H = cfg.data.fut_steps
        N = cfg.feat.num_agents
        B = 3
        from .dynamics import rollout
        v0 = torch.tensor([15.0, 15.0, 10.0], device=device)
        ctrl = torch.zeros(B, Ha, 2, device=device)
        traj = rollout(v0, ctrl, cfg)
        phys = torch.zeros(B, N, 8, device=device)
        ok = torch.zeros(B, N, dtype=torch.bool, device=device)
        # scene 0: stopped car 15 m ahead; scene 1: car 90 m ahead at 15 m/s; scene 2: pedestrian about to cross 12 m ahead
        phys[0, 0] = torch.tensor([15.0 + 1.5, 0.0, 0.0, 0.0, 0.0, 4.5, 1.9, 0.0], device=device)
        phys[1, 0] = torch.tensor([90.0, 0.0, 0.0, 15.0, 0.0, 4.5, 1.9, 0.0], device=device)
        phys[2, 0] = torch.tensor([12.0, -4.0, math.pi / 2, 0.0, 1.4, 0.6, 0.6, 1.0], device=device)
        ok[:, 0] = True
        t = torch.arange(1, H + 1, device=device).float() * cfg.data.dt
        pred = phys[:, :, None, :2].expand(B, N, H, 2).clone()
        pred[..., 0] = pred[..., 0] + phys[:, :, None, 3] * t
        pred[..., 1] = pred[..., 1] + phys[:, :, None, 4] * t
        rad = torch.full((B, N, H), 0.5, device=device)
        dims = torch.tensor([[5.176, 2.297, 1.461]] * B, device=device)
        res = sh.step(METHOD_TABLE["B7"], v0, ctrl, traj, ctrl - 0.05, ctrl + 0.05, phys, ok, pred, rad, dims,
                      torch.zeros(B, device=device))
        cert = res["certified"].tolist()
        assert cert == [False, True, False], f"certified={cert}"
        assert float(res["cmd"][0, 0]) < -3.0, f"stopped-lead scene must brake, got {res['cmd'][0]}"
        assert float(res["cmd"][2, 0]) < 0.0, f"pedestrian scene must slow down, got {res['cmd'][2]}"
        assert int(res["source"][1]) == SRC_NOMINAL
        return f"certified={cert} cmds={[round(float(a), 2) for a in res['cmd'][:, 0]]} sources={res['source'].tolist()}"

    def g_closed_loop():
        from .evaluate import closed_loop
        cfg = state["cfg"]
        res = closed_loop(cfg, state["ckpt"], methods=["B0", "B1", "B3", "B6", "B7"], split="test")
        s = res["summary"]
        for m in ("B0", "B1", "B7"):
            assert m in s and s[m]["n"]["value"] > 0
        return "B1 collision={:.2f} B7 collision={:.2f} B7 interventions={:.3f}".format(
            s["B1"]["collision"]["rate"], s["B7"]["collision"]["rate"], s["B7"]["intervention_rate"]["mean"])

    def g_multi():
        from .evaluate import closed_loop
        cfg = copy.deepcopy(state["cfg"])
        cfg.sim.num_controlled = 2
        cfg.sim.replan_interval = 2
        cfg.sim.latency_steps = 1
        res = closed_loop(cfg, state["ckpt"], methods=["B7"], split="test", tag="multi")
        return f"records={res['summary']['B7']['n']['value']}"

    def g_open_and_verify():
        from .evaluate import certificate_sweep, open_loop
        cfg = state["cfg"]
        ol = open_loop(cfg, state["ckpt"], split="test", out_dir=os.path.join(workdir, "open_loop"))
        sw = certificate_sweep(cfg, state["ckpt"], split="test", scales=(0.0, 1.0), max_batches=2,
                               onnx_dir=os.path.join(workdir, "onnx"))
        assert sw["sampling_soundness"]["violations"] == 0, sw["sampling_soundness"]
        return f"ADE={ol['model']['ADE']:.2f} CV={ol['CV']['ADE']:.2f} width@x1={sw['x1.0']['width_accel_mean']:.3f}"

    def g_ceg():
        from .evaluate import mine_failures
        from .train import ceg_refine
        cfg = copy.deepcopy(state["cfg"])
        cfg.train.ceg_epochs = 1
        path = os.path.join(workdir, "failures.json")
        fm = mine_failures(cfg, state["ckpt"], method="B1", out_path=path, max_scenes=8)
        if not fm["indices"]:
            fm["indices"] = [0]
            dump_json(fm, path)
        ck = ceg_refine(cfg, state["ckpt"], path, "selftest_ceg", hj_table=state["hj"])
        assert os.path.exists(ck)
        return f"mined {len(fm['indices'])} failing train scenes; refined checkpoint written"

    gate("data build (nuPlan schema + WOMD converter)", g_data)
    gate("no-future-leakage invariant", g_leak)
    gate("model forward + IBP soundness", g_model)
    gate("training + kinematic fit", g_train)
    gate("HJ table vs closed form", g_hj)
    gate("conformal calibration", g_conformal)
    gate("shield decisions on scripted states", g_shield)
    gate("closed loop B0/B1/B3/B6/B7", g_closed_loop)
    gate("multi-CAV coordination + latency", g_multi)
    gate("open-loop eval + certificate sweep + ONNX", g_open_and_verify)
    gate("counterexample mining + CEG refinement", g_ceg)
    ok = all(r[1] for r in results)
    print("\n" + "=" * 72)
    for name, passed, msg in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}  |  {msg}")
    print("=" * 72)
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    dump_json([dict(gate=n, passed=p, detail=m) for n, p, m in results], os.path.join(workdir, "selftest_report.json"))
    return ok
