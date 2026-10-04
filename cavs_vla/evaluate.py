"""Experiments A (open loop), B/C (closed loop B0-B7), conformal calibration, certificate sweeps, failure mining."""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from .config import METHOD_TABLE, Config, dump_json
from .data.build import RawSceneStore
from .data.dataset import RawSceneDataset
from .dynamics import rollout
from .features import featurize, scene_to_device
from .language import default_instruction
from .model.layers import precise_matmul
from .safety import conformal
from .safety.hj import HJTable
from .safety.verifier import certify, export_head_onnx
from .sim.logsim import LogSim
from .sim.metrics import markdown_table, paired_tests, summarize
from .train import amp_dtype, autocast_ctx, load_policy
from .utils import environment_info, get_logger, seed_everything

HORIZONS_S = (1.0, 3.0, 5.0, 8.0)


def _loader(cfg: Config, split: str, batch_size: int, max_items: int = 0) -> DataLoader:
    ds = RawSceneDataset(cfg, split)
    if max_items and len(ds) > max_items:
        ds = RawSceneDataset(cfg, split, indices=list(range(max_items)))
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds), num_workers=cfg.train.num_workers,
                      pin_memory=torch.cuda.is_available())


def _baselines(feat: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, torch.Tensor]:
    """Constant velocity and constant turn-rate-and-velocity from the last two history frames."""
    dt = cfg.data.dt
    H = cfg.data.fut_steps
    ego = feat["ego"]
    ps, vs = cfg.feat.pos_scale, cfg.feat.vel_scale
    v = feat["ego_speed"]
    psi_prev = torch.atan2(ego[:, -2, 3], ego[:, -2, 2])
    prev_ok = feat["ego_valid"][:, -2]
    yaw_rate = torch.where(prev_ok, -psi_prev / dt, torch.zeros_like(v))
    t = torch.arange(1, H + 1, device=v.device).float() * dt
    cv = torch.stack([v[:, None] * t, torch.zeros_like(v[:, None] * t)], -1)
    psi = yaw_rate[:, None] * t
    step = v[:, None] * dt
    x = torch.cumsum(step * torch.cos(psi - 0.5 * yaw_rate[:, None] * dt), 1)
    y = torch.cumsum(step * torch.sin(psi - 0.5 * yaw_rate[:, None] * dt), 1)
    return dict(CV=cv, CTRV=torch.stack([x, y], -1))


def _ablate(feat: Dict[str, torch.Tensor], what: str, cfg: Config) -> Dict[str, torch.Tensor]:
    f = dict(feat)
    if what == "no_agents":
        f["agents_valid"] = torch.zeros_like(feat["agents_valid"])
        f["agents"] = torch.zeros_like(feat["agents"])
    elif what == "no_map":
        f["map_valid"] = torch.zeros_like(feat["map_valid"])
        f["map"] = torch.zeros_like(feat["map"])
        f["map_attr"] = torch.zeros_like(feat["map_attr"])
    elif what == "no_ego_history":
        ev = feat["ego_valid"].clone()
        ev[:, :-1] = False
        f["ego_valid"] = ev
        f["ego"] = feat["ego"] * ev[..., None].float()
    elif what == "no_language":
        d = torch.as_tensor(default_instruction(cfg.feat.instr_len), device=feat["instr"].device).long()
        f["instr"] = d[None].expand_as(feat["instr"])
    return f


@torch.no_grad()
def open_loop(cfg: Config, ckpt: str, split: str = "test", out_dir: str = "results/open_loop",
              ablations: bool = True, max_items: int = 0) -> Dict:
    log = get_logger("open_loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_m, _ = load_policy(ckpt, device)
    cfg_m.data.out_dir = cfg.data.out_dir
    dl = _loader(cfg_m, split, cfg.train.batch_size, max_items)
    K = cfg_m.data.hist_steps
    dt = cfg_m.data.dt
    hidx = [(f"{h:g}s", int(round(h / dt)) - 1) for h in HORIZONS_S if int(round(h / dt)) <= cfg_m.data.fut_steps]
    variants = ["model"] + (["no_agents", "no_map", "no_ego_history", "no_language"] if ablations else [])
    acc: Dict[str, Dict[str, list]] = {}
    intent_correct, intent_n = 0, 0
    for batch in dl:
        scene = scene_to_device(batch, device)
        B = scene["traj"].shape[0]
        feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg_m, with_targets=True)
        gt = feat["ego_fut"][..., :2]
        gv = feat["ego_fut_valid"].float()
        for name, pred in _baselines(feat, cfg_m).items():
            _accumulate(acc, name, pred[:, None], None, gt, gv, hidx)
        for vname in variants:
            f = feat if vname == "model" else _ablate(feat, vname, cfg_m)
            with autocast_ctx(device, amp_dtype(cfg_m)):
                out = model(f)
            traj = out["traj"].float()[..., :2]
            _accumulate(acc, vname, traj, out["mode_logits"].float(), gt, gv, hidx)
            if vname == "model":
                intent_correct += int((out["intent_logits"].argmax(-1) == feat["intent"]).sum())
                intent_n += B
                hv = feat["ego_fut"][..., 2]
                top = out["mode_logits"].argmax(-1)
                ph = out["traj"].float()[torch.arange(B), top][..., 2]
                acc.setdefault("model", {}).setdefault("heading_err", []).append(
                    ((torch.atan2(torch.sin(ph - hv), torch.cos(ph - hv)).abs() * gv).sum(-1) / gv.sum(-1).clamp(min=1)).cpu())
                pv = out["traj"].float()[torch.arange(B), top][..., 3]
                acc["model"].setdefault("speed_err", []).append(
                    (((pv - feat["ego_fut"][..., 3]).abs() * gv).sum(-1) / gv.sum(-1).clamp(min=1)).cpu())
    res = {}
    for name, d in acc.items():
        res[name] = {k: float(torch.cat(v).mean()) for k, v in d.items()}
    res["model"]["intent_accuracy"] = intent_correct / max(intent_n, 1)
    os.makedirs(out_dir, exist_ok=True)
    dump_json(dict(results=res, split=split, checkpoint=ckpt, horizons_s=HORIZONS_S, environment=environment_info()),
              os.path.join(out_dir, f"open_loop_{split}.json"))
    log.info(json.dumps(res, indent=1))
    return res


def _accumulate(acc, name, traj, logits, gt, gv, hidx):
    d = acc.setdefault(name, {})
    err = torch.linalg.norm(traj - gt[:, None], dim=-1)                       # (B, M, H)
    ade = (err * gv[:, None]).sum(-1) / gv.sum(-1, keepdim=True).clamp(min=1)
    fde = err[..., -1]
    top = logits.argmax(-1) if logits is not None else torch.zeros(traj.shape[0], dtype=torch.long, device=traj.device)
    bi = torch.arange(traj.shape[0], device=traj.device)
    d.setdefault("ADE", []).append(ade[bi, top].cpu())
    d.setdefault("FDE", []).append(fde[bi, top].cpu())
    d.setdefault("minADE", []).append(ade.min(1).values.cpu())
    d.setdefault("minFDE", []).append(fde.min(1).values.cpu())
    d.setdefault("miss_rate_2m", []).append((fde.min(1).values > 2.0).float().cpu())
    for label, h in hidx:
        d.setdefault(f"FDE@{label}", []).append(err[bi, top, h].cpu())


@torch.no_grad()
def calibrate_conformal(cfg: Config, ckpt: str, cal_split: str = "val", test_split: str = "test") -> Dict:
    log = get_logger("conformal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_m, _ = load_policy(ckpt, device)
    cfg_m.data.out_dir = cfg.data.out_dir
    K = cfg_m.data.hist_steps

    def collect(split):
        sc, ty = [], []
        dl = _loader(cfg_m, split, cfg.train.batch_size)
        for bi, batch in enumerate(dl):
            if cfg.conformal.max_batches and bi >= cfg.conformal.max_batches:
                break
            scene = scene_to_device(batch, device)
            B = scene["traj"].shape[0]
            feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg_m, with_targets=True)
            out = model(feat)
            s = conformal.scores(out["agent_pred"].float().cpu().numpy(),
                                 np.exp(out["agent_logscale"].float().cpu().numpy()),
                                 feat["agent_fut"].cpu().numpy(), feat["agent_fut_valid"].cpu().numpy(),
                                 cfg.conformal.min_scale)
            ok = feat["agents_ok"].cpu().numpy()
            sc.append(s[ok])
            ty.append(feat["agents_phys"][..., 7].long().cpu().numpy()[ok])
        return sc, ty

    s_cal, t_cal = collect(cal_split)
    table = conformal.calibrate(s_cal, t_cal, cfg.conformal.alpha, cfg.conformal.min_scale)
    s_te, t_te = collect(test_split)
    table["test_coverage"] = conformal.coverage(s_te, t_te, table) if s_te else {}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.conformal.table_path)), exist_ok=True)
    conformal.save(table, cfg.conformal.table_path)
    log.info(f"conformal q={table['q']} test coverage={table['test_coverage']}")
    return table


@torch.no_grad()
def certificate_sweep(cfg: Config, ckpt: str, split: str = "test", scales=(0.0, 0.5, 1.0, 2.0, 4.0),
                      max_batches: int = 20, onnx_dir: Optional[str] = None) -> Dict:
    """Certificate width vs. perception uncertainty (scaling all eps together) + soundness spot-check."""
    log = get_logger("verify")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_m, _ = load_policy(ckpt, device)
    cfg_m.data.out_dir = cfg.data.out_dir
    K = cfg_m.data.hist_steps
    base = dict(cfg.cert.__dict__)
    res = {}
    dl = _loader(cfg_m, split, min(cfg.train.batch_size, 32))
    batches = []
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        batches.append(batch)
    for sc in scales:
        for k in ("eps_agent_pos", "eps_agent_vel", "eps_agent_heading_deg", "eps_ego_speed", "eps_map_pos"):
            setattr(cfg_m.cert, k, base[k] * sc)
        w_a, w_k, inside, unique = [], [], [], []
        for batch in batches:
            scene = scene_to_device(batch, device)
            B = scene["traj"].shape[0]
            feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg_m)
            out = model(feat)
            c = certify(model, feat, cfg_m, out)
            w_a.append(c["width"][:, 0, 0].cpu())
            w_k.append(c["width"][:, 0, 1].cpu())
            inside.append(c["nominal_inside"].float().cpu())
            unique.append((c["feasible_modes"].sum(-1) == 1).float().cpu())
        res[f"x{sc}"] = dict(width_accel_mean=float(torch.cat(w_a).mean()), width_accel_p90=float(torch.cat(w_a).quantile(0.9)),
                             width_kappa_mean=float(torch.cat(w_k).mean()),
                             nominal_inside=float(torch.cat(inside).mean()),
                             single_mode_fraction=float(torch.cat(unique).mean()))
        log.info(f"eps x{sc}: {res[f'x{sc}']}")
    for k, v in base.items():
        setattr(cfg_m.cert, k, v)
    res["sampling_soundness"] = _sampling_soundness(model, batches[0], cfg_m, device) if batches else {}
    if onnx_dir and batches:
        scene = scene_to_device(batches[0], device)
        B = scene["traj"].shape[0]
        feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg_m)
        try:
            res["onnx_export"] = export_head_onnx(model, feat, cfg_m, onnx_dir)
        except Exception as exc:  # ONNX export needs a torch build with the exporter (pip install onnx)
            res["onnx_export"] = dict(error=repr(exc))
            log.warning(f"ONNX export failed: {exc!r}")
    return res


@torch.no_grad()
def _sampling_soundness(model, batch, cfg: Config, device, n_samples: int = 64) -> Dict:
    """Falsification test: random inputs inside X0 must produce controls inside the certificate."""
    from .safety.verifier import input_box
    K = cfg.data.hist_steps
    scene = scene_to_device(batch, device)
    B = scene["traj"].shape[0]
    feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg)
    lo, hi = input_box(feat, cfg)
    bnd = model.bounds(feat, lo, hi)
    worst = 0.0
    violations = 0
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(n_samples):
        f = dict(feat)
        for k in ("ego", "agents", "map"):
            u = torch.rand(lo[k].shape, device=device, generator=g)
            f[k] = lo[k] + (hi[k] - lo[k]) * u
        with precise_matmul():
            out = model(f)
        ctrl = out["ctrl"].float()
        over = torch.maximum(bnd["ctrl_lo"] - ctrl, ctrl - bnd["ctrl_hi"]).clamp(min=0)
        worst = max(worst, float(over.max()))
        violations += int((over > 1e-4).any(-1).any(-1).any(-1).sum())
        s = out["mode_logits"].float()
        violations += int(((s < bnd["mode_lo"] - 1e-4) | (s > bnd["mode_hi"] + 1e-4)).any(-1).sum())
    return dict(samples=n_samples * B, violations=violations, worst_excess=worst)


def closed_loop(cfg: Config, ckpt: str, methods: Optional[List[str]] = None, split: Optional[str] = None,
                out_dir: Optional[str] = None, tag: str = "") -> Dict:
    log = get_logger("closed_loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_m, _ = load_policy(ckpt, device)
    # Runtime settings (sim/shield/hj/cert/conformal) come from the current config, the network from the checkpoint.
    for sec in ("sim", "shield", "hj", "cert", "conformal", "stl"):
        setattr(cfg_m, sec, getattr(cfg, sec))
    cfg_m.data.out_dir = cfg.data.out_dir
    methods = methods or cfg.sim.methods
    split = split or cfg.sim.split
    out_dir = out_dir or cfg.sim.out_dir
    os.makedirs(out_dir, exist_ok=True)
    hj = HJTable(cfg.hj.table_path, device) if os.path.exists(cfg.hj.table_path) else None
    if hj is None and any(METHOD_TABLE[m].get("hj") for m in methods):
        raise FileNotFoundError(f"HJ table {cfg.hj.table_path} missing: run `build-hj` first")
    conf = conformal.load(cfg.conformal.table_path) if os.path.exists(cfg.conformal.table_path) else None
    if conf is None:
        log.warning("No conformal table: prediction sets have zero radius (run `calibrate` first)")
    if hj is not None and abs(hj.params["w_bar"] - cfg.hj.w_bar) > 1e-6:
        raise ValueError("HJ table was built with a different w_bar than the current config")
    store = RawSceneStore(cfg_m, os.path.join(cfg.data.out_dir, split))
    n = min(len(store), cfg.sim.num_scenarios)
    sim = LogSim(cfg_m, model, hj, conf, device)
    records: List[Dict] = []
    bs = cfg.sim.batch_size
    for seed in cfg.sim.seeds:
        seed_everything(seed)
        for m in methods:
            for start in range(0, n, bs):
                idx = list(range(start, min(n, start + bs)))
                recs = sim.run(store.get_batch(idx), m, seed=seed)
                for r in recs:
                    r["scene"] = idx[r["scene"]]
                    r["log"] = store.logs[r["scene"]]
                records.extend(recs)
            log.info(f"seed {seed} {m}: done ({len(records)} records)")
    summary = summarize(records)
    tests = {ref: paired_tests(records, ref, methods) for ref in ("B1", "B7") if ref in methods}
    table = markdown_table(summary, methods)
    name = f"closed_loop_{split}{('_' + tag) if tag else ''}"
    dump_json(records, os.path.join(out_dir, f"{name}_records.json"))
    dump_json(dict(summary=summary, paired_tests=tests, latency=sim.latency.summary(), config=cfg.to_dict(),
                   checkpoint=ckpt, environment=environment_info()), os.path.join(out_dir, f"{name}_summary.json"))
    with open(os.path.join(out_dir, f"{name}_table.md"), "w") as f:
        f.write(table + "\n")
    _write_csv(records, os.path.join(out_dir, f"{name}_records.csv"))
    log.info("\n" + table)
    return dict(summary=summary, tests=tests, table=table)


def _write_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    keys = sorted({k for r in records for k in r})
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in records:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def mine_failures(cfg: Config, ckpt: str, method: str = "B1", out_path: str = "results/failures_train.json",
                  max_scenes: int = 5000) -> Dict:
    """Closed-loop rollouts on TRAINING scenes; scenes with a collision, deadlock, STL violation, or
    any emergency/fallback become counterexamples for CEG refinement."""
    cfg2 = copy.deepcopy(cfg)
    cfg2.sim.num_scenarios = max_scenes
    cfg2.sim.seeds = [0]
    res = closed_loop(cfg2, ckpt, methods=[method], split="train", out_dir=os.path.dirname(out_path) or ".",
                      tag=f"mining_{method}")
    with open(os.path.join(os.path.dirname(out_path) or ".", f"closed_loop_train_mining_{method}_records.json")) as f:
        recs = json.load(f)
    bad = sorted({r["scene"] for r in recs if r["collision"] or r["deadlock"] or r["stl_robustness"] < 0
                  or r.get("emergency_rate", 0) > 0})
    out = dict(split="train", method=method, indices=bad, n_scenes=len({r["scene"] for r in recs}),
               failure_rate=len(bad) / max(1, len({r["scene"] for r in recs})))
    dump_json(out, out_path)
    return out


@torch.no_grad()
def kinematic_fit(cfg: Config, split: str = "train", max_batches: int = 50) -> Dict:
    """Phase-2 check: do logged ego futures follow the kinematic bicycle used by policy, HJ and MPC?

    Rolls the ground-truth (a, kappa) controls (derived from the log) through the model and
    reports the position error against the logged positions. This bounds the error the
    action space itself introduces, independently of learning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = _loader(cfg, split, 64)
    K = cfg.data.hist_steps
    errs, fde = [], []
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        scene = scene_to_device(batch, device)
        B = scene["traj"].shape[0]
        feat = featurize(scene, K - 1, torch.zeros(B, dtype=torch.long, device=device), cfg, with_targets=True)
        tr = rollout(feat["ego_speed"], feat["gt_ctrl"], cfg)
        e = torch.linalg.norm(tr[..., :2] - feat["ego_fut"][..., :2], dim=-1)
        v = feat["ego_fut_valid"].float()
        errs.append(((e * v).sum(-1) / v.sum(-1).clamp(min=1)).cpu())
        fde.append(e[:, -1].cpu())
    e = torch.cat(errs)
    f = torch.cat(fde)
    return dict(ADE_mean=float(e.mean()), ADE_p95=float(e.quantile(0.95)), FDE_mean=float(f.mean()),
                FDE_p95=float(f.quantile(0.95)), n=int(e.numel()))
