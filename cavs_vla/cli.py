"""Command line entry point: python -m cavs_vla <command> [--config configs/default.yaml] [--set key=value ...]"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import dump_json, load_config


def _cfg(args):
    return load_config(args.config, args.set)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cavs_vla", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None)
        sp.add_argument("--set", nargs="*", default=[], help="overrides like train.lr=1e-4 data.sources=[nuplan,waymo]")
        return sp

    sp = add("selftest", "end-to-end test on synthetic nuPlan-format logs")
    sp.add_argument("--workdir", default="/tmp/cavs_selftest")
    sp = add("make-synthetic", "write synthetic nuPlan .db logs + JSON map")
    sp.add_argument("--out", required=True)
    sp.add_argument("--logs", type=int, default=10)
    sp = add("build-data", "nuPlan/WOMD -> leak-free raw scenes")
    sp.add_argument("--overwrite", action="store_true")
    add("estimate-disturbance", "measure the plant's acceleration mismatch (recommend hj.w_bar)")
    add("build-hj", "solve + validate the HJ value table")
    sp = add("train", "train the policy (torchrun-compatible)")
    sp.add_argument("--run", default="main")
    sp = add("kinematic-check", "dynamics validation: GT controls through the bicycle model")
    sp.add_argument("--split", default="train")
    for name, help_ in (("calibrate", "conformal calibration on val, coverage on test"),
                        ("verify", "certificate width vs eps, sampling soundness, ONNX/VNNLIB export"),
                        ("eval-open", "Experiment A: open-loop metrics, CV/CTRV baselines, input ablations"),
                        ("eval-closed", "Experiments B/C: closed-loop B0-B7 matrix with CIs"),
                        ("mine-failures", "closed-loop counterexample mining on the TRAIN split"),
                        ("ceg", "counterexample-guided refinement"),
                        ("carla", "CARLA stress-test suite")):
        sp = add(name, help_)
        sp.add_argument("--ckpt", required=True)
        sp.add_argument("--split", default=None)
        sp.add_argument("--methods", default=None, help="comma list, e.g. B1,B7")
        sp.add_argument("--out", default=None)
        sp.add_argument("--failures", default=None)
        sp.add_argument("--run", default="ceg")
        sp.add_argument("--method", default=None, help="mine-failures: policy to roll out (default B1); "
                                                         "carla: method to run (default carla.method)")
        sp.add_argument("--measure-disturbance", action="store_true")
        sp.add_argument("--tag", default="")
    sp = add("legacy-probe", "measure future leakage in the locked BASELINE-01")
    sp.add_argument("--legacy-root", required=True)
    sp.add_argument("--split", default="val")
    sp.add_argument("--numpy-only", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        from .selftest import run
        return 0 if run(args.workdir) else 1
    if args.cmd == "make-synthetic":
        from .synthetic import make_synthetic_nuplan
        print(json.dumps(make_synthetic_nuplan(args.out, n_logs=args.logs), indent=1))
        return 0
    if args.cmd == "legacy-probe":
        from .legacy_probe import leak_report_numpy, legacy_rescore
        root = os.path.expanduser(args.legacy_root)
        if args.numpy_only:
            res = leak_report_numpy(os.path.join(root, "data/processed", f"{args.split}.npz"))
        else:
            res = legacy_rescore(root, args.split)
        print(json.dumps(res, indent=1))
        if os.path.isdir(os.path.join(root, "results")):
            dump_json(res, os.path.join(root, "results", f"leakage_probe_{args.split}.json"))
        return 0
    cfg = _cfg(args)
    if args.cmd == "build-data":
        from .data.build import build_dataset
        build_dataset(cfg, overwrite=args.overwrite)
    elif args.cmd == "estimate-disturbance":
        import torch
        from .dynamics import estimate_disturbance
        res = estimate_disturbance(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        res["recommendation"] = f"set hj.w_bar >= {res['w_bar_quantile']:.3f} (current {cfg.hj.w_bar})"
        print(json.dumps(res, indent=1))
    elif args.cmd == "build-hj":
        from .safety.hj import build_table
        os.makedirs(os.path.dirname(os.path.abspath(cfg.hj.table_path)), exist_ok=True)
        out = build_table(cfg.hj, cfg.hj.table_path)
        rep = dict(before=out["report_before"], after=out["report_after"], solve_seconds=out["solve_seconds"])
        dump_json(rep, cfg.hj.table_path.replace(".npz", "_validation.json"))
        print(json.dumps(rep, indent=1))
    elif args.cmd == "train":
        from .train import train
        print(train(cfg, run_name=args.run))
    elif args.cmd == "kinematic-check":
        from .evaluate import kinematic_fit
        print(json.dumps(kinematic_fit(cfg, args.split), indent=1))
    elif args.cmd == "calibrate":
        from .evaluate import calibrate_conformal
        print(json.dumps(calibrate_conformal(cfg, args.ckpt), indent=1))
    elif args.cmd == "verify":
        from .evaluate import certificate_sweep
        res = certificate_sweep(cfg, args.ckpt, split=args.split or "test", onnx_dir=args.out or "artifacts/onnx")
        print(json.dumps(res, indent=1, default=str))
    elif args.cmd == "eval-open":
        from .evaluate import open_loop
        open_loop(cfg, args.ckpt, split=args.split or "test", out_dir=args.out or "results/open_loop")
    elif args.cmd == "eval-closed":
        from .evaluate import closed_loop
        methods = args.methods.split(",") if args.methods else None
        closed_loop(cfg, args.ckpt, methods=methods, split=args.split, out_dir=args.out, tag=args.tag)
    elif args.cmd == "mine-failures":
        from .evaluate import mine_failures
        print(json.dumps(mine_failures(cfg, args.ckpt, method=args.method or "B1",
                                       out_path=args.out or "results/failures_train.json"), indent=1))
    elif args.cmd == "ceg":
        import torch
        from .safety.hj import HJTable
        from .train import ceg_refine
        if not args.failures:
            raise SystemExit("--failures is required (output of mine-failures)")
        hj = HJTable(cfg.hj.table_path, torch.device("cuda" if torch.cuda.is_available() else "cpu")) \
            if os.path.exists(cfg.hj.table_path) else None
        print(ceg_refine(cfg, args.ckpt, args.failures, args.run, hj_table=hj))
    elif args.cmd == "carla":
        import torch
        from .safety import conformal
        from .safety.hj import HJTable
        from .sim.carla_env import CarlaBridge, run_carla_suite
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hj = HJTable(cfg.hj.table_path, dev) if os.path.exists(cfg.hj.table_path) else None
        conf = conformal.load(cfg.conformal.table_path) if os.path.exists(cfg.conformal.table_path) else None
        if args.measure_disturbance:
            from .train import load_policy
            model, cfg_m, _ = load_policy(args.ckpt, dev)
            bridge = CarlaBridge(cfg, model, hj, conf, dev)
            try:
                print(json.dumps(bridge.measure_disturbance(), indent=1))
            finally:
                bridge.close()
        else:
            if args.method:
                cfg.carla.method = args.method
            run_carla_suite(cfg, args.ckpt, hj, conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
