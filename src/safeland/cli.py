"""Command line interface: ``python -m safeland <command>``.

    run        train every method on every seed, certify, aggregate, plot
    shield     synthesize shields, check the certificate, print the reserve map
    spec       print the LTL monitor automaton
    plot       redraw figures for an existing results directory
    summarize  rebuild summary tables from saved per-seed metrics
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .config import Config


def _cfg(args) -> Config:
    return Config.load(args.config, args.set)


def cmd_run(args) -> int:
    from .experiment import run_experiment
    from .plots import plot_all

    cfg = _cfg(args)
    run_experiment(cfg)
    from pathlib import Path

    for p in plot_all(Path(cfg.experiment.out_dir) / cfg.experiment.name):
        print("wrote", p)
    return 0


def cmd_shield(args) -> int:
    from .experiment import build_context, shield_report
    from .shield import battery_reserve_map

    cfg = _cfg(args)
    ctx = build_context(cfg)
    print(ctx.world.ascii_map())
    print(json.dumps(shield_report(ctx), indent=2))
    if "sure_shield" in ctx.shields:
        r = battery_reserve_map(ctx.product, ctx.shields["sure_shield"], cfg.world.start_alt)
        print(f"battery reserve map at z={cfg.world.start_alt} (rows: y high -> low; '-' = unsafe):")
        for y in reversed(range(r.shape[1])):
            print(f"{y} " + " ".join(" -" if np.isnan(v) else f"{int(v):2d}" for v in r[:, y]))
    return 0


def cmd_spec(args) -> int:
    from .logic.ltl import Monitor, parse_ltl
    from .world import World

    cfg = _cfg(args)
    world = World(cfg.world)
    print(Monitor(parse_ltl(cfg.spec.safety_ltl), world.alphabet).describe())
    return 0


def cmd_plot(args) -> int:
    from .plots import plot_all

    for p in plot_all(args.results):
        print("wrote", p)
    return 0


def cmd_summarize(args) -> int:
    from pathlib import Path

    from .experiment import summarize_dir

    summarize_dir(args.results)
    print((Path(args.results) / "summary.md").read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="safeland", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("run", cmd_run), ("shield", cmd_shield), ("spec", cmd_spec)]:
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="YAML config file")
        p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        p.set_defaults(fn=fn)
    for name, fn in [("plot", cmd_plot), ("summarize", cmd_summarize)]:
        p = sub.add_parser(name)
        p.add_argument("results", help="results/<experiment name> directory")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
