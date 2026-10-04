"""End-to-end experiment: build -> shield -> train -> certify -> aggregate.

Directory layout (one run)::

    <out_dir>/<name>/
        config.yaml              exact configuration (with digest)
        environment.json         python / numpy / scipy versions, git commit
        shields.json             synthesis statistics + certificate per shield
        reserve_map.npy          sure-shield battery reserve map at start altitude
        <method>/seed_<k>/
            metrics.json         all scalar results of this run
            train_curve.csv      per-episode return / violation / landing / steps
            policy.npy           deployed deterministic policy (product states)
        summary.json             mean, sd, 95% t-CI and raw values per method
        summary.md               main results table
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

from .config import METHODS, Config
from .logic.ltl import Monitor, parse_ltl
from .product import Product
from .rl import greedy_policy, train_q_learning
from .shield import Shield, battery_reserve_map, build_shield, check_certificate
from .verification import (
    certify,
    clopper_pearson,
    counterexample,
    random_policy_baseline,
    smc_safe_landing,
)
from .world import World

SAFE_LANDING = "P=? [ !spec_violated U landed ]"
# 30+ SMC-vs-exact comparisons are made per experiment, so the agreement flag uses a
# 99% Clopper-Pearson interval (at 95% one or two chance disagreements are expected).
SMC_AGREEMENT_ALPHA = 0.01


def smc_agrees(m: dict) -> bool:
    exact = m["pctl"].get(SAFE_LANDING, {}).get("mean")
    if exact is None:
        return False
    lo, hi = clopper_pearson(m["smc_safe_landing"]["successes"], m["smc_safe_landing"]["samples"],
                             SMC_AGREEMENT_ALPHA)
    return bool(lo - 1e-12 <= exact <= hi + 1e-12)


@dataclass(eq=False)
class Context:
    cfg: Config
    world: World
    monitor: Monitor
    product: Product
    shields: dict[str, Shield]


def build_context(cfg: Config) -> Context:
    world = World(cfg.world)
    monitor = Monitor(parse_ltl(cfg.spec.safety_ltl), world.alphabet, cfg.spec.max_monitor_states)
    product = Product(world, monitor)
    shields = {
        m: build_shield(product, m, cfg.shield.prob_lambda, cfg.shield.value_tol)
        for m in cfg.experiment.methods
    }
    return Context(cfg, world, monitor, product, shields)


def shield_report(ctx: Context) -> dict:
    init = ctx.product.initial_states()
    live = ~ctx.product.terminal
    out = {}
    for m, sh in ctx.shields.items():
        cert = check_certificate(ctx.product, sh)
        out[m] = {
            "kind": sh.kind,
            "iterations": sh.iterations,
            "synthesis_time_s": sh.synthesis_time,
            "start_states": int(init.size),
            "start_fraction_guaranteed": float(sh.winning[init].mean()),
            "min_start_safety_probability": float(sh.safety_value[init].min()),
            "mean_allowed_actions": float(sh.allowed[live & sh.winning].sum(axis=1).mean()),
            "certificate": asdict(cert),
        }
    return out


def run_single(ctx: Context, method: str, seed: int, out: Path | None = None) -> dict:
    cfg, product = ctx.cfg, ctx.product
    shield = ctx.shields[method]
    rng = np.random.default_rng(seed)
    tr = train_q_learning(product, shield, cfg.rl, ctx.world.start_states, rng)
    policy = greedy_policy(tr.Q, shield)
    t0 = time.perf_counter()
    cert = certify(product, policy, cfg.spec.ctl, cfg.spec.pctl)
    verify_time = time.perf_counter() - t0
    smc = smc_safe_landing(product, policy, cfg.experiment.smc_samples, np.random.default_rng(10_000 + seed))
    tail = max(1, cfg.rl.episodes // 10)
    in_w = tr.start_in_winning
    metrics = {
        "method": method,
        "seed": seed,
        "train_episodes": cfg.rl.episodes,
        "train_steps": int(tr.steps.sum()),
        "train_time_s": tr.train_time,
        "train_violations": int(tr.violated.sum()),
        "train_violation_rate": float(tr.violated.mean()),
        "train_violations_from_guaranteed_starts": int((tr.violated & in_w).sum()),
        "train_success_rate_last10pct": float(tr.landed[-tail:].mean()),
        "train_return_last10pct": float(tr.returns[-tail:].mean()),
        "verify_time_s": verify_time,
        "smc_safe_landing": asdict(smc),
        **cert,
    }
    metrics["smc_consistent_with_exact"] = smc_agrees(metrics)
    if not cert["ctl"].get("AG !spec_violated", {"all": True})["all"]:
        metrics["counterexample"] = counterexample(product, policy, product.initial_states())[:40]
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        np.save(out / "policy.npy", policy)
        curve = np.column_stack(
            [np.arange(cfg.rl.episodes), tr.returns, tr.violated, tr.landed, tr.steps, in_w]
        )
        np.savetxt(
            out / "train_curve.csv",
            curve,
            delimiter=",",
            header="episode,return,violated,landed,steps,start_guaranteed",
            comments="",
            fmt=["%d", "%.4f", "%d", "%d", "%d", "%d"],
        )
    return metrics


# ---------------------------------------------------------------- aggregation
def headline(m: dict) -> dict[str, float]:
    """Scalar metrics reported in the main table."""
    p = m["pctl"]
    return {
        "train_violations": m["train_violations"],
        "train_violation_rate": m["train_violation_rate"],
        "train_success_rate_last10pct": m["train_success_rate_last10pct"],
        "P_safe_landing": p.get(SAFE_LANDING, {}).get("mean", np.nan),
        "P_safe_landing_min_start": p.get(SAFE_LANDING, {}).get("min", np.nan),
        "P_spec_violation": p.get("P=? [ F spec_violated ]", {}).get("mean", np.nan),
        "expected_steps": p.get("R=? [ F terminal ]", {}).get("mean_finite", np.nan),
        "AG_safe_fraction": m["ctl"].get("AG !spec_violated", {}).get(
            "fraction_initial_satisfied", np.nan
        ),
        "smc_safe_landing": m["smc_safe_landing"]["estimate"],
        "smc_consistent": float(smc_agrees(m)),
        "train_time_s": m["train_time_s"],
    }


def random_baselines(ctx: Context) -> dict:
    """Uniform-random policy restricted to each method's shield (no learning)."""
    return {m: random_policy_baseline(ctx.product, sh.allowed) for m, sh in ctx.shields.items()}


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95": half, "n": n, "raw": arr.tolist()}


def aggregate(results: dict[str, list[dict]]) -> dict:
    out = {}
    for method, runs in results.items():
        rows = [headline(r) for r in runs]
        out[method] = {k: summarize([row[k] for row in rows]) for k in rows[0]}
    return out


def summary_table(summary: dict, baselines: dict | None = None) -> str:
    cols = [
        ("train_violations", "Train violations", "{:.1f}"),
        ("train_success_rate_last10pct", "Train success (last 10%)", "{:.3f}"),
        ("P_safe_landing", "P(safe landing)", "{:.4f}"),
        ("P_safe_landing_min_start", "min-start P(safe landing)", "{:.4f}"),
        ("P_spec_violation", "P(spec violated)", "{:.2e}"),
        ("AG_safe_fraction", "AG safe (frac. starts)", "{:.3f}"),
        ("expected_steps", "E[steps]", "{:.2f}"),
        ("smc_consistent", "SMC agrees", "{:.2f}"),
    ]
    head = "| Method | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    for method, s in summary.items():
        cells = []
        for key, _, fmt in cols:
            v = s[key]
            cells.append(f"{fmt.format(v['mean'])} ± {fmt.format(v['ci95'])}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    n = next(iter(summary.values()))["P_safe_landing"]["n"]
    lines.append(f"\nmean ± 95% t-CI over {n} seeds; probabilities are exact (PCTL), "
                 "averaged over the uniform start distribution.")
    if baselines:
        lines += [
            "",
            "Shield vs. learning (exact): learned greedy policy vs. uniform-random actions "
            "inside the same shield",
            "",
            "| Shield | learned P(safe landing) | random P(safe landing) | "
            "learned E[steps] | random E[steps] |",
            "|---|---|---|---|---|",
        ]
        for method, b in baselines.items():
            if method not in summary:
                continue
            s = summary[method]
            lines.append(
                f"| {method} | {s['P_safe_landing']['mean']:.4f} | {b['P_safe_landing']:.4f} | "
                f"{s['expected_steps']['mean']:.2f} | {b['expected_steps']:.2f} |"
            )
    return "\n".join(lines)


def summarize_dir(root: str | Path) -> dict:
    """Rebuild summary.json / summary.md from saved per-seed metrics (no retraining)."""
    root = Path(root)
    results: dict[str, list[dict]] = {}
    for f in sorted(root.glob("*/seed_*/metrics.json"), key=lambda p: (p.parent.parent.name, int(p.parent.name[5:]))):
        m = json.loads(f.read_text())
        results.setdefault(m["method"], []).append(m)
    if not results:
        raise FileNotFoundError(f"no metrics under {root}")
    order = {m: i for i, m in enumerate(METHODS)}
    results = dict(sorted(results.items(), key=lambda kv: order.get(kv[0], 99)))
    summary = aggregate(results)
    cfg = Config.load(root / "config.yaml") if (root / "config.yaml").exists() else None
    baselines = None
    if cfg is not None:
        baselines = random_baselines(build_context(cfg))
        (root / "baselines.json").write_text(json.dumps(baselines, indent=2))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    (root / "summary.md").write_text(summary_table(summary, baselines) + "\n")
    return summary


def _environment() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - git is optional
        sha = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "git_commit": sha,
    }


def run_experiment(cfg: Config, verbose: bool = True) -> dict:
    root = Path(cfg.experiment.out_dir) / cfg.experiment.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(f"# digest: {cfg.digest()}\n" + cfg.to_yaml())
    (root / "environment.json").write_text(json.dumps(_environment(), indent=2))
    ctx = build_context(cfg)
    rep = shield_report(ctx)
    (root / "shields.json").write_text(json.dumps(rep, indent=2))
    baselines = random_baselines(ctx)
    (root / "baselines.json").write_text(json.dumps(baselines, indent=2))
    if "sure_shield" in ctx.shields:
        np.save(
            root / "reserve_map.npy",
            battery_reserve_map(ctx.product, ctx.shields["sure_shield"], cfg.world.start_alt),
        )
    for m, r in rep.items():
        if not r["certificate"]["ok"] and m == "sure_shield":
            raise RuntimeError(f"sure shield certificate failed: {r['certificate']}")
    if verbose:
        print(ctx.world.ascii_map())
        print(f"monitor states: {ctx.monitor.n_states}, product states: {ctx.product.n_states}")
        for m, r in rep.items():
            print(f"[{m}] guaranteed starts={r['start_fraction_guaranteed']:.3f} "
                  f"allowed/state={r['mean_allowed_actions']:.2f} "
                  f"certificate={'OK' if r['certificate']['ok'] else 'n/a'}")
    results: dict[str, list[dict]] = {}
    for method in cfg.experiment.methods:
        for seed in cfg.experiment.seeds:
            m = run_single(ctx, method, seed, root / method / f"seed_{seed}")
            results.setdefault(method, []).append(m)
            if verbose:
                h = headline(m)
                print(f"  {method:12s} seed={seed:<3d} train_viol={h['train_violations']:6d} "
                      f"P(safe landing)={h['P_safe_landing']:.4f} "
                      f"P(viol)={h['P_spec_violation']:.2e} AG-safe={h['AG_safe_fraction']:.3f} "
                      f"SMC={h['smc_safe_landing']:.3f} ({h['train_time_s']:.1f}s)")
    summary = aggregate(results)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    table = summary_table(summary, baselines)
    (root / "summary.md").write_text(table + "\n")
    if verbose:
        print(table)
    return summary
