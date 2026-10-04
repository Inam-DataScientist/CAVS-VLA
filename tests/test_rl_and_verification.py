import json

import numpy as np

from _helpers import small_cfg, small_ctx
from safeland.experiment import run_experiment
from safeland.logic.ctl import check, parse_ctl
from safeland.plots import plot_all
from safeland.rl import greedy_policy, train_q_learning
from safeland.verification import (
    certify,
    closed_loop,
    counterexample,
    smc_safe_landing,
)


def _train(method, seed=0, episodes=2000):
    ctx = small_ctx()
    cfg = small_cfg(rl={"episodes": episodes}).rl
    tr = train_q_learning(ctx.product, ctx.shields[method], cfg, ctx.world.start_states,
                          np.random.default_rng(seed))
    return ctx, tr


def test_training_is_deterministic():
    _, a = _train("sure_shield", seed=3, episodes=500)
    _, b = _train("sure_shield", seed=3, episodes=500)
    assert np.array_equal(a.Q, b.Q) and np.array_equal(a.violated, b.violated)


def test_sure_shield_never_violates_during_training_from_guaranteed_starts():
    _, tr = _train("sure_shield")
    assert not np.any(tr.violated & tr.start_in_winning)
    _, tr_u = _train("unshielded")
    assert tr_u.violated.sum() > 0  # the baseline does explore into violations


def test_deployed_policy_respects_shield_and_ag_holds_on_winning_starts():
    ctx, tr = _train("sure_shield")
    sh = ctx.shields["sure_shield"]
    pol = greedy_policy(tr.Q, sh)
    assert sh.allowed[np.arange(len(pol)), pol].all()
    init = ctx.product.initial_states()
    P, K, idx, remap = closed_loop(ctx.product, pol, init)
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
    ag = check(parse_ctl("AG !spec_violated"), K)[remap[init]]
    # theorem, cross-checked by an independent CTL model checker
    assert ag[sh.winning[init]].all()


def test_smc_agrees_with_exact_pctl():
    for method in ("unshielded", "sure_shield"):
        ctx, tr = _train(method)
        pol = greedy_policy(tr.Q, ctx.shields[method])
        cert = certify(ctx.product, pol, (), ("P=? [ !spec_violated U landed ]",))
        exact = cert["pctl"]["P=? [ !spec_violated U landed ]"]["mean"]
        smc = smc_safe_landing(ctx.product, pol, 3000, np.random.default_rng(5))
        assert smc.ci_low - 1e-9 <= exact <= smc.ci_high + 1e-9, (method, exact, smc)


def test_counterexample_for_random_unshielded_policy():
    ctx = small_ctx()
    rng = np.random.default_rng(0)
    pol = rng.integers(0, ctx.product.n_actions, size=ctx.product.n_states)
    trace = counterexample(ctx.product, pol, ctx.product.initial_states())
    assert trace and "monitor" in trace[-1] and "false" in trace[-1]


def test_end_to_end_experiment_writes_artifacts(tmp_path):
    cfg = small_cfg(rl={"episodes": 400},
                    experiment={"out_dir": str(tmp_path), "seeds": [0, 1], "smc_samples": 100})
    summary = run_experiment(cfg, verbose=False)
    root = tmp_path / "test"
    for f in ("config.yaml", "environment.json", "shields.json", "summary.json", "summary.md"):
        assert (root / f).exists(), f
    for m in cfg.experiment.methods:
        for s in (0, 1):
            d = root / m / f"seed_{s}"
            assert (d / "metrics.json").exists() and (d / "train_curve.csv").exists()
        assert summary[m]["train_violations"]["n"] == 2
    for s in (0, 1):
        m = json.loads((root / "sure_shield" / f"seed_{s}" / "metrics.json").read_text())
        assert m["train_violations_from_guaranteed_starts"] == 0
    outs = plot_all(root)
    assert all(p.exists() for p in outs)
