import numpy as np

from _helpers import small_ctx
from safeland.shield import Shield, check_certificate, synthesize_prob, synthesize_sure


def _adversary_attractor(product):
    """Independent reference: states from which the gust adversary can force a bad state."""
    n, A = product.n_states, product.n_actions
    attr = product.bad.copy()
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if attr[v]:
                continue
            forced = True
            for a in range(A):
                hits = any(
                    product.prob[v, a, k] > 0 and attr[product.next[v, a, k]] for k in range(2)
                )
                if not hits:
                    forced = False
                    break
            if forced:
                attr[v] = True
                changed = True
    return attr


def test_certificate_holds():
    ctx = small_ctx()
    cert = check_certificate(ctx.product, ctx.shields["sure_shield"])
    assert cert.ok, cert


def test_winning_region_is_maximal():
    ctx = small_ctx()
    W = ctx.shields["sure_shield"].winning
    assert np.array_equal(W, ~_adversary_attractor(ctx.product))


def test_shielded_random_play_never_violates():
    ctx = small_ctx()
    p, sh = ctx.product, ctx.shields["sure_shield"]
    rng = np.random.default_rng(0)
    starts = [v for v in p.initial_states() if sh.winning[v]]
    assert starts
    for _ in range(3000):
        v = int(rng.choice(starts))
        while not p.terminal[v]:
            a = int(rng.choice(np.flatnonzero(sh.allowed[v])))
            k = int(rng.integers(2)) if p.prob[v, a, 1] > 0 else 0  # adversarial-ish gusts
            v = int(p.next[v, a, k])
            assert not p.bad[v]


def test_broken_shield_is_caught_by_certificate():
    ctx = small_ctx()
    p, sh = ctx.product, ctx.shields["sure_shield"]
    allowed = sh.allowed.copy()
    # find a winning state with a disallowed action and allow it
    v, a = np.argwhere(sh.winning[:, None] & ~allowed)[0]
    allowed[v, a] = True
    broken = Shield("sure", allowed, sh.winning, sh.safety_value, 0, 0.0)
    assert not check_certificate(p, broken).ok


def test_rows_never_empty_and_relations():
    ctx = small_ctx()
    p = ctx.product
    sure = synthesize_sure(p)
    prob1 = synthesize_prob(p, 1.0)
    assert sure.allowed.any(axis=1).all() and prob1.allowed.any(axis=1).all()
    # sure-safe states have safety probability one
    assert np.all(sure.safety_value[sure.winning] > 1 - 1e-9)
    # lambda = 1 keeps exactly the value-maximising actions
    Q = np.einsum("nak,nak->na", p.prob, prob1.safety_value[p.next])
    qmax = np.broadcast_to(Q.max(axis=1, keepdims=True), Q.shape)
    assert np.all(Q[prob1.allowed] >= qmax[prob1.allowed] - 1e-9)
