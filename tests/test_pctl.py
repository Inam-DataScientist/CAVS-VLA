import numpy as np
import pytest
import scipy.sparse as sp

from safeland.logic.ctl import Kripke
from safeland.logic.pctl import (
    evaluate,
    expected_steps,
    parse_pctl,
    prob_bounded_until,
    prob_until,
)


def _random_dtmc(rng, n):
    P = rng.random((n, n)) * (rng.random((n, n)) < 0.4)
    for i in range(n):
        if P[i].sum() == 0:
            P[i, rng.integers(n)] = 1.0
    P /= P.sum(axis=1, keepdims=True)
    return sp.csr_matrix(P)


def test_unbounded_until_matches_long_horizon_iteration():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(2, 9))
        P = _random_dtmc(rng, n)
        phi, psi = rng.random(n) < 0.7, rng.random(n) < 0.3
        exact = prob_until(P, phi, psi)
        approx = prob_bounded_until(P, phi, psi, 20_000)
        assert np.allclose(exact, approx, atol=1e-6)


def test_expected_steps_matches_iteration():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(2, 8))
        P = _random_dtmc(rng, n).toarray()
        P = 0.8 * P
        P[:, 0] += 0.2  # state 0 reachable from everywhere with prob. 1
        P[0] = 0
        P[0, 0] = 1
        target = np.zeros(n, bool)
        target[0] = True
        e = expected_steps(sp.csr_matrix(P), target)
        x = np.zeros(n)
        for _ in range(5000):
            x = np.where(target, 0.0, 1.0 + P @ x)
        assert np.allclose(e, x, atol=1e-6)


def test_expected_steps_infinite_when_not_almost_sure():
    P = sp.csr_matrix(np.array([[0.5, 0.5, 0.0], [0, 1, 0], [0, 0, 1.0]]))
    target = np.array([False, False, True])
    assert np.all(np.isinf(expected_steps(P, target)[:2]))


def test_hand_example_and_parser():
    # 0 -> 1 (0.5) | 2 (0.5); 1 absorbing "goal", 2 absorbing "fail"
    P = sp.csr_matrix(np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1.0]]))
    atoms = {"goal": np.array([0, 1, 0], bool), "fail": np.array([0, 0, 1], bool)}
    K = Kripke(P, atoms)
    assert evaluate("P=? [ F goal ]", P, K)[0] == pytest.approx(0.5)
    assert evaluate("P=? [ !fail U goal ]", P, K)[0] == pytest.approx(0.5)
    assert evaluate("P=? [ G !fail ]", P, K)[0] == pytest.approx(0.5)
    assert evaluate("P=? [ F<=0 goal ]", P, K)[0] == 0.0
    assert evaluate("P=? [ true U<=1 goal ]", P, K)[0] == pytest.approx(0.5)
    assert np.isinf(evaluate("R=? [ F goal ]", P, K)[0])
    assert evaluate("R=? [ F (goal | fail) ]", P, K)[0] == pytest.approx(1.0)
    for bad in ("P=? F goal", "R=? [ G goal ]", "P=? [ goal ]"):
        with pytest.raises(SyntaxError):
            parse_pctl(bad)
