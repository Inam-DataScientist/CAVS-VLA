"""LTL tests, including the progression theorem checked on random lasso words:

    w, 0 |= φ   <=>   w, 1 |= prog(φ, w[0])
"""

import numpy as np
import pytest

from safeland.logic.ltl import (
    FALSE,
    TRUE,
    Always,
    And,
    Ap,
    BAlways,
    BEventually,
    Bot,
    Eventually,
    Monitor,
    Next,
    NotAp,
    Or,
    Release,
    Top,
    Until,
    b_always,
    b_eventually,
    conj,
    disj,
    negate,
    parse_ltl,
    progress,
    progress_trace,
)

ATOMS = ("a", "b")


# ------------------------------------------------------------ reference semantics on lassos
def holds(f, word, loop, i):
    n = len(word)

    def succ(j):
        return j + 1 if j + 1 < n else loop

    def path(j):
        seen, out = set(), []
        while j not in seen:
            seen.add(j)
            out.append(j)
            j = succ(j)
        return out

    if isinstance(f, Top):
        return True
    if isinstance(f, Bot):
        return False
    if isinstance(f, Ap):
        return f.name in word[i]
    if isinstance(f, NotAp):
        return f.name not in word[i]
    if isinstance(f, And):
        return all(holds(g, word, loop, i) for g in f.args)
    if isinstance(f, Or):
        return any(holds(g, word, loop, i) for g in f.args)
    if isinstance(f, Next):
        return holds(f.arg, word, loop, succ(i))
    if isinstance(f, Always):
        return all(holds(f.arg, word, loop, j) for j in path(i))
    if isinstance(f, Eventually):
        return any(holds(f.arg, word, loop, j) for j in path(i))
    if isinstance(f, Until):
        for j in path(i):
            if holds(f.right, word, loop, j):
                return True
            if not holds(f.left, word, loop, j):
                return False
        return False
    if isinstance(f, Release):
        for j in path(i):
            if not holds(f.right, word, loop, j):
                return False
            if holds(f.left, word, loop, j):
                return True
        return True
    if isinstance(f, (BEventually, BAlways)):
        js, j = [], i
        for _ in range(f.bound + 1):
            js.append(j)
            j = succ(j)
        vals = [holds(f.arg, word, loop, j) for j in js]
        return any(vals) if isinstance(f, BEventually) else all(vals)
    raise TypeError(f)


def random_formula(rng, depth):
    if depth == 0 or rng.random() < 0.2:
        a = ATOMS[rng.integers(len(ATOMS))]
        return Ap(a) if rng.random() < 0.5 else NotAp(a)
    op = rng.integers(10)
    sub = lambda: random_formula(rng, depth - 1)  # noqa: E731
    if op == 0:
        return conj(sub(), sub())
    if op == 1:
        return disj(sub(), sub())
    if op == 2:
        return Next(sub())
    if op == 3:
        return Always(sub())
    if op == 4:
        return Eventually(sub())
    if op == 5:
        return Until(sub(), sub())
    if op == 6:
        return Release(sub(), sub())
    if op == 7:
        return b_eventually(int(rng.integers(0, 4)), sub())
    if op == 8:
        return b_always(int(rng.integers(0, 4)), sub())
    return negate(sub())


def random_lasso(rng):
    n = int(rng.integers(1, 7))
    word = [frozenset(a for a in ATOMS if rng.random() < 0.5) for _ in range(n)]
    return word, int(rng.integers(0, n))


def suffix_from_one(word, loop):
    """The lasso word read from position 1 on, as a lasso of its own."""
    n = len(word)
    if n == 1:
        return word, 0
    if loop >= 1:
        return word[1:], loop - 1
    return word[1:] + word[:1], 0  # ... w[n-1], w[0], then back to w[1]


def test_progression_theorem_on_random_lassos():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        f = random_formula(rng, 4)
        word, loop = random_lasso(rng)
        suf, sl = suffix_from_one(word, loop)
        assert holds(f, word, loop, 0) == holds(progress(f, word[0]), suf, sl, 0), (f, word, loop)


def test_negation_is_semantic_complement():
    rng = np.random.default_rng(1)
    for _ in range(1500):
        f = random_formula(rng, 3)
        word, loop = random_lasso(rng)
        assert holds(f, word, loop, 0) != holds(negate(f), word, loop, 0)


def test_parser_and_simplification():
    f = parse_ltl("G(!a & !b) & G(a -> F[<=2] b)")
    assert isinstance(f, And)
    assert parse_ltl("a & !a") == FALSE
    assert parse_ltl("a | !a") == TRUE
    assert parse_ltl("F[<=0] a") == Ap("a")
    assert parse_ltl("G G a") == parse_ltl("G a")
    assert parse_ltl("true U b") == parse_ltl("F b")
    assert parse_ltl("a -> b") == parse_ltl("!a | b")
    assert conj(b_eventually(3, Ap("a")), b_eventually(5, Ap("a"))) == b_eventually(3, Ap("a"))
    with pytest.raises(SyntaxError):
        parse_ltl("G(a & )")
    with pytest.raises(SyntaxError):
        parse_ltl("a b")


def test_deadline_semantics():
    f = parse_ltl("F[<=2] p")
    assert progress_trace(f, [frozenset(), frozenset(), frozenset()]) == FALSE
    assert progress_trace(f, [frozenset(), frozenset(), frozenset({"p"})]) == TRUE
    g = parse_ltl("G !x")
    assert progress_trace(g, [frozenset(), frozenset({"x"})]) == FALSE


def test_monitor_is_small_and_detects_violation():
    alphabet = [frozenset(), frozenset({"low"}), frozenset({"landed"}), frozenset({"crash"})]
    m = Monitor(parse_ltl("G !crash & G(low -> F[<=5] landed)"), alphabet)
    assert m.n_states <= 5 + 4
    assert m.is_bad.sum() == 1
    crash = alphabet.index(frozenset({"crash"}))
    assert m.is_bad[m.step(0, crash)]
    with pytest.raises(ValueError):
        Monitor(parse_ltl("G !typo"), alphabet)
