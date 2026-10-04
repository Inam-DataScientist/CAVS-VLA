"""CTL checker against brute-force path enumeration on random small Kripke structures."""

import itertools

import numpy as np
import pytest

from safeland.logic.ctl import Kripke, check, parse_ctl


def _random_kripke(rng, n):
    succ = []
    for _ in range(n):
        k = int(rng.integers(1, 4))
        succ.append(sorted(set(int(x) for x in rng.integers(0, n, size=k))))
    atoms = {a: rng.random(n) < 0.5 for a in ("a", "b")}
    return succ, atoms


def _paths(succ, s, length):
    out = [[s]]
    for _ in range(length - 1):
        out = [p + [t] for p in out for t in succ[p[-1]]]
    return out


def _brute(op, succ, A, B):
    """Exact via prefixes of length n+1 (any longer prefix repeats a state)."""
    n = len(succ)
    L = n + 1
    res = np.zeros(n, dtype=bool)
    for s in range(n):
        ps = _paths(succ, s, L)
        if op == "EX":
            res[s] = any(A[t] for t in succ[s])
        elif op == "AX":
            res[s] = all(A[t] for t in succ[s])
        elif op in ("EU", "AU"):
            def sat(p):
                for x in p:
                    if B[x]:
                        return True
                    if not A[x]:
                        return False
                return False
            res[s] = any(map(sat, ps)) if op == "EU" else all(map(sat, ps))
        elif op == "EG":
            res[s] = any(all(A[x] for x in p) for p in ps)
        elif op == "AG":
            res[s] = all(all(A[x] for x in p) for p in ps)
        elif op == "EF":
            res[s] = any(any(A[x] for x in p) for p in ps)
        elif op == "AF":
            res[s] = all(any(A[x] for x in p) for p in ps)
    return res


CASES = {
    "EX": "EX a",
    "AX": "AX a",
    "EU": "E[a U b]",
    "AU": "A[a U b]",
    "EG": "EG a",
    "AG": "AG a",
    "EF": "EF a",
    "AF": "AF a",
}


@pytest.mark.parametrize("op", sorted(CASES))
def test_against_brute_force(op):
    rng = np.random.default_rng(hash(op) % 2**32)
    for _ in range(150):
        n = int(rng.integers(1, 6))
        succ, atoms = _random_kripke(rng, n)
        K = Kripke.from_successors(succ, atoms)
        got = check(parse_ctl(CASES[op]), K)
        want = _brute(op, succ, atoms["a"], atoms["b"])
        assert np.array_equal(got, want), (op, succ, atoms)


def test_boolean_structure_and_parser():
    succ = [[1], [1]]
    atoms = {"a": np.array([True, False]), "b": np.array([False, True])}
    K = Kripke.from_successors(succ, atoms)
    assert check(parse_ctl("a -> EX b"), K).all()
    assert not check(parse_ctl("AG a"), K)[0]
    assert check(parse_ctl("!(a & b) | false"), K).all()
    assert check(parse_ctl("E[ a U (b & !a) ]"), K).all()
    for bad in ("AG", "A[a U ]", "a &", "EX (a"):
        with pytest.raises(SyntaxError):
            parse_ctl(bad)


def test_kripke_validation():
    with pytest.raises(ValueError):
        Kripke.from_successors([[0], []], {"a": np.array([True, False])})
    with pytest.raises(KeyError):
        check(parse_ctl("zzz"), Kripke.from_successors([[0]], {"a": np.array([True])}))


def test_identities_on_random_structures():
    rng = np.random.default_rng(7)
    for _ in range(100):
        succ, atoms = _random_kripke(rng, int(rng.integers(1, 7)))
        K = Kripke.from_successors(succ, atoms)
        for lhs, rhs in itertools.combinations(["AG a", "!EF !a", "!E[true U !a]"], 2):
            assert np.array_equal(check(parse_ctl(lhs), K), check(parse_ctl(rhs), K))
