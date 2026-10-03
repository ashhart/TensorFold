"""Depth policies and window costs: the expected-rate rule finds the depth with the most kept tokens a millisecond."""

from __future__ import annotations

import random

import pytest

from tensorfold.cuda.drafting import ExpectedRate, Lookup, StaticDepth, WindowCosts, best_k, expected_tokens


def test_window_costs_are_monotone_extend_on_their_last_line_and_cross_ranks_as_ints():
    c = WindowCosts([10.0, 12.0, 11.0, 15.0], draft=2.5)
    assert c.table == [10.0, 12.0, 12.0, 15.0] and c.ms(0) == 10.0 and c.ms(6) == 21.0
    assert WindowCosts.decode(c.encode()).table == c.table and WindowCosts.decode(c.encode()).draft == 2.5
    assert WindowCosts.linear(30.0, 2.0, rows=3).table == [32.0, 34.0, 36.0]
    with pytest.raises(ValueError):
        WindowCosts([])


def test_expected_tokens():
    assert expected_tokens([0.5, 0.5]) == [1.0, 1.5, 1.75]


def ratio_best(qs, ms) -> int:
    """Brute force: the k with the most expected tokens per millisecond."""

    es = expected_tokens(qs)
    return max(range(1, len(qs) + 1), key=lambda k: (es[k] / ms(k + 1), -k))


@pytest.mark.parametrize("seed", range(40))
def test_the_rate_rule_at_its_own_rate_is_the_brute_force_best(seed):
    rng = random.Random(seed)
    qs = sorted((rng.uniform(0.05, 0.98) for _ in range(rng.randint(1, 10))), reverse=rng.random() < 0.5)
    costs = WindowCosts.linear(rng.uniform(5, 40), rng.uniform(0.2, 8), rows=16)
    rate, k = 0.01, 1
    for _ in range(50):  # rate <- the rate its own choice achieves (Dinkelbach)
        k = best_k(qs, costs.ms, rate)
        nxt = expected_tokens(qs)[k] / costs.ms(k + 1)
        if abs(nxt - rate) < 1e-12:
            break
        rate = nxt
    want = ratio_best(qs, costs.ms)
    es = expected_tokens(qs)
    assert es[k] / costs.ms(k + 1) == pytest.approx(es[want] / costs.ms(want + 1), rel=1e-9)


def test_expected_rate_settles_on_the_best_depth_for_a_lane():
    rng = random.Random(5)
    truth = [0.9, 0.85, 0.8, 0.3, 0.2, 0.1]
    costs = WindowCosts.linear(20.0, 3.0, rows=16, draft=1.0)
    policy = ExpectedRate(costs)
    policy.reset(0)
    picks = []
    for _ in range(400):
        k = policy.depth(0, truth, len(truth))
        kept = 0
        while kept < k and rng.random() < truth[kept]:
            kept += 1
        policy.record(0, k, kept)
        picks.append(k)
    best = ratio_best(truth, lambda r: costs.ms(r) + costs.draft)
    tail = picks[-100:]
    assert tail.count(best) >= 80, (best, sorted(set(tail)))


def test_expected_rate_learns_acceptance_without_confidences():
    costs = WindowCosts.linear(20.0, 1.0)
    good, bad = ExpectedRate(costs), ExpectedRate(costs)
    for p, keep in ((good, lambda k: k), (bad, lambda k: 0)):
        for _ in range(60):
            k = p.depth(0, None, 8)
            p.record(0, k, keep(k))
    assert good.depth(0, None, 8) == 8 and bad.depth(0, None, 8) == 1
    assert bad.depth(0, None, 0) == 0


def test_two_policies_fed_the_same_rounds_agree():
    a, b = (ExpectedRate(WindowCosts.linear(15.0, 2.0)) for _ in range(2))
    for p in (a, b):
        for i in range(30):
            k = p.depth(i % 3, [0.7, 0.6, 0.5, 0.4], 4)
            p.record(i % 3, k, i % (k + 1))
    assert a.state() == b.state()


def test_static_depth():
    s = StaticDepth(3)
    assert [s.depth(0, None, m) for m in (0, 2, 5)] == [0, 2, 3]


def test_lookup_proposes_what_followed_the_latest_match():
    d = Lookup(block=4, n=2)
    d.reset(0, [1, 2, 3, 4, 9, 1, 2, 7, 8, 1])
    d.observe(0, [2])
    assert d.propose([0], [2], [10], [4], [None]).tokens == [[7, 8, 1, 2]]
    d.reset(1, [5, 6])
    assert d.propose([1], [6], [1], [3], [None]).tokens == [[]]
