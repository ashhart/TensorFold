"""A family with ``plain_guard`` drafts only where drafts pay for their rows: plain rounds otherwise, with probes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_family import FamilyRounds

COSTS = {1: 8.0, 2: 11.0, 3: 11.6, 4: 12.1, 5: 13.0}         # ms by verify rows (the M3 Ultra's Qwen3.6 curve)


def policy(rates, *, guard=True, measured=None):
    engine = FamilyRounds()
    engine.family_costs = dict(COSTS)
    engine.mtp_step_ms = 0.5
    engine.most_drafts = 4
    engine.plain_guard = guard
    engine._round_ms = {} if guard else dict(measured or {})
    engine._depth_state = {"s": {"p": list(rates), "rounds": 0, **({"ms": dict(measured or {})} if guard else {})}}
    return engine


def stream():
    return SimpleNamespace(stream_id="s", draft_room=128)


def test_drafts_that_land_are_verified():
    assert policy([0.9, 0.85, 0.8, 0.75])._depth(stream()) >= 2


def test_drafts_that_miss_give_plain_rounds_and_probes_that_back_off():
    engine = policy([0.1, 0.1, 0.1, 0.1])
    depths = [engine._depth(stream()) for _ in range(250)]
    assert [i for i, d in enumerate(depths) if d] == [7, 23, 55, 119, 247]      # after 8, 16, 32, 64, 128 plain rounds


def test_a_probe_that_shows_drafts_paying_resets_the_wait():
    engine = policy([0.1, 0.1, 0.1, 0.1])
    assert [engine._depth(stream()) for _ in range(24)].count(1) == 2
    engine._depth_state["s"]["p"] = [0.95, 0.9, 0.9, 0.9]
    assert engine._depth(stream()) >= 1
    engine._depth_state["s"]["p"] = [0.1, 0.1, 0.1, 0.1]
    depths = [engine._depth(stream()) for _ in range(8)]
    assert depths == [0] * 7 + [1]                            # the wait is eight plain rounds again


def test_a_depth_must_beat_plain_by_the_margin():
    engine = policy([0.9] * 4, measured={0: 26.0, 1: 48.0, 2: 70.0, 3: 92.0, 4: 114.0})   # 1.9 / 48 = 1.03x plain
    assert engine._depth(stream()) == 0
    engine._depth_state["s"]["ms"][1] = 44.0                                              # 1.9 / 44 = 1.12x plain
    assert engine._depth(stream()) == 1


def test_without_the_guard_every_round_verifies_a_draft():
    engine = policy([0.1, 0.1, 0.1, 0.1], guard=False)
    assert min(engine._depth(stream()) for _ in range(16)) == 1


def test_an_untimed_depth_costs_its_model_plus_the_least_extra_a_timed_row_took():
    engine = policy([0.5] * 4, measured={2: 11.6 + 1.0 + 3.0, 3: 12.1 + 1.5 + 2.0})
    s = stream()
    assert engine._round_cost(0, s) == pytest.approx(8.0 + 0.5)                # 2.0 ms over 4 rows is the least
    assert engine._round_cost(1, s) == pytest.approx(11.0 + 0.5 + 2 * 0.5)
    engine._observe_cost(0, 9.0, stream=s)
    assert engine._round_cost(0, s) == pytest.approx(9.0)


def test_rows_that_cost_far_more_than_their_model_keep_untimed_depths_plain():
    engine = policy([0.5] * 4, measured={0: 26.0})               # a sampled row's host work, 18 ms past the forward
    assert engine._round_cost(1, stream()) == pytest.approx(11.0 + 0.5 + 2 * 18.0)
    assert engine._depth(stream()) == 0


def test_each_stream_learns_its_own_round_times():
    engine = policy([0.8] * 4, measured={0: 26.0, 1: 48.0})     # a hot stream's rows cost their host work
    engine._depth_state["t"] = {"p": [0.9] * 4, "rounds": 0}
    assert engine._depth(stream()) == 0
    assert engine._depth(SimpleNamespace(stream_id="t", draft_room=128)) >= 2      # a greedy one starts from the model
    engine._observe_cost(2, 13.0, stream=SimpleNamespace(stream_id="t"))
    assert engine._depth_state["s"]["ms"] == {0: 26.0, 1: 48.0} and engine._round_ms == {}


def test_plain_round_times_are_learned_only_with_the_guard():
    engine = policy([0.5] * 4, guard=False)
    engine._observe_cost(0, 9.0, stream=stream())
    assert 0 not in engine._round_ms


def test_shared_rounds_draft_nothing_where_nothing_pays_and_still_probe(monkeypatch):
    from tensorfold.engine import allocate

    engine = policy([0.1] * 4)
    engine.node_probabilities = False
    engine.shared_costs = {}
    engine.batch_rows = 32
    engine._overhead_ms = {}
    engine._copy_proposal = lambda s: []
    monkeypatch.setattr(allocate, "allocate", lambda fixed, probs, *a: [0] * len(probs))
    budgets = [engine._draft_budgets([SimpleNamespace(stream_id="s", draft_room=128, finished=False, force=[])])[0]
               for _ in range(56)]
    assert [i for i, b in enumerate(budgets) if b] == [7, 23, 55]
