"""Tests for the multi-stream lane engine.

Pure bookkeeping is tested directly. The cache protocol is tested against
mlx_lm's real BatchKVCache and ArraysCache with tiny arrays (no model). The
round loop is tested end to end against the history-dependent fake target in
``tests/lane_fakes.py``, so any rollback or refeed mistake shows up as a
byte divergence from the fake's own serial decode.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorfold.engine.lane_engine import (
    LaneEngine,
    LaneStream,
    SuffixLookupProposer,
    apply_round,
    plan_windows,
    verify_window,
)
from tests.lane_fakes import FakeEngine, PatternProposer, fake_serial


# --------------------------------------------------------------------------
# pure bookkeeping
# --------------------------------------------------------------------------


def test_plan_windows_never_trims_pending_and_respects_row_budget() -> None:
    windows, width = plan_windows(
        [[1], [2, 3, 4], [5]],
        [[10, 11, 12, 13, 14], [], [20]],
        max_rows=9,
    )
    # 9 rows / 3 streams = 3 per row; pending of 3 fills its row untouched.
    assert windows == [[1, 10, 11], [2, 3, 4], [5, 20]]
    assert width == 3


def test_plan_windows_lets_long_pending_exceed_the_budget() -> None:
    windows, width = plan_windows([[1, 2, 3, 4, 5]], [[9, 9]], max_rows=2)
    assert windows == [[1, 2, 3, 4, 5]]
    assert width == 5


def test_plan_windows_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="align"):
        plan_windows([[1]], [], max_rows=8)
    with pytest.raises(ValueError, match="pending"):
        plan_windows([[]], [[1]], max_rows=8)
    assert plan_windows([], [], max_rows=8) == ([], 0)


def test_verify_window_accepts_prefix_and_returns_bonus() -> None:
    # window = [anchor, d1, d2, d3]; preds after each position
    accepted, bonus = verify_window([7, 8, 9, 10], 1, [8, 9, 42, 99])
    assert (accepted, bonus) == (2, 42)
    accepted, bonus = verify_window([7, 8, 9, 10], 1, [8, 9, 10, 11])
    assert (accepted, bonus) == (3, 11)
    accepted, bonus = verify_window([7], 1, [5])
    assert (accepted, bonus) == (0, 5)


def test_verify_window_judges_drafts_from_the_last_pending_position() -> None:
    # three refed pending tokens, then two drafts; predictions at the pending
    # positions are ignored, the judge of draft 0 is preds[2].
    accepted, bonus = verify_window([1, 2, 3, 4, 5], 3, [0, 0, 4, 6, 0])
    assert (accepted, bonus) == (1, 6)


def test_verify_window_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        verify_window([1, 2], 0, [1, 2])
    with pytest.raises(ValueError):
        verify_window([1, 2], 1, [1])


def _stream(**kwargs: Any) -> LaneStream:
    base: dict[str, Any] = {"stream_id": "s", "prompt_ids": [1, 2, 3], "max_new_tokens": 100}
    base.update(kwargs)
    return LaneStream(**base)


def test_apply_round_full_accept_advances_cache_and_resets_pending() -> None:
    stream = _stream(emitted=[9], pending=[9], cache_len=3)
    landed, rollback = apply_round(stream, [9, 10, 11], [10, 11, 12])
    assert landed == [10, 11, 12]
    assert rollback == 0
    assert stream.cache_len == 6
    assert stream.pending == [12]
    assert stream.emitted == [9, 10, 11, 12]
    assert (stream.full_rounds, stream.partial_rounds) == (1, 0)
    assert stream.cache_len + len(stream.pending) == len(stream.prompt_ids) + len(stream.emitted)


def test_apply_round_partial_accept_rolls_back_whole_window_and_grows_pending() -> None:
    stream = _stream(emitted=[9], pending=[9], cache_len=3)
    landed, rollback = apply_round(stream, [9, 10, 11, 12], [10, 50, 0, 0])
    assert landed == [10, 50]
    assert rollback == 4
    assert stream.cache_len == 3
    assert stream.pending == [9, 10, 50]
    assert stream.refeed_tokens == 3
    assert stream.cache_len + len(stream.pending) == len(stream.prompt_ids) + len(stream.emitted)


def test_apply_round_with_no_drafts_absorbs_pending() -> None:
    stream = _stream(emitted=[9, 10, 50], pending=[9, 10, 50], cache_len=3)
    landed, rollback = apply_round(stream, [9, 10, 50], [0, 0, 77])
    assert landed == [77]
    assert rollback == 0
    assert stream.cache_len == 6
    assert stream.pending == [77]


def test_apply_round_stops_at_eos_inside_accepted_drafts() -> None:
    stream = _stream(emitted=[9], pending=[9], cache_len=3, eos_ids=frozenset({11}))
    landed, _ = apply_round(stream, [9, 10, 11, 12], [10, 11, 12, 13])
    assert landed == [10, 11]
    assert stream.finished and stream.finish_reason == "stop"


def test_apply_round_respects_max_new_tokens() -> None:
    stream = _stream(emitted=[9], pending=[9], cache_len=3, max_new_tokens=3)
    landed, _ = apply_round(stream, [9, 10, 11, 12], [10, 11, 12, 13])
    assert landed == [10, 11]
    assert stream.finished and stream.finish_reason == "length"


def test_apply_round_reports_to_the_proposer() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.seen: list[tuple[int, int]] = []

        def observe(self, proposed: int, accepted: int) -> None:
            self.seen.append((proposed, accepted))

    recorder = Recorder()
    stream = _stream(emitted=[9], pending=[9], cache_len=3, proposer=recorder)
    apply_round(stream, [9, 10, 11], [10, 0, 0])
    assert recorder.seen == [(2, 1)]


# --------------------------------------------------------------------------
# suffix lookup proposer
# --------------------------------------------------------------------------


def test_suffix_lookup_proposes_continuation_of_longest_evidence() -> None:
    proposer = SuffixLookupProposer(ngram=2, min_match=3)
    # ... 5 6 7 8 9 ... 5 6 7  -> the suffix (5 6 7) matched 3 deep; propose 8 9
    context = [1, 5, 6, 7, 8, 9, 2, 3, 5, 6, 7]
    assert proposer.propose(context, 4) == [8, 9, 2, 3]
    assert proposer.propose(context, 1) == [8]


def test_suffix_lookup_requires_min_match_evidence() -> None:
    proposer = SuffixLookupProposer(ngram=2, min_match=4)
    context = [1, 5, 6, 7, 8, 9, 2, 3, 5, 6, 7]
    assert proposer.propose(context, 4) == []
    assert proposer.propose([5, 6, 7], 4) == []


def test_suffix_lookup_prefers_longest_match_over_most_recent() -> None:
    proposer = SuffixLookupProposer(ngram=2, min_match=2)
    # first occurrence of (3 4) is preceded by 2 (longer match with the tail
    # 2 3 4); the later occurrence is preceded by 9.
    context = [2, 3, 4, 100, 9, 3, 4, 200, 2, 3, 4]
    assert proposer.propose(context, 1) == [100]


def test_suffix_lookup_goes_silent_after_a_run_of_rejections() -> None:
    proposer = SuffixLookupProposer(ngram=2, min_match=2, silence_rounds=3, window=2)
    context = [1, 2, 3, 1, 2]
    assert proposer.propose(context, 1) == [3]
    proposer.observe(1, 0)
    assert proposer.propose(context, 1) == [3]
    proposer.observe(1, 0)  # two straight rejections: silence
    assert proposer.propose(context, 1) == []
    assert proposer.propose(context, 1) == []
    assert proposer.propose(context, 1) == []
    assert proposer.propose(context, 1) == [3]  # back after silence_rounds
    proposer.observe(1, 1)
    proposer.observe(1, 0)
    assert proposer.propose(context, 1) == [3]  # an acceptance in the window keeps it talking
    assert proposer.telemetry()["silenced_rounds"] == 3


def test_suffix_lookup_survives_context_replacement() -> None:
    proposer = SuffixLookupProposer(ngram=2, min_match=2)
    assert proposer.propose([1, 2, 3, 1, 2], 1) == [3]
    assert proposer.propose([7, 8, 9, 7, 8], 1) == [9]


# --------------------------------------------------------------------------
# the real mlx_lm cache protocol, no model
# --------------------------------------------------------------------------


def _mx():
    return pytest.importorskip("mlx.core")


def test_batch_kv_rollback_rolls_rejected_tail_into_left_padding() -> None:
    mx = _mx()
    from mlx_lm.models.cache import BatchKVCache

    cache = BatchKVCache([0, 0])
    prefix = mx.arange(2 * 1 * 4 * 1, dtype=mx.float32).reshape(2, 1, 4, 1) + 1000
    cache.update_and_fetch(prefix, prefix)
    step = mx.arange(2 * 1 * 3 * 1, dtype=mx.float32).reshape(2, 1, 3, 1) + 1
    cache.prepare(lengths=[3, 1], right_padding=[0, 2])
    cache.update_and_fetch(step, step)
    cache.finalize()
    mx.eval(cache.keys)
    assert cache._idx == 7
    assert cache.offset.tolist() == [7, 5]
    assert cache.left_padding.tolist() == [0, 2]
    row1 = cache.keys[1, 0, :, 0].tolist()
    assert row1[2:7] == [1004, 1005, 1006, 1007, 4]  # prefix then the one real token

    engine = LaneEngine.__new__(LaneEngine)
    engine.copy_recurrent_snapshots = False
    engine._rollback_rows([cache], [3, 0], [None])
    mx.eval(cache.keys)
    assert cache.offset.tolist() == [4, 5]
    assert cache.left_padding.tolist() == [3, 2]
    row0 = cache.keys[0, 0, :, 0].tolist()
    assert row0[3:7] == [1000, 1001, 1002, 1003]  # row 0 back to its prefix
    assert cache.keys[1, 0, :, 0].tolist()[2:7] == [1004, 1005, 1006, 1007, 4]


def test_arrays_cache_rows_restore_from_snapshot() -> None:
    mx = _mx()
    from mlx_lm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    before = [mx.zeros((3, 2)), mx.ones((3, 2))]
    cache.state = list(before)
    engine = LaneEngine.__new__(LaneEngine)
    engine.copy_recurrent_snapshots = False
    snap = engine._snapshot_recurrent([cache])
    cache.state = [mx.full((3, 2), 5.0), mx.full((3, 2), 7.0)]
    engine._rollback_rows([cache], [4, 0, 2], snap)
    mx.eval(*cache.state)
    assert cache.state[0].tolist() == [[0, 0], [5, 5], [0, 0]]
    assert cache.state[1].tolist() == [[1, 1], [7, 7], [1, 1]]


def test_copy_single_cache_detaches_kv_and_recurrent_arrays() -> None:
    mx = _mx()
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    keys = mx.ones((1, 1, 3, 2))
    kv.update_and_fetch(keys, keys)
    arrays = ArraysCache(size=2)
    arrays.state = [mx.zeros((1, 2)), mx.ones((1, 2))]
    clone = LaneEngine.copy_single_cache([kv, arrays])
    assert clone[0] is not kv and clone[0].offset == 3
    clone[0].keys[..., 0, :] = 9.0
    clone[1].state[0][0, 0] = 5.0
    mx.eval(clone[0].keys, clone[1].state[0], kv.keys, arrays.state[0])
    assert kv.keys[0, 0, 0, 0].item() == 1.0
    assert arrays.state[0][0, 0].item() == 0.0


# --------------------------------------------------------------------------
# end to end against the history-dependent fake target
# --------------------------------------------------------------------------


def test_fake_target_streams_match_their_serial_decodes() -> None:
    engine = FakeEngine(max_rows=24, max_draft=6, pending_cap=8)
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    limits = [40, 25, 31]
    eos_stream_two = fake_serial(prompts[2], 31, set())[12]
    eos_sets = [set(), set(), {eos_stream_two}]
    references = [fake_serial(p, n, e) for p, n, e in zip(prompts, limits, eos_sets)]
    patterns = [[3, 0, 6, 1], [2, 2, 0], [6, 6, 1, 0, 3]]
    streams = [
        LaneStream(
            stream_id=f"s{i}",
            prompt_ids=prompts[i],
            max_new_tokens=limits[i],
            eos_ids=frozenset(eos_sets[i]),
            proposer=PatternProposer(patterns[i]),
        )
        for i in range(3)
    ]
    for stream in streams:
        engine.add_stream(stream)

    seen: dict[str, list[int]] = {s.stream_id: [] for s in streams}
    while True:
        landed = engine.step()
        if not landed:
            break
        for stream_id, tokens in landed.items():
            seen[stream_id].extend(tokens)
        for stream in streams:
            if not stream.finished:
                assert stream.cache_len + len(stream.pending) == len(stream.prompt_ids) + len(
                    stream.emitted
                )

    for stream, reference in zip(streams, references):
        assert stream.emitted == reference, stream.stream_id
        assert stream.finished
        assert stream.emitted[1:] == seen[stream.stream_id]
    assert streams[2].finish_reason == "stop"
    assert streams[0].finish_reason == "length"
    summary = engine.summary()
    assert summary["committed_tokens"] == sum(len(r) - 1 for r in references)
    assert any(r.ragged for r in engine.round_stats)
    assert any(r.rollbacks for r in engine.round_stats)
    assert any(s.full_rounds for s in streams) and any(s.partial_rounds for s in streams)
    assert engine.filter_calls, "streams finishing at different times must filter rows"
    assert engine.active_count == 0
    assert engine.finished_caches == {}


def test_fake_engine_pending_cap_forces_an_absorb_round() -> None:
    engine = FakeEngine(max_rows=8, max_draft=3, pending_cap=3)
    stream = LaneStream(
        stream_id="s",
        prompt_ids=[3, 1, 4],
        max_new_tokens=30,
        proposer=PatternProposer([1]),  # every round: one right, then wrong
    )
    engine.add_stream(stream)
    while not stream.finished:
        engine.step()
        assert len(stream.pending) <= engine.pending_cap + 2
    assert stream.emitted == fake_serial([3, 1, 4], 30, set())
    assert any(r.width == len(stream.pending) or r.rollbacks == 0 for r in engine.round_stats)


def test_streams_can_join_mid_flight() -> None:
    engine = FakeEngine(max_rows=16, max_draft=4, pending_cap=6)
    first = LaneStream("a", [1, 1], 12, proposer=PatternProposer([4, 0]))
    engine.add_stream(first)
    engine.step()
    engine.step()
    second = LaneStream("b", [2, 2, 2], 9, proposer=PatternProposer([2]))
    engine.add_stream(second)
    assert engine.active_count == 2
    engine.run()
    assert first.emitted == fake_serial([1, 1], 12, set())
    assert second.emitted == fake_serial([2, 2, 2], 9, set())
    assert max(r.streams for r in engine.round_stats) == 2


def test_retained_cache_holds_exactly_the_absorbed_prefix_and_resumes() -> None:
    engine = FakeEngine(max_rows=16, max_draft=4, pending_cap=6, retain_finished_caches=True)
    prompt = [5, 6, 7]
    stream = LaneStream("a", prompt, 14, proposer=PatternProposer([2, 0, 5]))
    engine.add_stream(stream)
    engine.run()
    reference = fake_serial(prompt, 14, set())
    assert stream.emitted == reference
    absorbed, cache = engine.finished_caches["a"]
    assert absorbed == (prompt + reference)[: stream.cache_len]
    assert cache[0].rows[0] == absorbed
    assert stream.cache_len <= len(prompt) + len(reference)

    # Next turn: the conversation continues past the absorbed prefix.
    next_prompt = prompt + reference + [40, 41]
    resumed = LaneStream("b", next_prompt, 9, proposer=PatternProposer([3]))
    engine.add_stream(resumed, cache=engine.copy_single_cache(cache), cached_tokens=len(absorbed))
    assert engine.prefill_calls[-1] == ("b", len(absorbed))
    engine.run()
    assert resumed.emitted == fake_serial(next_prompt, 9, set())
    assert resumed.cached_tokens == len(absorbed)


def test_prefill_checkpoints_snapshot_each_boundary_in_order() -> None:
    engine = FakeEngine(max_rows=8, max_draft=2, pending_cap=4)
    stream = LaneStream("a", [1, 2, 3, 4, 5], 6)
    engine.add_stream(stream, checkpoints_at=[4, 2, 2, 9, 0])
    assert [tokens for tokens, _ in stream.history_checkpoints] == [[1, 2], [1, 2, 3, 4]]
    assert stream.history_checkpoints[1][1][0].rows[0] == [1, 2, 3, 4]
    engine.run()
    assert stream.emitted == fake_serial([1, 2, 3, 4, 5], 6, set())
    tokens, cache = stream.history_checkpoints[0]
    later = LaneStream("b", [1, 2, 9], 4)
    engine.add_stream(later, cache=engine.copy_single_cache(cache), cached_tokens=2, checkpoints_at=[2])
    assert later.history_checkpoints == []  # nothing new before the boundary
    engine.run()
    assert later.emitted == fake_serial([1, 2, 9], 4, set())


def test_reset_drops_rows_but_keeps_stream_state() -> None:
    engine = FakeEngine(max_rows=8, max_draft=2, pending_cap=4)
    stream = LaneStream("a", [1, 2], 5)
    engine.add_stream(stream)
    engine.step()
    engine.reset()
    assert engine.active_count == 0
    assert engine.step() == {}
    assert stream.emitted  # the tokens already committed survive


# --------------------------------------------------------------------------
# fan-out admission: forked lanes, grouped shedding, joins while running
# --------------------------------------------------------------------------


def _fan(engine: FakeEngine, prefix: list[int], suffixes: list[list[int]], budgets: list[int]):
    shared = engine.prefill_prefix(prefix)
    lanes = []
    for i, (suffix, budget) in enumerate(zip(suffixes, budgets)):
        lane = LaneStream(f"lane{i}", [*prefix, *suffix], budget)
        engine.add_forked(lane, cache=engine.copy_single_cache(shared), cached_tokens=len(prefix))
        lanes.append(lane)
    return lanes


def test_one_token_mode_walks_a_long_suffix_and_matches_serial() -> None:
    engine = FakeEngine(max_rows=64, max_draft=8, one_token=True)
    prefix = [5, 6, 7, 8]
    suffixes = [[1], [2, 3], [4, 5, 6], [9, 9, 9, 9]]
    lanes = _fan(engine, prefix, suffixes, [7, 9, 5, 11])
    engine.run()
    assert engine.round_stats and all(stat.width == 1 for stat in engine.round_stats)
    for lane, suffix, budget in zip(lanes, suffixes, [7, 9, 5, 11]):
        assert lane.emitted == fake_serial([*prefix, *suffix], budget, set())


def test_forked_lanes_absorb_their_suffix_in_one_round_and_match_serial() -> None:
    engine = FakeEngine(max_rows=64, max_draft=0)
    prefix = [5, 6, 7, 8]
    suffixes = [[1], [2, 3], [4, 5, 6], [9, 9, 9, 9]]
    lanes = _fan(engine, prefix, suffixes, [7, 9, 5, 11])
    assert engine.prefill_calls == []  # no per-lane prefill
    engine.run()
    for lane, suffix, budget in zip(lanes, suffixes, [7, 9, 5, 11]):
        assert lane.emitted == fake_serial([*prefix, *suffix], budget, set())
    assert engine.round_stats[0].width == 4 and engine.round_stats[0].ragged


def test_add_forked_requires_a_suffix() -> None:
    engine = FakeEngine(max_rows=8, max_draft=0)
    shared = engine.prefill_prefix([1, 2, 3])
    with pytest.raises(ValueError):
        engine.add_forked(LaneStream("x", [1, 2, 3], 4), cache=shared, cached_tokens=3)


def test_shrink_slack_sheds_in_groups_without_changing_any_stream() -> None:
    prefix = [3, 1, 4]
    suffixes = [[i + 1] for i in range(8)]
    budgets = [3, 4, 5, 6, 7, 8, 9, 10]
    eager = FakeEngine(max_rows=64, max_draft=0)
    eager_lanes = _fan(eager, prefix, suffixes, budgets)
    eager.run()
    lazy = FakeEngine(max_rows=64, max_draft=0, shrink_slack=3)
    lazy_lanes = _fan(lazy, prefix, suffixes, budgets)
    lazy.run()
    for a, b in zip(eager_lanes, lazy_lanes):
        assert a.emitted == b.emitted
    assert len(lazy.filter_calls) < len(eager.filter_calls)
    assert all(s.finished for s in lazy_lanes) and lazy.active_count == 0


def test_idle_placeholders_do_not_count_as_active_or_commit() -> None:
    engine = FakeEngine(max_rows=64, max_draft=0, shrink_slack=8)
    lanes = _fan(engine, [2, 2], [[1], [3]], [2, 9])
    engine.step()
    engine.step()
    assert lanes[0].finished and not lanes[1].finished
    assert engine.active_count == 1
    frozen = list(lanes[0].emitted)
    engine.step()
    assert lanes[0].emitted == frozen
    assert engine.round_stats[-1].streams == 1


def test_wide_suffix_joiners_do_not_widen_a_running_batch() -> None:
    engine = FakeEngine(max_rows=64, max_draft=0, shrink_slack=4)
    prefix = [7, 7, 7]
    first = _fan(engine, prefix, [[1], [2]], [12, 12])
    engine.step()
    engine.step()
    shared = engine.prefill_prefix(prefix)
    late = LaneStream("late", [*prefix, 4, 5, 6, 7, 8], 6)
    engine.add_forked(late, cache=engine.copy_single_cache(shared), cached_tokens=len(prefix))
    landed = engine.step()
    assert "late" in landed and landed["late"]  # its first token arrives with the join
    assert engine.round_stats[-1].width == 1  # running rows were not padded to 5
    engine.run()
    assert late.emitted == fake_serial([*prefix, 4, 5, 6, 7, 8], 6, set())
    for lane, suffix in zip(first, [[1], [2]]):
        assert lane.emitted == fake_serial([*prefix, *suffix], 12, set())


def test_join_sheds_idle_rows_in_the_same_resize() -> None:
    engine = FakeEngine(max_rows=64, max_draft=0, shrink_slack=8)
    prefix = [9, 1]
    lanes = _fan(engine, prefix, [[1], [2], [3]], [2, 2, 20])
    engine.step()
    engine.step()
    assert engine.filter_calls == []  # two finished rows idle inside the slack
    shared = engine.prefill_prefix(prefix)
    late = LaneStream("late", [*prefix, 5], 4)
    engine.add_forked(late, cache=engine.copy_single_cache(shared), cached_tokens=len(prefix))
    engine.step()
    assert engine.filter_calls == [[2]]
    engine.run()
    assert late.emitted == fake_serial([*prefix, 5], 4, set())
    assert lanes[2].emitted == fake_serial([*prefix, 3], 20, set())


def test_unconfident_drafts_stay_inside_the_cheap_window() -> None:
    from lane_fakes import FakeEngine

    class Proposer:
        last_confident = False

        def propose(self, context, max_draft):
            return list(range(max_draft))

    engine = FakeEngine(max_rows=64, max_draft=32, pending_cap=8)
    engine.cheap_window = 4
    stream = LaneStream(stream_id="s", prompt_ids=[1, 2, 3], max_new_tokens=100, proposer=Proposer())
    stream.pending = [7]
    assert len(engine._drafts_for(stream)) == 3
    stream.proposer.last_confident = True
    assert len(engine._drafts_for(stream)) == 32


def test_shadow_stats_score_the_targets_samples_past_a_rejection() -> None:
    """Rows below the rejected token (first-child chain) are the target's samples for the positions after
    it; once those positions commit, each is tallied as a hit or miss by its offset."""
    from types import SimpleNamespace

    from tensorfold.engine.lane_engine import LaneEngine

    engine = LaneEngine.__new__(LaneEngine)
    engine.shadow_stats = {}
    # root 0 at position 10; 1 under 0; 2 under 1 (rejected); 3 under 2; 4 under 0 (a second child)
    rows_parents = [-1, 0, 1, 2, 0]
    depths = [0, 1, 2, 3, 1]
    preds = [101, 102, 205, 206, 999]        # the target's sample after each row
    stream = SimpleNamespace(context=list(range(12)))    # positions 0..11 committed (10 = root, 11 = accepted row 1)
    engine._shadow_stats(stream, rows_parents, preds, [0, 1], 10, depths)
    assert stream._shadows == {13: (1, 205), 14: (2, 206)}
    stream.context += [102, 205, 7]           # 12 = the bonus, 13 matches its shadow, 14 does not
    engine._shadow_stats(stream, [-1], [0], [0], 14, [0])
    assert engine.shadow_stats == {1: [1, 1], 2: [0, 1]}


def test_sanitize_tree_truncates_and_drops_orphans() -> None:
    from tensorfold.engine.lane_engine import sanitize_tree

    assert sanitize_tree([9, 8, 7], [-1, 0, 1], budget=2) == ([9, 8], [-1, 0])
    tokens, parents = sanitize_tree([9, 8, 7, 6], [-1, 2, 0, 1], budget=4)   # node 1's parent comes after it
    assert tokens == [9, 7] and parents == [-1, 0]                            # node 3 followed node 1 out
    assert sanitize_tree([9, 8, 7], [1, 0, 5], budget=3) == ([], [])         # nothing valid survives
