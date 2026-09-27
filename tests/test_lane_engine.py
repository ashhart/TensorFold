"""Tests for the lane engine: the copy proposer, cache copies and draft-tree bookkeeping directly, and rounds end
to end against the history-dependent fake target served as a family (``tests/lane_fakes.py``), so any rollback or
refeed mistake shows up as a byte divergence from the fake's own serial decode.
"""

from __future__ import annotations

import gc

import pytest

pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream, SuffixLookupProposer  # noqa: E402
from tests.lane_fakes import FakeEngine, PatternProposer, fake_serial  # noqa: E402


# --------------------------------------------------------------------------
# proposers, cache copies, trees
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


def test_copy_single_cache_copies_a_view_out_of_its_base() -> None:
    mx = _mx()
    from mlx_lm.models.cache import ArraysCache

    mx.set_cache_limit(0)
    base = mx.random.normal((64, 256, 256))
    mx.eval(base)
    arrays = ArraysCache(size=1)
    arrays.state = [base[5:6]]
    clone = LaneEngine.copy_single_cache([arrays])
    mx.eval(clone[0].state[0])
    assert mx.array_equal(clone[0].state[0], base[5:6]).item()
    before = mx.get_active_memory()
    del arrays, base
    gc.collect()
    assert before - mx.get_active_memory() >= 63 * 256 * 256 * 4       # the base is freed, the copy kept


# --------------------------------------------------------------------------
# end to end against the history-dependent fake target
# --------------------------------------------------------------------------


def test_fake_target_streams_match_their_serial_decodes() -> None:
    engine = FakeEngine(max_rows=24, max_draft=6)
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    limits = [40, 25, 31]
    eos_stream_two = fake_serial(prompts[2], 31, set())[12]
    eos_sets = [set(), set(), {eos_stream_two}]
    references = [fake_serial(p, n, e) for p, n, e in zip(prompts, limits, eos_sets)]
    patterns = [[3, 0, 6, 1], [2, 2, 0], [6, 6, 1, 0, 3]]
    streams = [LaneStream(stream_id=f"s{i}", prompt_ids=prompts[i], max_new_tokens=limits[i],
                          eos_ids=frozenset(eos_sets[i]), proposer=PatternProposer(patterns[i])) for i in range(3)]
    for stream in streams:
        engine.add_stream(stream)
    seen: dict[str, list[int]] = {s.stream_id: [] for s in streams}
    while engine.active_count:
        for stream_id, tokens in engine.step().items():
            seen[stream_id].extend(tokens)
    for stream, reference in zip(streams, references):
        assert stream.emitted == reference, stream.stream_id
        assert stream.finished
        assert stream.emitted[1:] == seen[stream.stream_id]
    assert streams[2].finish_reason == "stop"
    assert streams[0].finish_reason == "length"
    assert any(r.streams > 1 for r in engine.round_stats)          # shared rounds
    assert any(r.rollbacks for r in engine.round_stats)            # windows kept in part
    assert engine.active_count == 0
    assert engine.finished_caches == {}


def test_streams_can_join_mid_flight() -> None:
    engine = FakeEngine(max_rows=16, max_draft=4)
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
    engine = FakeEngine(max_rows=16, max_draft=4, retain_finished_caches=True)
    prompt = [5, 6, 7]
    stream = LaneStream("a", prompt, 14, proposer=PatternProposer([2, 0, 5]))
    engine.add_stream(stream)
    engine.run()
    reference = fake_serial(prompt, 14, set())
    assert stream.emitted == reference
    absorbed, cache = engine.finished_caches["a"]
    assert absorbed == (prompt + reference)[: stream.cache_len]
    assert cache[0].rows[0] == absorbed
    next_prompt = prompt + reference + [40, 41]
    resumed = LaneStream("b", next_prompt, 9, proposer=PatternProposer([3]))
    engine.add_stream(resumed, cache=engine.copy_single_cache(cache), cached_tokens=len(absorbed))
    assert engine.prefill_calls[-1] == ("b", len(absorbed))
    engine.run()
    assert resumed.emitted == fake_serial(next_prompt, 9, set())
    assert resumed.cached_tokens == len(absorbed)


def test_prefill_checkpoints_snapshot_each_boundary_in_order() -> None:
    engine = FakeEngine(max_rows=8, max_draft=2)
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
    engine = FakeEngine(max_rows=8, max_draft=2)
    stream = LaneStream("a", [1, 2], 5)
    engine.add_stream(stream)
    engine.step()
    engine.reset()
    assert engine.active_count == 0
    assert engine.step() == {}
    assert stream.emitted  # the tokens already committed survive


def test_only_a_family_model_runs_on_the_engine() -> None:
    with pytest.raises(TypeError, match="family"):
        LaneEngine(object())


def test_sanitize_tree_truncates_and_drops_orphans() -> None:
    from tensorfold.engine.lane_engine import sanitize_tree

    assert sanitize_tree([9, 8, 7], [-1, 0, 1], budget=2) == ([9, 8], [-1, 0])
    tokens, parents = sanitize_tree([9, 8, 7, 6], [-1, 2, 0, 1], budget=4)   # node 1's parent comes after it
    assert tokens == [9, 7] and parents == [-1, 0]                            # node 3 followed node 1 out
    assert sanitize_tree([9, 8, 7], [1, 0, 5], budget=3) == ([], [])         # nothing valid survives
