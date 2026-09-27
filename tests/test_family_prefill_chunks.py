"""A prompt resumed from a checkpoint gets a fresh prefill's chunks, so its bits (a fake that records every chunk)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402
from tensorfold.engine.prefill_plan import PrefillPlan  # noqa: E402

V = 97
GRID = 4


class _Cache:
    def __init__(self) -> None:
        self.chunks: list[tuple[int, ...]] = []      # every prompt chunk and decode window, as fed
        self.state = None


class ChunkModel:
    """Its next token depends on how the context was chunked, as MLX's prefill does (the head reads the chunk count)."""

    lane_family = True
    exact_width = 1
    gpu_tokens = False
    mtp = None

    def make_cache(self) -> list[_Cache]:
        return [_Cache()]

    def hidden(self, inputs, cache):
        tokens = tuple(int(t) for t in np.array(inputs).reshape(-1))
        cache[0].chunks.append(tokens)
        marks = [len(cache[0].chunks) * 1000 + t for t in tokens]
        return mx.array(marks, dtype=mx.float32).reshape(1, -1, 1)

    def head(self, hidden):
        marks = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(marks), V), dtype=np.float32)
        for i, m in enumerate(marks):
            logits[0, i, (m // 1000 * 7 + m % 1000) % V] = 10.0
        return mx.array(logits)

    def keep_rows(self, cache, rows: int, keep: int) -> None:
        pass


def _engine(retain: bool = True, markers: bool = False) -> LaneEngine:
    engine = LaneEngine(ChunkModel(), retain_finished_caches=retain)
    engine.prefill_plan = PrefillPlan(GRID, (OPEN,), 2, (OPEN, ASSIST)) if markers else PrefillPlan(GRID)
    return engine


def _run(engine: LaneEngine, stream: LaneStream, **kwargs) -> list[Any]:
    engine.add_stream(stream, **kwargs)
    cache = engine._live[-1][1] if engine._live else None
    while engine.active_count:
        engine.step()
    return cache


def test_a_prompt_resumed_from_a_grid_checkpoint_equals_a_fresh_one():
    prompt = list(range(10, 21))                                     # 11 tokens: chunks 4 + 4 + 3
    fresh = LaneStream("fresh", prompt, 5)
    cache = _run(_engine(), fresh)
    assert cache[0].chunks[:3] == [tuple(prompt[0:4]), tuple(prompt[4:8]), tuple(prompt[8:11])]

    first = LaneStream("first", prompt[:9], 3)
    engine = _engine()
    engine.add_stream(first, checkpoints_at=[7])                     # snapped to the grid point 4
    assert [len(tokens) for tokens, _ in first.history_checkpoints] == [4]
    tokens, stored = first.history_checkpoints[0]
    resumed = LaneStream("resumed", prompt, 5)
    engine = _engine()
    engine.add_stream(resumed, cache=engine.copy_single_cache(stored), cached_tokens=len(tokens))
    while engine.active_count:
        engine.step()
    assert resumed.cached_tokens == 4
    assert resumed.emitted == fresh.emitted


def test_a_state_off_the_grid_is_not_resumed_and_decoded_states_are_not_kept():
    prompt = list(range(30, 43))
    fresh = LaneStream("fresh", prompt, 6)
    engine = _engine()
    _run(engine, fresh)
    assert engine.finished_caches == {}                              # decoded rows are not a prefill's

    engine = _engine()
    off = engine.model.make_cache()
    engine.model.hidden(mx.array([prompt[:6]], dtype=mx.uint32), off)   # a state after 6 tokens, off the grid
    again = LaneStream("again", prompt, 6)
    engine.add_stream(again, cache=off, cached_tokens=6)
    while engine.active_count:
        engine.step()
    assert again.cached_tokens == 0
    assert again.emitted == fresh.emitted


def test_without_a_plan_decoded_states_are_kept():
    engine = LaneEngine(ChunkModel(), retain_finished_caches=True)
    engine.prefill_plan = None
    stream = LaneStream("a", list(range(5)), 4)
    _run(engine, stream)
    assert "a" in engine.finished_caches


def test_every_prompt_chunk_counts_as_progress():
    engine = _engine()
    _run(engine, LaneStream("a", list(range(10, 21)), 3))          # 11 tokens on a grid of 4: 3 chunks
    assert engine.prefill_chunks == 3


OPEN, ASSIST = 90, 91                  # a message's first token; the role token of an assistant message


def test_a_follow_up_resumed_at_its_reply_start_equals_the_conversation_fed_fresh():
    turn1 = [OPEN, 11, 12, 13, 14, 15, OPEN, 16, 17, OPEN, ASSIST]      # starts 0, 4 (grid), 6, 9
    first = LaneStream("first", turn1, 3)
    engine = _engine(markers=True)
    engine.add_stream(first, checkpoints_at=[9])                      # the generation prompt's start
    assert [len(tokens) for tokens, _ in first.history_checkpoints] == [9]
    tokens, stored = first.history_checkpoints[0]
    turn2 = [*turn1[:9], OPEN, ASSIST, 20, 21, OPEN, 22, 23, OPEN, ASSIST]   # its reply and a new message
    fresh = LaneStream("fresh", turn2, 5)
    cache = _run(_engine(markers=True), fresh)
    resumed = LaneStream("resumed", turn2, 5)
    engine = _engine(markers=True)
    engine.add_stream(resumed, cache=engine.copy_single_cache(stored), cached_tokens=len(tokens))
    resumed_cache = engine._live[-1][1]
    while engine.active_count:
        engine.step()
    assert resumed.cached_tokens == 9 and resumed.emitted == fresh.emitted
    assert resumed_cache[0].chunks == cache[0].chunks                 # the same chunks, prompt and decode alike
    # the grid alone cuts turn 2 differently, and so answers differently
    grid_only = LaneStream("grid", turn2, 5)
    _run(_engine(), grid_only)
    assert grid_only.emitted != fresh.emitted


def test_a_state_between_chunk_starts_is_not_resumed():
    turn = [OPEN, 11, 12, OPEN, 13, 14, 15, 16, OPEN, 17]
    fresh = LaneStream("fresh", turn, 4)
    _run(_engine(markers=True), fresh)
    engine = _engine(markers=True)
    state = engine.model.make_cache()
    engine.model.hidden(mx.array([turn[:5]], dtype=mx.uint32), state)   # 5 is no chunk start of this prompt
    again = LaneStream("again", turn, 4)
    engine.add_stream(again, cache=state, cached_tokens=5)
    while engine.active_count:
        engine.step()
    assert again.cached_tokens == 0 and again.emitted == fresh.emitted


def test_the_guards_run_at_every_message_chunk():
    from tensorfold.server.cancellation import Cancellation, PrefillGuard, RequestCancelled

    calls: list[tuple[str, int]] = []

    class Memory:
        def before_chunk(self, cache, rows):
            calls.append(("before", rows))

        def after_chunk(self, cache, rows):
            calls.append(("after", rows))

        def allow_checkpoint(self, cache):
            return False                                           # no room: the prefill goes on without it

    turn = [OPEN, 11, 12, 13, 14, 15, OPEN, 16, 17, OPEN, ASSIST]    # chunks of 4, 2, 3 and 2 tokens
    engine = _engine(markers=True)
    engine.prefill_guard = PrefillGuard(Cancellation(), Memory())
    stream = LaneStream("s", turn, 2)
    engine.add_stream(stream, checkpoints_at=[9])
    assert stream.history_checkpoints == []
    assert calls == [(side, rows) for rows in (4, 2, 3, 2) for side in ("before", "after")]

    stop = Cancellation()

    class Cancel(Memory):
        def after_chunk(self, cache, rows):
            super().after_chunk(cache, rows)
            stop.cancel()                                           # the client left during the first chunk

    calls.clear()
    engine = _engine(markers=True)
    engine.prefill_guard = PrefillGuard(stop, Cancel())
    with pytest.raises(RequestCancelled):
        engine.add_stream(LaneStream("c", turn, 2))
    assert calls == [("before", 4), ("after", 4)] and not engine._live


def test_a_prefill_stopped_between_chunks_keeps_its_last_start_and_a_retry_equals_fresh():
    from tensorfold.server.cancellation import Cancellation, PrefillGuard, RequestCancelled

    turn = [OPEN, 11, 12, 13, 14, 15, OPEN, 16, 17, OPEN, ASSIST]     # chunks start at 0, 4, 6 and 9
    fresh = LaneStream("fresh", turn, 5)
    _run(_engine(markers=True), fresh)
    stop, done = Cancellation(), []

    class StopAfterTwo:
        def before_chunk(self, cache, rows):
            pass

        def after_chunk(self, cache, rows):
            done.append(rows)
            if len(done) == 2:
                stop.cancel()                                       # the client left during the second chunk

        def allow_checkpoint(self, cache):
            return True

    engine = _engine(markers=True)
    engine.prefill_guard = PrefillGuard(stop, StopAfterTwo())
    stopped = LaneStream("stopped", turn, 5)
    with pytest.raises(RequestCancelled):
        engine.add_stream(stopped)
    assert [len(tokens) for tokens, _ in stopped.history_checkpoints] == [6]     # the start the prefill reached
    tokens, stored = stopped.history_checkpoints[0]
    retry = LaneStream("retry", turn, 5)
    engine = _engine(markers=True)
    engine.add_stream(retry, cache=engine.copy_single_cache(stored), cached_tokens=len(tokens))
    while engine.active_count:
        engine.step()
    assert retry.cached_tokens == 6 and retry.emitted == fresh.emitted
