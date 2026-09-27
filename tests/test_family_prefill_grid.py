"""Family prompts are prefilled on the engine's grid: a prompt resumed from a checkpoint gets a fresh prefill's chunks
and so its bits, whatever the chunking does to a model's state (a fake whose state records every chunk)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

V = 97
GRID = 4


class _Cache:
    def __init__(self) -> None:
        self.chunks: list[tuple[int, ...]] = []      # every prompt chunk and decode window, as fed
        self.state = None


class ChunkModel:
    """Its next token depends on how the context was chunked (as MLX's prefill does): the head reads the number of
    chunks fed so far."""

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


def _engine(retain: bool = True) -> LaneEngine:
    engine = LaneEngine(ChunkModel(), retain_finished_caches=retain)
    engine.prefill_align = GRID
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


def test_without_a_grid_decoded_states_are_kept():
    engine = LaneEngine(ChunkModel(), retain_finished_caches=True)
    engine.prefill_align = 0
    stream = LaneStream("a", list(range(5)), 4)
    _run(engine, stream)
    assert "a" in engine.finished_caches
