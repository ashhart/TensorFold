"""Image prompts in the prefix store: keyed by their images, resumed like text prompts, never mixed up."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tensorfold.vision.qwen_processing import PreparedVisionPrompt, prefix_key
from tests.lane_fakes import FakeFamily, fake_serial
from tests.test_prompt_fill import GridEngine, _prompt, _run, _tokens

PLACEHOLDER = 7


def seen(prepared: Any) -> list[int]:
    """What the fake reads: an image row as its image's own content, apart from any prefix-store id."""

    rows = [int(t) for t in prepared.token_ids]
    for (begin, end), digest in zip(prepared.image_spans, prepared.image_hashes):
        pixels = int(hashlib.sha256(digest.encode()).hexdigest()[:8], 16)
        rows[begin:end] = [1000 + pixels % 9000 + row for row in range(end - begin)]
    return rows


class ImageFamily(FakeFamily):
    """The fake target reading image rows as their pixels: a prefix resumed through other pixels shows in the reply."""

    image_resume = True

    def encode_vision(self, prepared: Any, cache: list[Any], start: int = 0) -> Any:
        return SimpleNamespace(rows=seen(prepared))

    def prefill_vision(self, inputs: Any, cache: list[Any], encoded: Any, begin: int, end: int) -> Any:
        return self.hidden(encoded.rows[begin:end], cache)


class ImageEngine(GridEngine):
    """A chunk every 4 prompt tokens; records each prefill's resume and checks the cache against what it read."""

    def __init__(self) -> None:
        super().__init__(ImageFamily())

    def _family_prefill_steps(self, stream: Any, *, cache: Any, cached_tokens: int, checkpoints_at: Any) -> Any:
        rows = seen(stream.prompt_data) if stream.prompt_data is not None else stream.prompt_ids
        cached = int(cached_tokens) if cache is not None else 0
        if cache is not None and list(cache[0].rows[0]) != list(rows[:cached]):
            raise AssertionError("a stored prefix that is not the prompt's own")
        self.prefill_calls.append((stream.stream_id, cached))
        return (yield from LaneEngine._family_prefill_steps(self, stream, cache=cache, cached_tokens=cached_tokens,
                                                            checkpoints_at=checkpoints_at))


def image_job(name: str, before: list[int], rows: int, after: list[int], image: str, history: int = 0) -> ChatJob:
    ids = [*before, *[PLACEHOLDER] * rows, *after]
    prepared = PreparedVisionPrompt(tuple(ids), np.zeros((4 * rows, 1)), np.array([[1, 2, 2 * rows]]),
                                    np.zeros((3, 1, len(ids)), dtype=np.int32), 0,
                                    ((len(before), len(before) + rows),), (image,))
    return ChatJob(name, ids, 4, 0.0, history_len=history, vision=prepared, cache_ids=prefix_key(prepared))


def solo(job: ChatJob) -> bool:
    rows = seen(job.vision) if job.vision is not None else job.prompt_ids
    return job.error is None and _tokens(job) == fake_serial(rows, job.max_tokens, {-1})


def test_an_image_prompt_resumes_through_the_same_image_and_never_through_another():
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = ImageEngine()
    first = image_job("first", _prompt(1, 6), 8, _prompt(2, 8), "cat", history=18)   # rows 6..13 the image
    _run(Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store), [first])
    stored = {len(e.tokens): e.tokens for e in store._entries}
    assert 16 in stored and stored[16] == first.cache_ids[:16] and max(stored[16][6:14]) < 0   # the history's chunk
    # the next turn: the same history, the same image, another ending: it resumes where the history's chunk starts
    again = image_job("again", _prompt(1, 6), 8, [*_prompt(2, 8)[:4], *_prompt(3, 9)], "cat")
    # the same ids, another image: no stored prefix reaches past its first image row
    other = image_job("other", _prompt(1, 6), 8, [*_prompt(2, 8)[:4], *_prompt(3, 9)], "dog")
    _run(Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store), [again, other])
    assert engine.prefill_calls == [("first", 0), ("again", 16), ("other", 0)]
    assert solo(first) and solo(again) and solo(other)


def test_a_text_prompt_resumes_from_the_text_an_image_prompt_stored_before_its_image():
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = ImageEngine()
    shared = _prompt(4, 12)                                       # a system block, then the image
    first = image_job("first", shared, 8, _prompt(5, 6), "cat", history=8)
    text = ChatJob("text", [*shared, *_prompt(6, 10)], 4, 0.0)
    _run(Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store), [first])
    _run(Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store), [text])
    assert engine.prefill_calls == [("first", 0), ("text", 8)] and solo(first) and solo(text)


def test_image_workspace_is_sized_from_the_row_the_prompt_resumes_and_checked_again_if_that_prefix_went():

    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = ImageEngine()
    first = image_job("first", _prompt(1, 6), 8, _prompt(2, 8), "cat", history=18)
    _run(Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store), [first])
    calls: list[tuple[int, Any]] = []
    estimates: list[int] = []

    class Memory:
        def begin(self, *args: Any, **kwargs: Any) -> None:
            return None

        def require(self, **kwargs: Any) -> None:
            return None

        def fits_now(self, extra_bytes: int = 0) -> bool:
            return True

        def require_workspace(self, size: int, current_cache: Any = None, keep: Any = None) -> None:
            calls.append((size, keep))

    engine.model.vision = SimpleNamespace(estimate_workspace_bytes=lambda prepared, start=0: estimates.append(start)
                                          or 0)
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset({-1}), checkpoints=store)
    scheduler.prompt_memory = Memory()
    again = image_job("again", _prompt(1, 6), 8, [*_prompt(2, 8)[:4], *_prompt(3, 9)], "cat")
    filling = scheduler._open_job(again)
    assert filling is not None and estimates == [16]
    assert calls and calls[0][1] is not None and calls[0][1].tokens == first.cache_ids[:16]
    scheduler._start_fill(filling)
    assert estimates == [16, 16, 16] and len(calls) == 2      # in the copy-or-take choice, then checked again
    other = image_job("other", _prompt(1, 6), 8, [*_prompt(2, 8)[:4], *_prompt(3, 9)], "cat")
    filling = scheduler._open_job(other)
    store._entries.clear()                        # another prompt took the room before this one's first chunk
    scheduler._start_fill(filling)
    assert estimates[3:] == [16, 0] and len(calls) == 4       # its images need their workspace after all


def test_a_saved_image_prefix_reads_back_with_its_keys_and_positions(tmp_path):
    """An image prompt's prefix on disk: its negative ids and its rotary table come back as they were stored."""

    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot
    from tensorfold.families.qwen4_exp.model_layers import AttentionCache
    from tensorfold.vision.rotary import attach_positions

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        job = image_job("saved", _prompt(1, 6), 8, _prompt(2, 6), "cat")
        cache = AttentionCache()
        cache.update(mx.ones((1, 1, 20, 4)), mx.ones((1, 1, 20, 4)), mx.ones((1, 20, 4)))
        table = np.arange(60, dtype=np.int32).reshape(3, 20)
        attach_positions([cache], table, -3)
        path = save_snapshot(tmp_path, "model", job.cache_ids, [cache])
        tokens, loaded = load_snapshot(path, "model")
        assert tokens == job.cache_ids and min(tokens) < -(1 << 64)
        assert np.array_equal(loaded[0].vision_positions, table) and loaded[0].vision_rope_delta == -3
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize(("budget", "workspace"), [(2000, 0), (2300, 200)])
def test_an_image_prompt_resumes_by_transfer_when_only_the_transfer_fits(budget, workspace):
    """As a text prompt does (test_resume_memory), its image workspace counted in the copy-or-take choice."""

    from tensorfold.server.memory_budget import cache_nbytes
    from tensorfold.server.prompt_memory import PromptMemory
    from tests.test_prompt_memory import Runtime
    from tests.test_resume_memory import refuse_copy, sized

    job = image_job("resume", _prompt(1, 4), 8, _prompt(2, 52), "cat")         # images wholly before row 48
    store = CheckpointStore(3, copier=refuse_copy, sizer=cache_nbytes)
    store.insert([7] * 8, sized([7] * 8, 400), last_prompt=[7] * 8)
    store.insert(job.cache_ids[:48], sized(seen(job.vision)[:48], 400), last_prompt=job.cache_ids[:48])
    runtime = Runtime()
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    fused = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    # resuming needs 1,800 bytes once the other prefix goes, a copy beside the stored arrays 2,200, each with the workspace
    memory = PromptMemory(budget, fused, runtime=runtime, store=store, window_tokens=8192, overhead_bytes=0,
                          bootstrap_bytes=0)
    store.admit_oversize = True
    engine = ImageEngine()
    engine.model.vision = SimpleNamespace(estimate_workspace_bytes=lambda prepared, start=0: workspace)
    scheduler = Scheduler(engine, lanes=1, eos_ids=frozenset(), checkpoints=store, prompt_memory=memory)
    scheduler._start_job(job)
    assert job.error is None and engine.prefill_calls == [("resume", 48)]
