"""Resumes at message starts through the scheduler and the saved-block warm-up (the fake engine with openers)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tensorfold.server.checkpoints import CheckpointStore
from tests.lane_fakes import FakeBatchItem, FakeEngine
from tests.test_lane_server import make_app

OPEN, ASSIST = 90, 91                   # a message's first token; the role token of an assistant message


class MessageEngine(FakeEngine):
    """Chunks start at message openers at least 3 tokens apart, and every 8 tokens."""

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        from tensorfold.engine.prefill_plan import PrefillPlan

        super().__init__(model, **kwargs)
        self.prefill_plan = PrefillPlan(8, (OPEN,), 3, (OPEN, ASSIST))


def _message_scheduler() -> tuple[Any, CheckpointStore, MessageEngine]:
    from tensorfold.server.scheduler import Scheduler

    store = CheckpointStore(4, copier=lambda c: c)
    engine = MessageEngine()
    return Scheduler(engine, lanes=1, eos_ids=frozenset(), checkpoints=store), store, engine


def test_a_saved_block_ending_at_a_message_is_warmed_to_its_end(tmp_path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from tensorfold.engine.prefix_snapshots import save_snapshot
    from tensorfold.server.scheduler import Scheduler

    block = [OPEN, *range(20, 30)]                           # a system message; the next message began at 11
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 11, 4)), mx.ones((1, 2, 11, 4)))
    save_snapshot(tmp_path, "/models/fake|kernels=old", block, [kv])
    seen: list[list[int]] = []
    real = Scheduler.submit

    def submit(self: Any, job: Any) -> None:
        seen.append(list(job.prompt_ids))
        real(self, job)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Scheduler, "submit", submit)
        app = make_app(engine_factory=MessageEngine, snapshot_dir=tmp_path, model_id="/models/fake|kernels=new")
        try:
            deadline = time.time() + 10
            while time.time() < deadline and not any(e.pinned for e in app.checkpoints._entries):
                time.sleep(0.02)
        finally:
            app.scheduler.stop(timeout=5.0)
    # the grid start 8, then the block's end with an opener after it, so its end is a chunk start as it was
    assert seen == [block[:9], [*block, OPEN]]
    assert [e.tokens for e in app.checkpoints._entries if e.pinned] == [block]


def test_a_follow_up_resumes_at_its_latest_reply_start() -> None:
    from tensorfold.server.scheduler import ChatJob

    scheduler, store, engine = _message_scheduler()
    turn1 = [OPEN, 5, 6, 7, OPEN, 8, 9, OPEN, ASSIST]       # system, user, then the generation prompt at 7
    scheduler._start_job(ChatJob(job_id="a", prompt_ids=turn1, max_tokens=2, temperature=0.0, history_len=7))
    assert [len(e.tokens) for e in store._entries] == [7]
    # the reply and a new message after the same history: resumed where the reply began
    turn2 = [*turn1[:7], OPEN, ASSIST, 12, 13, OPEN, 14, 15, 16, OPEN, ASSIST]
    scheduler._start_job(ChatJob(job_id="b", prompt_ids=turn2, max_tokens=2, temperature=0.0, history_len=15))
    assert engine.prefill_calls[-1] == ("b", 7)
    assert sorted(len(e.tokens) for e in store._entries) == [7, 15]


def test_a_start_too_close_to_the_last_is_merged_and_the_checkpoint_falls_back() -> None:
    from tensorfold.server.scheduler import ChatJob

    scheduler, store, engine = _message_scheduler()
    turn1 = [OPEN, 5, 6, 7, OPEN, 8, OPEN, ASSIST]          # the generation prompt 2 tokens after the user's start
    scheduler._start_job(ChatJob(job_id="a", prompt_ids=turn1, max_tokens=2, temperature=0.0, history_len=6))
    assert [len(e.tokens) for e in store._entries] == [4]   # its chunk began at the user message
    turn2 = [*turn1[:6], OPEN, ASSIST, 12, OPEN, 14, OPEN, ASSIST]
    scheduler._start_job(ChatJob(job_id="b", prompt_ids=turn2, max_tokens=2, temperature=0.0))
    assert engine.prefill_calls[-1] == ("b", 4)


def test_a_stored_prefix_resumes_only_a_prompt_with_a_chunk_start_there() -> None:
    from tensorfold.server.scheduler import ChatJob

    scheduler, store, engine = _message_scheduler()
    stored = [OPEN, 5, 6, 7, OPEN, 8, 9]
    store.insert(stored, [FakeBatchItem([stored])], last_prompt=stored)
    # the same tokens followed by plain text: no chunk starts at 7 in this prompt, so the state is not resumed
    scheduler._start_job(ChatJob(job_id="plain", prompt_ids=[*stored, 3, 4], max_tokens=2, temperature=0.0))
    assert engine.prefill_calls[-1] == ("plain", 0)
    scheduler._start_job(ChatJob(job_id="reply", prompt_ids=[*stored, OPEN, ASSIST], max_tokens=2, temperature=0.0))
    assert engine.prefill_calls[-1] == ("reply", 7)


def test_a_shared_system_block_is_kept_at_the_first_message_after_it() -> None:
    from tensorfold.server.scheduler import ChatJob

    scheduler, store, engine = _message_scheduler()
    system = [OPEN, *range(20, 30)]                         # 11 tokens: a grid start at 8, the user's opener at 11
    first = [*system, OPEN, 5, 6, OPEN, ASSIST]
    scheduler._start_job(ChatJob(job_id="a", prompt_ids=first, max_tokens=2, temperature=0.0, history_len=14,
                                 shared_prefix_lens=(13,)))
    assert sorted((len(e.tokens), e.pinned) for e in store._entries) == [(11, True), (14, False)]
    other = [*system, OPEN, 7, 8, 9, OPEN, ASSIST]           # another session with the same system block
    scheduler._start_job(ChatJob(job_id="b", prompt_ids=other, max_tokens=2, temperature=0.0))
    assert engine.prefill_calls[-1] == ("b", 11)
