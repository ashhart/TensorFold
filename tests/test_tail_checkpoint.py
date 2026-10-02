"""A long history also keeps the chunk start before its boundary, so a turn that edits the end of the last message
resumes one chunk back instead of prefilling the whole prompt again (``Scheduler(tail_checkpoint_tokens=...)``)."""

from __future__ import annotations

import pytest

from tensorfold.engine.prefill_plan import PrefillPlan
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.lane_fakes import FakeEngine


class GridEngine(FakeEngine):
    prefill_plan = PrefillPlan(4)


def _scheduler(tail: int) -> tuple[Scheduler, CheckpointStore]:
    store = CheckpointStore(8, copier=lambda c: c)
    return Scheduler(GridEngine(), lanes=1, eos_ids=frozenset(), checkpoints=store,
                     tail_checkpoint_tokens=tail), store


def _run(scheduler: Scheduler, job_id: str, prompt: list[int], history_len: int) -> ChatJob:
    job = ChatJob(job_id=job_id, prompt_ids=prompt, max_tokens=2, temperature=0.0, history_len=history_len)
    scheduler._start_job(job)
    return job


PROMPT = list(range(1, 31))      # 30 tokens; grid starts 0, 4, ..., 28
HISTORY = 28                     # the generation suffix is the last two tokens


def test_a_long_history_keeps_the_chunk_start_before_its_boundary() -> None:
    scheduler, store = _scheduler(tail=16)
    _run(scheduler, "a", PROMPT, HISTORY)
    assert sorted(len(e.tokens) for e in store._entries) == [24, 28]


def test_the_tail_checkpoint_is_off_at_zero_and_below_its_length() -> None:
    for tail in (0, 29):
        scheduler, store = _scheduler(tail=tail)
        _run(scheduler, "a", PROMPT, HISTORY)
        assert sorted(len(e.tokens) for e in store._entries) == [28]


@pytest.mark.parametrize("tail, resumed", [(16, 24), (0, 0)])
def test_an_edit_at_the_end_of_the_last_message_resumes_one_chunk_back(tail: int, resumed: int) -> None:
    scheduler, _ = _scheduler(tail=tail)
    _run(scheduler, "a", PROMPT, HISTORY)
    # the same conversation with a line appended to its last message: it diverges at 26, inside the last chunk
    edited = PROMPT[:26] + [91, 92, 93, 94] + PROMPT[26:]
    job = _run(scheduler, "b", edited, HISTORY + 4)
    assert job.cached_tokens == resumed


def test_a_negative_tail_is_refused() -> None:
    with pytest.raises(ValueError):
        Scheduler(GridEngine(), lanes=1, eos_ids=frozenset(), tail_checkpoint_tokens=-1)
