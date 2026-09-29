"""Streams take memory as they grow: the newest wait, or end, when the next round's growth can't fit."""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace
from typing import Any

from tensorfold.engine.memory import Admission, StreamMemory
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.errors import RequestError
from tensorfold.server.memory_budget import cache_nbytes
from tensorfold.server.prompt_memory import PromptMemory
from tensorfold.server.scheduler import ChatJob, Scheduler
from tensorfold.server.stream_gate import StreamGate
from tests.lane_fakes import FakeEngine, fake_serial
from tests.test_memory_window import Runtime, SizedFamily, SizedKV
from tests.test_prompt_memory import populated

KIB, MIB = 1024, 2**20


class Held:
    """PromptMemory's surface the gate reads: MLX's use, freed buffers, retained prefixes, one reclaim step."""

    def __init__(self, used: int, cache: int = 0, store: Any = None) -> None:
        self.base, self.cache, self.store = used, cache, store
        self.runtime = SimpleNamespace(get_cache_memory=lambda: self.cache)
        self._memory_lock = threading.RLock()

    def _used(self) -> int:
        return self.base + self.cache + (self.store.nbytes if self.store is not None else 0)

    def _reclaim(self) -> bool:
        if self.cache:
            self.cache = 0
            return True
        return self.store is not None and self.store.evict_one()


def prefix_store() -> CheckpointStore:
    store = CheckpointStore(4, copier=lambda c: c, sizer=cache_nbytes)
    store.insert([1], populated(64), last_prompt=[1])                             # 256 bytes
    return store


def test_streams_that_fit_all_run():
    gate = StreamGate(Held(1000), per_token=1.0, work=10, budget=1000 + 10 + 3 * 100, horizon=100)
    plan = gate.plan([("a", 50, 500), ("b", 50, 500), ("c", 50, 500)])
    assert plan.run == ["a", "b", "c"] and not plan.paused and not plan.ended


def test_a_stream_near_its_end_holds_only_what_it_can_still_grow():
    gate = StreamGate(Held(1000), per_token=1.0, work=0, budget=1000 + 100 + 20, horizon=100)
    assert gate.plan([("a", 50, 500), ("b", 480, 500)]).run == ["a", "b"]       # b grows by 20 at most


def test_freed_buffers_and_retained_prefixes_go_before_any_stream_waits():
    held = Held(1000, cache=50, store=prefix_store())
    gate = StreamGate(held, per_token=1.0, work=0, budget=1000 + 2 * 100, horizon=100)
    assert gate.plan([("a", 0, 500), ("b", 0, 500)]).run == ["a", "b"]
    assert held.cache == 0 and len(held.store) == 0


def test_a_prefix_that_could_not_make_room_for_every_stream_goes_only_once_one_waits():
    held = Held(1000, store=prefix_store())
    gate = StreamGate(held, per_token=1.0, work=0, budget=1000 + 150, horizon=100)
    plan = gate.plan([("a", 0, 500), ("b", 0, 500)])       # both: 1,200 even with the prefix gone; a alone: 1,100
    assert plan.run == ["a"] and plan.paused == ["b"] and len(held.store) == 0
    held = Held(1000, store=prefix_store())
    gate = StreamGate(held, per_token=1.0, work=0, budget=1000 + 50, horizon=100)
    plan = gate.plan([("a", 0, 500)])                      # a lone stream that can't fit keeps the prefixes too
    assert plan.run == ["a"] and len(held.store) == 1


def test_the_newest_waits_and_a_lone_stream_always_runs():
    gate = StreamGate(Held(1000), per_token=1.0, work=0, budget=1000 + 150, horizon=100)
    plan = gate.plan([("a", 0, 500), ("b", 0, 500), ("c", 0, 500)])
    assert plan.run == ["a"] and plan.paused == ["b", "c"] and not plan.ended
    gate.budget = 900
    assert gate.plan([("a", 0, 500)]).run == ["a"]


def test_a_round_of_fewer_streams_needs_its_share_of_the_working_memory():
    gate = StreamGate(Held(1000), per_token=1.0, work=400, budget=1000 + 2 * 100 + 200, horizon=100, lanes=4)
    plan = gate.plan([("a", 0, 500), ("b", 0, 500), ("c", 0, 500), ("d", 0, 500)])
    assert plan.run == ["a", "b"] and plan.paused == ["c", "d"] and not plan.ended   # 1,400 of 1,400: two run


def test_when_even_the_oldest_cannot_grow_the_newest_ends():
    gate = StreamGate(Held(1000), per_token=1.0, work=0, budget=1050, horizon=100)
    plan = gate.plan([("a", 0, 500), ("b", 0, 500), ("c", 0, 500)])
    assert plan.run == ["a"] and plan.paused == ["b"] and plan.ended == ["c"]


# -- a scheduler's streams growing into a small budget ------------------------------------------------------------
class Layers(SizedFamily):
    """Eight layers of the fake 1 KiB-a-position KV, sharing one history: 8 KiB a token, in 256-token steps."""

    def make_cache(self) -> list[Any]:
        layers = [SizedKV() for _ in range(8)]
        for layer in layers[1:]:
            layer.rows = layers[0].rows
        return layers


def served(budget: int, lanes: int = 3):
    """A scheduler whose prompt memory and admission count the fake caches' real growth."""

    gc.collect()
    model = Layers()
    model.args = type("Args", (), {"num_attention_heads": 1, "head_dim": 128})()
    runtime = Runtime(MIB)
    base = runtime.get_active_memory()                  # caches other tests left alive count as resident
    memory = PromptMemory(base + budget, model, runtime=runtime, overhead_bytes=0, bootstrap_bytes=0,
                          chunk_rows=256)
    probe = model.make_cache()
    probe[0].rows[0].extend(range(256))
    memory.observe_cache(probe, workspace=False)
    streams = StreamMemory(64, 256 * 8 * KIB, 320, 512 * 8 * KIB, 8.0 * KIB, 0.0, 0.0, 0)
    admission = Admission(base + budget, streams, used=memory.held)
    return Scheduler(FakeEngine(model), lanes=lanes, eos_ids=frozenset(), admission=admission,
                     prompt_memory=memory)


def run(scheduler: Scheduler, work: list[ChatJob]) -> dict[str, list[int]]:
    for job in work:
        scheduler.submit(job)
    scheduler.start()
    try:
        assert all(job.done.wait(60) for job in work)
    finally:
        scheduler.stop()
    out = {}
    for job in work:
        tokens = []
        while (chunk := job.chunks.get_nowait()) is not None:
            tokens += chunk
        out[job.job_id] = tokens
    return out


def jobs(count: int, prompt: int, reply: int) -> list[ChatJob]:
    return [ChatJob(job_id=f"j{i}", prompt_ids=[1 + (7 * i + k) % 97 for k in range(prompt)], max_tokens=reply,
                    temperature=0.0) for i in range(count)]


def test_a_stream_beside_others_holds_its_next_horizon_not_its_whole_reply():
    scheduler = served(budget=48 * MIB)
    first, second = jobs(2, 300, 6000)
    scheduler._admit()
    scheduler.submit(first)
    scheduler._admit()
    while scheduler._filling is not None:
        scheduler._fill()
    assert scheduler._fits(second)                      # 300 + 2,048 tokens held, the other stream at its length
    gate, scheduler.gate = scheduler.gate, None
    assert not scheduler._fits(second)                  # the whole 6,300-token reply: 49 MiB, past the budget
    scheduler.gate = gate


def test_streams_admitted_by_use_take_turns_past_the_budget_and_each_equals_its_solo_run():
    work = jobs(3, 300, 3000)
    scheduler = served(budget=48 * MIB)                 # each reaches 3,300 tokens: 26 MiB; three never fit at once
    got = run(scheduler, work)
    for job in work:
        assert job.error is None
        assert got[job.job_id] == fake_serial(job.prompt_ids, job.max_tokens, set())     # streams == solo
    assert scheduler.gate.waits > 0 and scheduler.gate.ends == 0


def test_the_newest_stream_ends_with_a_clear_error_when_even_the_oldest_cannot_grow():
    work = jobs(2, 300, 6000)
    scheduler = served(budget=44 * MIB, lanes=2)        # both admitted; 6,300 tokens is 50 MiB: they can't both finish
    got = run(scheduler, work)
    ended = [job for job in work if job.error is not None]
    assert len(ended) == 1 and isinstance(ended[0].error, RequestError)
    assert "ran out of memory" in str(ended[0].error)
    for job in work:                                     # the survivor as it would run alone, the ended one a prefix
        solo = fake_serial(job.prompt_ids, job.max_tokens, set())
        assert got[job.job_id] == solo[:len(got[job.job_id])] and (job.error is not None or got[job.job_id] == solo)
    assert scheduler.gate.ends == 1
