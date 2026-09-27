"""A long conversation keeps and resumes its newest prefix under memory admission."""

from types import SimpleNamespace

import pytest

from tensorfold.server.app import ChatJob, CheckpointStore, Scheduler
from tensorfold.server.http import RequestError
from tensorfold.server.memory_budget import cache_nbytes
from tensorfold.server.prompt_memory import PromptMemory
from tests.lane_fakes import FakeBatchItem, FakeEngine
from tests.test_memory_budget import Array
from tests.test_prompt_memory import Runtime, controller, populated


class SizedItem(FakeBatchItem):
    """A fake cache layer: its history rows plus ``nbytes`` of arrays."""

    def __init__(self, rows, nbytes):
        super().__init__(rows)
        self.buffer = Array((nbytes,), element_bytes=1)


def sized(tokens, nbytes):
    return [SizedItem([list(tokens)], nbytes)]


def refuse_copy(cache):
    pytest.fail("the resumed prefix was copied")


def stored(store):
    return [entry.tokens for entry in store._entries]


def test_an_oversized_newest_prefix_is_kept_beside_the_pinned_blocks():
    store = CheckpointStore(3, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: c[0])
    store.admit_oversize = True
    store.insert([9], [100], last_prompt=[9], pinned=True)
    store.insert([1], [300], last_prompt=[1])
    store.insert([2], [1200], last_prompt=[2])
    assert store.match([2, 5]) == (1, [1200], [2])
    assert store.match([9, 5]) == (1, [100], [9])
    assert store.match([1, 5]) is None
    assert store.nbytes == 1300


def test_an_oversized_checkpoint_is_copied_when_memory_admits_it():
    cache = populated()
    store = CheckpointStore(3, copier=lambda c: c, budget_bytes=16, sizer=cache_nbytes)
    store.admit_oversize = True
    memory = controller(store=store)
    memory.begin(256, 64)
    memory.observe_cache(cache)
    assert memory.allow_checkpoint(cache)


def test_an_oversized_checkpoint_is_suppressed_when_memory_cannot_hold_it():
    cache = populated()
    store = CheckpointStore(3, copier=lambda c: c, budget_bytes=16, sizer=cache_nbytes)
    store.admit_oversize = True
    runtime = Runtime()
    memory = controller(budget=runtime.resident + 2 * cache_nbytes(cache), store=store, runtime=runtime)
    memory.begin(256, 64)
    memory.observe_cache(cache)
    runtime.caches.append(cache)
    assert not memory.allow_checkpoint(cache)


def scheduler_with(store, budget):
    """A scheduler whose MLX memory is 1,000 resident bytes plus the retained prefixes."""

    runtime = Runtime()
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    fused = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(budget, fused, runtime=runtime, store=store, window_tokens=8192,
                          overhead_bytes=0, bootstrap_bytes=0)
    store.admit_oversize = True
    engine = FakeEngine()
    return Scheduler(engine, lanes=1, eos_ids=frozenset(), checkpoints=store, prompt_memory=memory), engine


def test_admission_never_evicts_the_prefix_a_prompt_resumes():
    prompt = [3] * 64
    store = CheckpointStore(3, copier=refuse_copy, sizer=cache_nbytes)
    store.insert([7] * 8, sized([7] * 8, 400), last_prompt=[7] * 8)
    store.insert(prompt[:48], sized(prompt[:48], 400), last_prompt=prompt[:48])
    # resuming needs 1,800 bytes once the other prefix goes; a copy beside the stored arrays needs 2,200
    scheduler, engine = scheduler_with(store, budget=2000)
    job = ChatJob("resume", prompt, 4, 0.0)
    scheduler._start_job(job)
    assert job.error is None
    assert engine.prefill_calls == [("resume", 48)]
    assert [7] * 8 not in stored(store)
    assert prompt[:48] not in stored(store)


def test_a_resumed_prefix_is_copied_and_kept_when_memory_allows():
    prompt = [3] * 64
    copies = []
    store = CheckpointStore(3, copier=lambda c: copies.append(c) or c, sizer=cache_nbytes)
    store.insert(prompt[:48], sized(prompt[:48], 400), last_prompt=prompt[:48])
    scheduler, engine = scheduler_with(store, budget=1_000_000)
    job = ChatJob("resume", prompt, 4, 0.0)
    scheduler._start_job(job)
    assert job.error is None
    assert engine.prefill_calls == [("resume", 48)]
    assert len(copies) == 1
    assert prompt[:48] in stored(store)


def test_a_refused_prompt_keeps_the_prefix_it_would_resume():
    prompt = [3] * 4096
    store = CheckpointStore(3, copier=refuse_copy, sizer=cache_nbytes)
    store.insert(prompt[:48], sized(prompt[:48], 400), last_prompt=prompt[:48])
    scheduler, engine = scheduler_with(store, budget=1200)
    scheduler.prompt_memory.observe_cache(populated(), workspace=False)
    job = ChatJob("large", prompt, 64, 0.0)
    scheduler._start_job(job)
    assert isinstance(job.error, RequestError)
    assert engine.prefill_calls == []
    assert stored(store) == [prompt[:48]]


def test_prefixes_freed_by_admission_do_not_count_as_prefill_workspace():
    runtime = Runtime()
    store = CheckpointStore(3, copier=lambda c: c, sizer=cache_nbytes)
    store.insert([7] * 8, sized([7] * 8, 4000), last_prompt=[7] * 8)
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    fused = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(1_000_000, fused, runtime=runtime, store=store, overhead_bytes=0, bootstrap_bytes=0)
    memory.begin(64, 4)
    store.evict_one()                          # admission frees a prefix after the request began
    cache = sized([3] * 64, 400)
    memory.before_chunk(cache, 64)
    memory.after_chunk(cache, 64)
    assert memory.observed_work == 0


def test_a_prompt_that_fits_only_alone_waits_for_the_running_stream_instead_of_being_refused():
    from tensorfold.engine.lane_engine import LaneStream

    store = CheckpointStore(3, copier=lambda c: c, sizer=cache_nbytes)
    scheduler, engine = scheduler_with(store, budget=2000)
    scheduler.lanes = 2
    running = LaneStream("running", [1, 2], 8, eos_ids=frozenset({-1}))
    engine.add_stream(running)
    runtime = scheduler.prompt_memory.runtime
    runtime.resident = 1900                        # the running stream's caches hold the memory
    job = ChatJob("waits", [3] * 64, 4, 0.0)
    scheduler.prompt_memory.observe_cache(sized([3] * 8, 400), workspace=False)
    scheduler.submit(job)
    scheduler._admit()
    assert scheduler._held is job and job.error is None and not job.done.is_set()
