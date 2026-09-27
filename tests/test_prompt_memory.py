"""Admission and prefill guards without models or Metal buffers."""

from types import SimpleNamespace

import pytest

from tensorfold.server.app import ChatJob, CheckpointStore, Scheduler
from tensorfold.server.memory_budget import cache_nbytes
from tensorfold.server.http import RequestError
from tensorfold.server.prompt_memory import PromptMemory, attention_geometry
from tests.test_memory_budget import Array
from tests.lane_fakes import FakeEngine


def test_prefill_guard_can_refuse_after_one_existing_chunk():
    engine = FakeEngine()
    engine.prefill_step = 4
    calls = []
    forward = engine.model.hidden

    def counted(rows, cache, parents=None):
        calls.append(int(rows.size))
        return forward(rows, cache, parents)

    def refuse(cache, rows):
        raise RequestError("too large")

    engine.model.hidden = counted
    engine.prefill_guard = SimpleNamespace(before_chunk=lambda cache, rows: None, after_chunk=refuse)
    with pytest.raises(RequestError, match="too large"):
        engine._family_feed(list(range(12)), engine.model.make_cache(), engine.prompt_chunks(range(12)).between(0, 12))
    assert calls == [4]


class Runtime:
    def __init__(self, resident=1000):
        self.resident = resident
        self.caches = []
        self.cache = 0
        self.peak = resident

    def get_active_memory(self):
        return self.resident + sum(cache_nbytes(cache) for cache in self.caches)

    def get_cache_memory(self):
        return self.cache

    def get_peak_memory(self):
        return max(self.peak, self.get_active_memory())

    def reset_peak_memory(self):
        self.peak = self.get_active_memory()

    def clear_cache(self):
        self.cache = 0


def populated(tokens=256):
    keys, values = Array((1, 1, tokens, 1)), Array((1, 1, tokens, 1))
    return [SimpleNamespace(keys=keys, values=values, state=(keys, values), offset=tokens)]


def controller(budget=1_000_000, store=None, runtime=None):
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1))
    return PromptMemory(budget, model, runtime=runtime or Runtime(), store=store, window_tokens=8192,
                        overhead_bytes=0, bootstrap_bytes=0)


def test_large_first_request_refuses_after_bounded_chunk_and_names_a_fitting_prompt():
    runtime = Runtime()
    memory = controller(runtime=runtime)
    memory.begin(4096, 64)
    cache = populated()
    runtime.caches.append(cache)
    with pytest.raises(RequestError, match="fits up to") as error:
        memory.after_chunk(cache, 256)
    import re

    fit = int(re.search(r"fits up to ([\d,]+) tokens", str(error.value)).group(1).replace(",", ""))
    assert 0 < fit < 4096
    assert memory.projected(fit, current_cache=cache) <= memory.budget
    assert memory.projected(fit + 1, current_cache=cache) > memory.budget
    memory.end()
    runtime.caches.clear()
    with pytest.raises(RequestError):
        memory.begin(4096, 64)


def test_checkpoint_copy_is_suppressed_before_allocation_when_it_exceeds_store_budget():
    cache = populated()
    store = CheckpointStore(3, copier=lambda cache: pytest.fail("oversized copy allocated"),
                            budget_bytes=512, sizer=cache_nbytes)
    memory = controller(store=store)
    memory.begin(512, 64)
    memory.observe_cache(cache)
    assert not memory.allow_checkpoint(cache)


def test_retained_prefixes_are_evicted_before_refusing_the_next_request():
    cache = populated()
    runtime = Runtime()
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=4096, sizer=cache_nbytes)
    store.insert([1], cache, last_prompt=[1], pinned=True)
    store.insert([2], cache, last_prompt=[2])
    memory = controller(budget=1200, store=store, runtime=runtime)
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    with pytest.raises(RequestError):
        memory.begin(512, 64)
    assert not len(store)
    assert store.evictions == 2


def test_freed_cache_and_nonmlx_footprint_are_reserved():
    runtime = Runtime(resident=800)
    runtime.cache = 300
    memory = PromptMemory(1100, None, runtime=runtime, overhead_bytes=200, bootstrap_bytes=0)
    memory.begin(32, 4)
    assert memory.budget == 900
    assert runtime.cache == 0
    runtime.resident = 901
    with pytest.raises(RequestError, match="process overhead"):
        memory.begin(32, 4)


def test_workspace_profile_survives_freed_buffers_and_released_probe_cache():
    runtime = Runtime(resident=1000)
    cache = populated()
    runtime.caches.append(cache)
    runtime.peak = runtime.get_active_memory() + 400
    runtime.cache = 500
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(5300, model, runtime=runtime, window_tokens=8192,
                          overhead_bytes=0, bootstrap_bytes=0)
    memory.observe_cache(cache)
    assert memory.observed_work == 400
    runtime.clear_cache()
    runtime.caches.clear()
    with pytest.raises(RequestError, match="fits up to"):
        memory.begin(512, 0)


def test_retained_cache_metadata_does_not_suppress_first_real_workspace_profile():
    runtime = Runtime()
    cache = populated()
    runtime.caches.append(cache)
    store = SimpleNamespace(_entries=[SimpleNamespace(cache=cache)])
    memory = controller(store=store, runtime=runtime)
    memory.begin(512, 0)
    assert memory.profile is not None
    assert not memory.workspace_profiled
    runtime.peak = runtime.get_active_memory() + 400
    peak = runtime.get_peak_memory()
    memory.after_chunk(cache, 256)
    assert memory.workspace_profiled and memory.observed_work == 400
    assert runtime.get_peak_memory() == peak
    runtime.peak += 1000
    memory.after_chunk(cache, 256)
    assert memory.observed_work == 400


@pytest.mark.parametrize("seeded", [False, True])
def test_unknown_geometry_keeps_refusing_without_poisoning_a_cache_profile(seeded):
    memory = PromptMemory(100000, None, runtime=Runtime(), overhead_bytes=0, bootstrap_bytes=0)
    if seeded:
        memory.observe_cache([], workspace=False)
    previous = memory.profile
    for _ in range(3):
        with pytest.raises(RequestError, match="attention workspace"):
            memory.after_chunk(populated(), 256)
        assert memory.profile is previous
        assert not memory.workspace_profiled


def test_health_reset_waits_for_in_progress_workspace_capture():
    from threading import Event, Thread

    runtime = Runtime()
    cache = populated()
    runtime.caches.append(cache)
    memory = controller(runtime=runtime)
    memory.begin(512, 0)
    runtime.peak = runtime.get_active_memory() + 400
    entered, release, health_started, health_done = (Event() for _ in range(4))
    reads, errors = [], []

    def peak():
        value = max(runtime.peak, runtime.get_active_memory())
        if not entered.is_set():
            entered.set()
            assert release.wait(2)
        return value

    runtime.get_peak_memory = peak

    def observe():
        try:
            memory.after_chunk(cache, 256)
        except Exception as exc:
            errors.append(exc)

    def health():
        health_started.set()
        reads.append(memory.memory_snapshot(True))
        health_done.set()

    observer, reader = Thread(target=observe), Thread(target=health)
    observer.start()
    try:
        assert entered.wait(2)
        reader.start()
        assert health_started.wait(2)
        assert not health_done.wait(0.02)
    finally:
        release.set()
        observer.join(2)
        if reader.ident is not None:
            reader.join(2)
    assert not errors and health_done.is_set()
    assert memory.observed_work == 400
    assert reads[0]["peak"] == reads[0]["active"] + 400
    assert runtime.peak == runtime.get_active_memory()


def test_geometry_uses_configuration_and_kernel_parts_without_loading_arrays():
    class Attention:
        q_proj, k_proj, heads, split_rows = object(), object(), 24, 256

    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=24), layers=[Attention()])
    assert attention_geometry(model) == (24, 256)


@pytest.mark.parametrize("dim", [64, 80, 128])
def test_confirmed_fused_head_sizes_do_not_reserve_materialized_scores(dim):
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=32, head_dim=dim))
    assert attention_geometry(model) == (0, 144)
    memory = PromptMemory(1_000_000, model, runtime=Runtime(), overhead_bytes=0, bootstrap_bytes=0)
    memory.begin(8192, 64)
    memory.after_chunk(populated(), 256)
    assert memory.projected(8192) < 100_000


def test_fallback_head_size_and_unknown_configuration_keep_score_reservations():
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=24, head_dim=256))
    assert attention_geometry(model) == (24, 144)
    assert attention_geometry(None) == (-1, 144)


def test_scheduler_refuses_before_copying_a_retained_prefix_or_running_prefill():
    cache = populated()
    store = CheckpointStore(3, copier=lambda cache: pytest.fail("prefix copied before admission"),
                            budget_bytes=4096, sizer=cache_nbytes)
    store.insert([1], cache, last_prompt=[1])
    memory = controller(budget=1200, store=store)
    engine = FakeEngine()
    scheduler = Scheduler(engine, lanes=1, eos_ids=frozenset(), checkpoints=store, prompt_memory=memory)
    job = ChatJob("large", [1] * 4096, 64, 0.0)
    scheduler._start_job(job)
    assert isinstance(job.error, RequestError)
    assert job.done.is_set()
    assert engine.prefill_calls == []
    assert engine.prefill_guard is None


def test_oversized_disk_snapshot_is_skipped_before_tensor_load(monkeypatch, tmp_path):
    from tensorfold.engine import prefix_snapshots

    path = tmp_path / "prefix.safetensors"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(prefix_snapshots, "read_metadata", lambda path: {"model": "fixture"})
    monkeypatch.setattr(prefix_snapshots, "load_snapshot", lambda *args: pytest.fail("snapshot allocated"))
    assert list(prefix_snapshots.load_snapshots(tmp_path, "fixture", allow=lambda path: False)) == []
