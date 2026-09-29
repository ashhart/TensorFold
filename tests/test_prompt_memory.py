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
                        overhead_bytes=0, bootstrap_bytes=0, chunk_rows=256)


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


def retained_prefix_memory(budget):
    """An in-flight cache and an unrelated retained prefix compete for the same MLX budget."""

    work = populated(64)
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=1 << 20, sizer=cache_nbytes)
    store.insert([1], populated(64), last_prompt=[1])
    runtime = Runtime()
    runtime.cache = 128
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes + cache_nbytes(work)
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(budget, model, runtime=runtime, store=store, overhead_bytes=0,
                          bootstrap_bytes=0)
    memory.begin(64, 0, admit=False)
    memory.observe_cache(work, workspace=False)
    return memory, runtime, store, work


def test_impossible_checkpoint_copy_preserves_retained_prefixes():
    memory, runtime, store, work = retained_prefix_memory(3200)
    size = cache_nbytes(work)
    assert memory.projected(64, current_cache=work, extra_bytes=size) - runtime.cache - store.nbytes > memory.budget

    assert not memory.allow_checkpoint(work)
    assert store.match([1, 2]) is not None
    assert store.evictions == 0
    assert runtime.cache == 128


def test_impossible_snapshot_load_preserves_retained_prefixes():
    memory, runtime, store, work = retained_prefix_memory(3500)
    size = cache_nbytes(work)
    assert memory.projected(64, extra_bytes=size) - runtime.cache - store.nbytes > memory.budget

    assert not memory.allow_load(size)
    assert store.match([1, 2]) is not None
    assert store.evictions == 0
    assert runtime.cache == 128


@pytest.mark.parametrize("kind,budget", [("checkpoint", 3400), ("load", 3600)])
def test_reclaim_still_admits_a_copy_that_can_fit(kind, budget):
    memory, runtime, store, work = retained_prefix_memory(budget)
    size = cache_nbytes(work)
    assert memory.projected(64, current_cache=work if kind == "checkpoint" else None,
                            extra_bytes=size) > memory.budget

    allowed = memory.allow_checkpoint(work) if kind == "checkpoint" else memory.allow_load(size)
    assert allowed
    assert runtime.cache == 0
    assert store.evictions == 1


def shelf_memory(*, store=True, entries=(), cache=0):
    """A 64-token working cache beside retained prefixes of ``entries`` tokens (oldest first) and ``cache`` freed bytes."""

    work = populated(64)
    shelf = CheckpointStore(8, copier=lambda c: c, budget_bytes=1 << 30, sizer=cache_nbytes) if store else None
    for i, tokens in enumerate(entries):
        shelf.insert([100 + i], populated(tokens), last_prompt=[100 + i])
    runtime = Runtime()
    runtime.cache = cache
    runtime.get_active_memory = lambda: runtime.resident + (shelf.nbytes if shelf else 0) + cache_nbytes(work)
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(1 << 40, model, runtime=runtime, store=shelf, overhead_bytes=0, bootstrap_bytes=0)
    memory.begin(64, 0, admit=False)
    memory.observe_cache(work, workspace=False)
    return memory, runtime, shelf, work


def ask(memory, kind, work):
    """The projected need of a checkpoint copy of ``work`` or a snapshot load of its size, and the call that asks for it."""

    size = cache_nbytes(work)
    if kind == "checkpoint":
        return memory.projected(64, current_cache=work, extra_bytes=size), lambda: memory.allow_checkpoint(work)
    return memory.projected(64, extra_bytes=size), lambda: memory.allow_load(size)


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
def test_a_copy_or_load_exactly_at_the_budget_frees_nothing(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 64), cache=100)
    memory.budget, call = ask(memory, kind, work)
    assert call() and runtime.cache == 100 and shelf.evictions == 0 and len(shelf) == 2


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
def test_freed_buffers_go_before_any_prefix(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 64), cache=100)
    need, call = ask(memory, kind, work)
    memory.budget = need - 100
    assert call() and runtime.cache == 0 and shelf.evictions == 0 and len(shelf) == 2


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
def test_without_freed_buffers_only_the_oldest_prefixes_needed_go(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64))
    oldest, middle = cache_nbytes(populated(128)), cache_nbytes(populated(96))
    need, call = ask(memory, kind, work)
    memory.budget = need - oldest
    assert call() and shelf.evictions == 1 and [e.tokens for e in shelf._entries] == [[102], [101]]
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64))
    memory.budget = need - oldest - middle
    assert ask(memory, kind, work)[1]() and shelf.evictions == 2 and [e.tokens for e in shelf._entries] == [[102]]


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
@pytest.mark.parametrize("cache", [0, 100])
def test_an_impossible_copy_or_load_keeps_every_prefix_and_freed_buffer(kind, cache):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64), cache=cache)
    need, call = ask(memory, kind, work)
    memory.budget = need - cache - shelf.nbytes - 1
    assert not call() and runtime.cache == cache and shelf.evictions == 0 and len(shelf) == 3
    memory.budget += 1                                            # everything freed is exactly enough
    assert call() and runtime.cache == 0 and len(shelf) == 0


@pytest.mark.parametrize("store", [False, True])
def test_with_no_store_or_an_empty_one_only_freed_buffers_make_room(store):
    memory, runtime, shelf, work = shelf_memory(store=store, cache=100)
    if not store:
        assert not memory.allow_checkpoint(work) and runtime.cache == 100    # nowhere to keep a copy
    for kind in ["checkpoint", "load"] if store else ["load"]:
        need, call = ask(memory, kind, work)
        memory.budget = need - 101
        assert not call() and runtime.cache == 100
        memory.budget = need - 100
        assert call() and runtime.cache == 0
        runtime.cache = 100


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
def test_an_eviction_that_frees_less_than_its_size_stops_the_reclaim(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64))
    oldest, active = shelf._entries[-1], runtime.get_active_memory        # first to go; something else holds its arrays
    runtime.get_active_memory = lambda: active() + (0 if oldest in shelf._entries else cache_nbytes(oldest.cache))
    need, call = ask(memory, kind, work)
    memory.budget = need - shelf.nbytes                            # enough on paper, not once the oldest frees nothing
    assert not call() and shelf.evictions == 1 and [e.tokens for e in shelf._entries] == [[102], [101]]


@pytest.mark.parametrize("kind", ["checkpoint", "load"])
def test_the_store_budget_refuses_an_oversized_copy_or_load_before_reclaiming(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(64,), cache=100)
    shelf.budget_bytes = cache_nbytes(work) - 1
    need, call = ask(memory, kind, work)
    memory.budget = need - 100
    assert not call() and runtime.cache == 100 and shelf.evictions == 0
    shelf.admit_oversize = True                                    # a memory controller evicts on demand instead
    assert call() and runtime.cache == 0


def test_a_request_that_cannot_fit_even_with_the_store_empty_is_refused_without_evicting():
    cache = populated()
    runtime = Runtime()
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=4096, sizer=cache_nbytes)
    store.insert([1], cache, last_prompt=[1], pinned=True)
    store.insert([2], cache, last_prompt=[2])
    memory = controller(budget=1200, store=store, runtime=runtime)
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    with pytest.raises(RequestError):
        memory.begin(512, 64)
    assert len(store) == 2 and store.evictions == 0          # the other conversations keep their prefixes


def admit(memory, kind, work, shelf):
    """The projected need of admitting the prompt as ``kind`` asks, what could be freed for it, and the call."""

    if kind == "workspace":
        def call():
            try:
                memory.require_workspace(500)
            except RequestError:
                return False
            return True
        return memory.projected(64, extra_bytes=500), shelf.nbytes, call
    if kind == "resume":                                          # the newest prefix resumed: kept, never evicted
        keep = shelf._entries[0]
        return (memory.projected(64, current_cache=keep.cache), shelf.nbytes - keep.nbytes,
                lambda: memory.fits(keep.cache, keep=keep))
    current = work if kind == "chunk" else None                   # a prefill chunk's check, or a request's start
    return memory.projected(64, current_cache=current), shelf.nbytes, lambda: memory.fits(current)


@pytest.mark.parametrize("kind", ["start", "chunk", "resume", "workspace"])
def test_an_impossible_prompt_keeps_every_prefix_and_freed_buffer(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64), cache=100)
    need, stored, call = admit(memory, kind, work, shelf)
    memory.budget = need - 100 - stored - 1
    assert not call() and runtime.cache == 100 and shelf.evictions == 0 and len(shelf) == 3
    memory.budget += 1                                            # everything that can go is exactly enough
    assert call() and runtime.cache == 0 and len(shelf) == (1 if kind == "resume" else 0)


@pytest.mark.parametrize("kind", ["start", "chunk", "resume", "workspace"])
def test_a_prompt_that_fits_once_the_oldest_prefix_goes_evicts_only_it(kind):
    memory, runtime, shelf, work = shelf_memory(entries=(128, 96, 64), cache=100)
    need, stored, call = admit(memory, kind, work, shelf)
    memory.budget = need - 100 - cache_nbytes(populated(128))
    assert call() and runtime.cache == 0 and shelf.evictions == 1
    assert [e.tokens for e in shelf._entries] == [[102], [101]]


def test_freed_cache_and_nonmlx_footprint_are_reserved():
    runtime = Runtime(resident=800)
    runtime.cache = 300
    memory = PromptMemory(1100, None, runtime=runtime, overhead_bytes=200, bootstrap_bytes=0)
    memory.begin(32, 4)
    assert memory.budget == 900
    assert runtime.cache == 0
    runtime.resident = 901
    with pytest.raises(RequestError, match=r"of the 0\.0 GiB .* rest of the process") as error:
        memory.begin(32, 4)
    assert "--prompt-cache-gib 0" not in str(error.value)     # retained prefixes were evicted before refusing
    assert "reserved in full" in str(error.value) and "smaller --context" in str(error.value)


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


def test_a_short_first_chunk_does_not_fix_the_workspace_a_full_chunk_needs():
    runtime = Runtime()
    cache = populated()
    runtime.caches.append(cache)
    memory = controller(runtime=runtime)
    memory.begin(64, 0)
    memory.before_chunk(None, 64)
    runtime.peak = runtime.get_active_memory() + 100
    memory.after_chunk(cache, 64)
    assert memory.observed_work == 100 and not memory.workspace_profiled
    memory.end()
    memory.begin(512, 0)
    memory.before_chunk(cache, 256)
    assert runtime.get_peak_memory() == runtime.get_active_memory()      # the full chunk's peak starts here
    runtime.peak = runtime.get_active_memory() + 400
    memory.after_chunk(cache, 256)
    assert memory.observed_work == 400 and memory.workspace_profiled


class ProbeEngine:
    """Feeds a prompt in chunks through the prefill guard; each chunk peaks 400 bytes above what stays active."""

    prefill_step = prefill_align = 256
    prefill_guard = None

    def __init__(self, runtime, carry=0):
        self.runtime = runtime
        self.probes, self.fed = [], []
        self.carry = carry                    # held between chunks outside the cache (the last hidden states, taps)

    def prefill_prefix(self, tokens, *, cache=None, cached_tokens=0):
        self.probes.append(len(tokens))
        self.fed = list(tokens)
        done, cache = 0, None
        while done < len(tokens):
            rows = min(self.prefill_step, len(tokens) - done)
            self.prefill_guard.before_chunk(cache, rows)
            done += rows
            cache = populated(-(-done // 256) * 256)
            self.runtime.caches[:] = [cache]
            self.runtime.resident += self.carry if done == rows else 0
            self.runtime.peak = self.runtime.get_active_memory() + 400
            self.prefill_guard.after_chunk(cache, rows)
        self.runtime.caches.clear()
        self.runtime.resident -= self.carry
        self.runtime.cache += 1000            # the probe's buffers, freed
        return cache


def test_profile_probe_sizes_a_full_chunk_before_any_request_and_releases_it():
    runtime = Runtime()
    memory = controller(runtime=runtime)
    engine = ProbeEngine(runtime)
    memory.profile_probe(engine)
    assert engine.probes == [256 + 64]
    assert memory.workspace_profiled and memory.observed_work == 400
    assert memory.profile.bytes_per_token == 4 and memory.profile.step == 256
    assert engine.prefill_guard is None and runtime.cache == 0


def test_largest_window_releases_retained_prefixes_and_freed_buffers_but_nothing_live():
    runtime = Runtime(resident=1000)
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=1 << 20, sizer=cache_nbytes)
    memory = controller(budget=200_000, store=store, runtime=runtime)
    memory.profile_probe(ProbeEngine(runtime))
    empty = memory.largest_window(8192)

    def total(tokens):
        return runtime.resident + memory.profile.cache_bytes(tokens) + memory._work(tokens)

    assert 0 < empty < 8192
    assert total(empty) <= memory.budget < total(empty + 1)
    store.insert([1], populated(4096), last_prompt=[1], pinned=True)
    store.insert([2], populated(1024), last_prompt=[2])
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    runtime.cache = 50_000
    assert memory.largest_window(8192) == empty
    assert memory.largest_window(100) == 100
    runtime.resident += 100_000               # a live stream's memory is not released
    assert memory.largest_window(8192) < empty
    runtime.resident += 10**9
    assert memory.largest_window(8192) == 0


def test_growth_is_priced_per_entry_in_flight():
    from tensorfold.server.memory_budget import CacheMemory

    memory = controller()
    memory.profile = CacheMemory(1000, 16 * 64, 256, 64)          # sixteen entries of 64 bytes a position
    memory.observed_work = 500
    assert memory._work(4096) == 500 + 1000 + 4096 * 2 * 64 + 2 * 144 * 1 * 4096 * 2


def test_the_default_window_keeps_room_to_retain_the_prompt_for_the_next_turn():
    runtime = Runtime(resident=1000)
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=1 << 20, sizer=cache_nbytes)
    memory = controller(budget=200_000, store=store, runtime=runtime)
    memory.profile_probe(ProbeEngine(runtime))
    bare, resumable = memory.largest_window(8192), memory.largest_window(8192, resumable=True)

    def total(tokens, kept):
        return (runtime.resident + (1 + kept) * memory.profile.cache_bytes(tokens) + memory._work(tokens))

    assert 0 < resumable < bare
    assert total(resumable, 1) <= memory.budget < total(resumable + 1, 1)
    assert memory.fit_window(8192, True) == (resumable // 1024 * 1024 if resumable >= 1024 else resumable, True)
    assert memory.fit_window(bare, False) == (bare, False)          # an explicit window up to the bare most fits
    with pytest.raises(ValueError, match="does not fit"):
        memory.fit_window(bare + 1, False)


def test_concurrent_admission_counts_freed_buffers_and_retained_prefixes_as_reclaimable():
    from tensorfold.engine.memory import Admission, StreamMemory

    gib = 1024**3
    runtime = Runtime(resident=19 * gib)          # the second pass's 48 GB state: weights, a live stream
    runtime.cache = 6 * gib                       # freed buffers MLX keeps for reuse
    store = CheckpointStore(3, copier=lambda cache: cache, budget_bytes=8 * gib, sizer=lambda cache: 3 * gib)
    store.insert([1], [object()], last_prompt=[1])
    runtime.get_active_memory = lambda: runtime.resident + store.nbytes
    memory = PromptMemory(int(0.70 * 48 * gib), None, runtime=runtime, store=store)
    stream = StreamMemory(64, 200 * 1024**2, 2112, 330 * 1024**2, 64 * 1024, 20_000.0, 2.0, gib // 2)
    admission = Admission(memory.budget, stream, used=memory.held)
    scheduler = Scheduler(SimpleNamespace(active_count=1), lanes=8, eos_ids=frozenset(), admission=admission,
                          prompt_memory=memory)
    job = ChatJob(job_id="second", prompt_ids=[1] * 8_000, max_tokens=4_096, temperature=0.0)
    assert memory.held() == 19 * gib
    assert memory.would_fit(8_000, 4_096) and scheduler._fits(job)


def test_the_window_counts_what_a_request_holds_between_chunks_and_the_probe_reads_real_text():
    runtime = Runtime(resident=1000)
    memory = controller(budget=200_000, runtime=runtime)
    engine = ProbeEngine(runtime, carry=3000)
    text = list(range(7, 7 + 100))
    memory.profile_probe(engine, tokens=text)
    assert engine.fed[:100] == text and len(engine.fed) == 256 + 64       # the text, repeated to a full chunk
    assert memory.carry == 3000 and runtime.resident == 1000
    window = memory.largest_window(8192)

    def total(tokens):
        return runtime.resident + memory.carry + memory.profile.cache_bytes(tokens) + memory._work(tokens)

    assert total(window) <= memory.budget < total(window + 1)


def test_probe_text_is_this_modules_source_through_the_models_tokenizer():
    from tensorfold.server.prompt_memory import probe_tokens

    seen = []
    tokens = probe_tokens(SimpleNamespace(encode=lambda text, add_special_tokens: seen.append(text) or [1, 2, 3]))
    assert tokens == [1, 2, 3] and "class PromptMemory" in seen[0]
    assert probe_tokens(object()) == []


def test_sizing_releases_what_the_probes_left_in_reference_cycles():
    import gc
    import weakref

    live = weakref.WeakSet()

    class Buffer:                              # an array only a reference cycle keeps alive
        def __init__(self):
            live.add(self)
            self.me = self

    runtime = Runtime(resident=1000)
    runtime.get_active_memory = lambda: 1000 + 100_000 * len(live) + sum(cache_nbytes(c) for c in runtime.caches)
    memory = controller(budget=200_000, runtime=runtime)

    def probes():
        Buffer()                               # garbage the concurrency probes leave behind
        return "admission"

    gc.disable()
    try:
        assert memory.sized(ProbeEngine(runtime), probes) == "admission"
        assert len(live) == 0
    finally:
        gc.enable()


def test_sizing_releases_the_probes_last_round_before_measuring_the_model():
    runtime = Runtime(resident=1000)
    memory = controller(budget=200_000, runtime=runtime)
    engine = ProbeEngine(runtime)
    order = []

    def release_rounds():                     # the family drops its last shared round's per-row states
        order.append("release")
        runtime.resident -= 150_000

    engine.release_rounds = release_rounds

    def probes():
        order.append("concurrency")
        runtime.resident += 150_000
        return "admission"

    original = engine.prefill_prefix
    engine.prefill_prefix = lambda *a, **k: order.append("prompt") or original(*a, **k)
    assert memory.sized(engine, probes) == "admission"
    assert order == ["concurrency", "release", "prompt"]
    assert memory.largest_window(8192) > 0


def test_nemotron_releases_its_last_rounds_row_states():
    from tensorfold.families.nemotron_h.model import NemotronH

    family = NemotronH.__new__(NemotronH)
    family.fused = SimpleNamespace(row_states={0: ("conv", "ssm")})
    family._last_hidden = "hidden"
    family.release_rounds()
    assert family.fused.row_states == {} and family._last_hidden is None


@pytest.mark.parametrize("fused", [False, True])
def test_flash_next_sizing_releases_probe_rows_and_draft_references(fused):
    import weakref

    from tensorfold.families.qwen4_exp.runtime import FlashNext

    class Buffer:
        pass

    live = weakref.WeakSet()
    runtime = Runtime(resident=1000)
    active = runtime.get_active_memory
    runtime.get_active_memory = lambda: active() + 150_000 * len(live)
    memory = controller(budget=200_000, runtime=runtime)
    engine = ProbeEngine(runtime)
    family = FlashNext.__new__(FlashNext)
    family.model = SimpleNamespace()
    family.fused = SimpleNamespace(row_states={}, _last_heads=[]) if fused else None
    family.mtp_fused = SimpleNamespace(row_states={}, _last_heads=[]) if fused else None
    family._specs = {}
    engine.release_rounds = family.release_rounds

    def probes():
        buffer = Buffer()
        live.add(buffer)
        family._streams = family.model.last_streams = buffer
        family._specs[0] = (buffer, 4)
        for decode in (family.fused, family.mtp_fused):
            if decode is not None:
                decode.row_states[0] = [(buffer, buffer, 0)]
                decode._last_heads.append(buffer)
                decode.last_streams = buffer
        return "admission"

    assert memory.sized(engine, probes) == "admission"
    assert not live and memory.largest_window(8192) > 0


def test_the_engine_releases_rounds_only_with_no_stream_live():
    engine = FakeEngine()
    calls = []
    engine.model.release_rounds = lambda: calls.append(1)
    engine.release_rounds()
    assert calls == [1]
    engine._live = [(SimpleNamespace(finished=False), [])]
    engine.release_rounds()
    assert calls == [1]


def test_the_window_counts_a_live_streams_growth_beyond_its_cache():
    memory = controller(budget=40_000_000, store=CheckpointStore(4, copier=lambda c: c, sizer=cache_nbytes))
    memory.begin(256, 64, admit=False)
    memory.observe_cache(populated(), workspace=False)
    per = memory.profile.bytes_per_token
    kept, alone = memory.largest_window(1 << 20, resumable=True), memory.largest_window(1 << 20)
    memory.stream_per_token = per                                   # a stream that is only its cache: no change
    assert memory.largest_window(1 << 20, resumable=True) == kept and memory.largest_window(1 << 20) == alone
    memory.stream_per_token = 3 * per                               # twice as much again outside it, as measured
    assert 0 < memory.largest_window(1 << 20, resumable=True) < kept
    assert 0 < memory.largest_window(1 << 20) < alone


def kv(tokens):
    keys, values = Array((1, 1, tokens, 256)), Array((1, 1, tokens, 256))    # 1 KiB of keys and values a token
    return [SimpleNamespace(keys=keys, values=values, state=(keys, values), offset=tokens)]


@pytest.mark.parametrize("counted", [True, False])
def test_a_growing_conversation_keeps_each_turns_prompt_up_to_the_fitted_window(counted):
    """Bounded memory: a turn's cache, its stream's growth outside it and a kept copy fit, turn after turn."""

    extra = 512                                                     # a stream's own growth a token (draft state)
    store = CheckpointStore(4, copier=lambda c: c, budget_bytes=1 << 20, sizer=cache_nbytes)
    store.admit_oversize = True
    runtime, live = Runtime(resident=64 << 20), []
    runtime.get_active_memory = lambda: (runtime.resident + store.nbytes
                                         + sum(cache_nbytes(c) + extra * c[0].offset for c in live))
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    memory = PromptMemory(96 << 20, model, runtime=runtime, store=store, overhead_bytes=0, bootstrap_bytes=0,
                          chunk_rows=256)
    memory.observe_cache(kv(256), workspace=False)
    memory.stream_per_token = memory.profile.bytes_per_token + (extra if counted else 0)   # the probe's reading
    window, fitted = memory.fit_window(1 << 20, True)
    assert fitted and 0 < window < 16_000
    kept = []
    for prompt in [*range(1_000, window - 64, 1_000), window - 64]:  # the last turn fills the window
        memory.begin(prompt, 64, admit=False)
        store.match(list(range(prompt)), take=True)                 # the last turn's prefix becomes this turn's cache
        boundary = prompt // 256 * 256                              # the last chunk start: the copy happens here
        live[:] = [kv(boundary)]
        kept.append(memory.allow_checkpoint(live[0]))
        assert memory.projected(prompt, current_cache=live[0], extra_bytes=cache_nbytes(live[0])) <= memory.budget \
            or not kept[-1]
        if kept[-1]:
            store.insert(list(range(boundary)), kv(boundary), last_prompt=list(range(prompt)))
        live[0] = kv(prompt + 64)                                   # the prompt and reply, then the turn ends
        assert runtime.get_active_memory() <= memory.budget or not counted
        live.clear()
        memory.end()
    assert len(kept) > 8
    assert all(kept) if counted else not kept[-1]                   # uncounted, the window's top can't be kept
