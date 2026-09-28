"""Evicted conversations go to disk and come back instead of a full prefill; older checkpoints of a live one do not."""

from __future__ import annotations

import pytest

from tensorfold.server.checkpoints import CheckpointStore


def _store(slots=2, budget=None, spilled=None):
    return CheckpointStore(slots, copier=lambda c: c, budget_bytes=budget, sizer=lambda c: c[0],
                           on_evict=(lambda entry: spilled.append(list(entry.tokens))) if spilled is not None else None)


def test_an_evicted_conversation_is_handed_to_the_hook():
    spilled = []
    store = _store(slots=2, spilled=spilled)
    store.insert([1, 2], [10], last_prompt=[1, 2])
    store.insert([5, 6], [10], last_prompt=[5, 6])
    store.insert([8, 9], [10], last_prompt=[8, 9])
    assert spilled == [[1, 2]] and store.spilled == 1


def test_an_older_checkpoint_of_a_conversation_that_moved_on_is_not_spilled():
    spilled = []
    store = _store(slots=2, spilled=spilled)
    store.insert([1, 2], [10], last_prompt=[1, 2])
    store.insert([1, 2, 3, 4], [10], last_prompt=[1, 2, 3, 4])   # the same conversation, one turn on
    store.insert([7, 8], [10], last_prompt=[7, 8])               # evicts [1, 2], which [1, 2, 3, 4] extends
    assert spilled == []


def test_budget_evictions_are_spilled_too():
    spilled = []
    store = _store(slots=8, budget=25, spilled=spilled)
    store.insert([1], [10], last_prompt=[1])
    store.insert([2], [10], last_prompt=[2])
    store.insert([3], [10], last_prompt=[3])
    assert spilled == [[1]]


def test_memory_pressure_evictions_are_spilled():
    spilled = []
    store = _store(slots=4, spilled=spilled)
    store.insert([1], [10], last_prompt=[1])
    store.insert([2], [10], last_prompt=[2])
    assert store.evict_one() and spilled == [[1]]


def test_system_blocks_are_never_spilled():
    spilled = []
    store = _store(slots=4, spilled=spilled)
    store.insert([1, 2, 3], [10], last_prompt=[1, 2, 3], pinned=True)
    assert store.evict_one() and spilled == []


def test_a_failing_hook_never_fails_the_insert():
    def boom(entry):
        raise OSError("disk full")

    store = CheckpointStore(1, copier=lambda c: c, sizer=lambda c: c[0], on_evict=boom)
    store.insert([1], [10], last_prompt=[1])
    store.insert([2], [10], last_prompt=[2])
    assert len(store) == 1 and store.spilled == 0


class Layer:
    def __init__(self, n: int) -> None:
        import mlx.core as mx

        self.keys = mx.ones((1, 2, n, 4))
        self.offset = n


def test_a_spilled_conversation_is_found_and_read_back(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine.prefix_snapshots import DiskBlocks, load_snapshot
    from tensorfold.server.checkpoints import spill_conversation

    model = "/models/qwen|mlx=1"
    store = CheckpointStore(1, copier=lambda c: c, sizer=lambda c: sum(int(x.keys.nbytes) for x in c),
                            on_evict=lambda e: spill_conversation(e, tmp_path, model, limit_bytes=1 << 30))
    store.insert([1, 2, 3], [Layer(3)], last_prompt=[1, 2, 3])
    store.insert([9, 9], [Layer(2)], last_prompt=[9, 9])            # evicts and writes [1, 2, 3]
    hit = DiskBlocks(tmp_path, model).best([1, 2, 3, 4], 0)
    assert hit is not None and hit[1] == [1, 2, 3]
    tokens, cache = load_snapshot(hit[0], model)
    assert tokens == [1, 2, 3] and cache[0].offset == 3 and bool(mx.array_equal(cache[0].keys, Layer(3).keys).item())


def test_the_spill_directory_keeps_the_newest_within_its_byte_limit(tmp_path):
    import os

    pytest.importorskip("mlx.core")

    from tensorfold.server.checkpoints import CheckpointEntry, spill_conversation

    model = "/models/qwen|mlx=1"
    for i in range(3):
        spill_conversation(CheckpointEntry([i, 1], [Layer(64)], [i, 1], nbytes=10), tmp_path, model, limit_bytes=1 << 30)
    files = sorted(tmp_path.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    for n, path in enumerate(files):                                # distinct mtimes, oldest first
        os.utime(path, (1_000_000 + n, 1_000_000 + n))
    one = files[0].stat().st_size
    spill_conversation(CheckpointEntry([7, 1], [Layer(64)], [7, 1], nbytes=10), tmp_path, model,
                       limit_bytes=int(2.5 * one))
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    assert not files[0].exists() and not files[1].exists()
