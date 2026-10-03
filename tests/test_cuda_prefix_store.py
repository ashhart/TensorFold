"""Disk generations and lazy prefix selection without a device runtime."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tensorfold.cuda.prefix_store import PrefixStore
from tensorfold.cuda.streams import PrefixCache


class Codec:
    def __init__(self):
        self.loads = []
        self.fail_save = False

    def save_prefix(self, path, ids, state, snap, *, identity):
        if self.fail_save:
            path.write_bytes(b"partial")
            raise OSError("disk full")
        content = json.dumps([identity, ids, state.value, snap]).encode()
        path.write_bytes(content)
        return {"tokens": list(ids), "sha256": hashlib.sha256(content).hexdigest(),
                    "file_bytes": len(content), "tensor_bytes": 64}

    def verify_prefix(self, path, receipt, *, identity):
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt["sha256"] or json.loads(content)[0] != identity:
            raise ValueError("invalid snapshot")

    def load_prefix(self, path, receipt, *, identity, device, room=None):
        self.verify_prefix(path, receipt, identity=identity)
        _, ids, value, snap = json.loads(path.read_bytes())
        self.loads.append(ids)
        return ids, SimpleNamespace(value=value, room=room), snap


@pytest.fixture
def store(tmp_path):
    codec = Codec()
    with PrefixStore(tmp_path, "identity", codec) as store:
        yield store, codec


def cache(*prefixes, keep=4):
    result = PrefixCache(keep)
    for ids in prefixes:
        result.add(ids, SimpleNamespace(value=sum(ids)), ["draft", len(ids)])
    return result


def test_wake_indexes_without_loading_then_uses_longest_strict_prefix(store):
    store, codec = store
    store.save(cache([1], [1, 2], [9]))
    restored = store.attach(PrefixCache(4), device="cpu")
    assert not restored.entries and not codec.loads
    assert restored.longest([8, 7]) is None and not codec.loads
    hit = restored.longest([1, 2, 3])
    assert hit[0] == [1, 2] and hit[1].value == 3 and hit[2] == ["draft", 2]
    assert codec.loads == [[1, 2]]
    assert restored.longest([1, 2, 4]) is hit and len(codec.loads) == 1
    assert restored.longest([9]) is None


def test_failed_write_leaves_previous_generation_and_live_cache_usable(store):
    store, codec = store
    store.save(cache([1]))
    previous = store.directory
    live = cache([2], [3])
    codec.fail_save = True
    with pytest.raises(OSError, match="disk full"):
        store.save(live)
    assert store.directory == previous
    store.verify()
    assert live.longest([3, 4])[1].value == 3
    assert len(list(previous.parent.iterdir())) == 1
    assert store.attach(PrefixCache(), device="cpu").longest([1, 2])[0] == [1]


def test_resleep_keeps_untouched_disk_prefixes_and_bounds_generations(store):
    store, codec = store
    store.save(cache([1], [2], keep=3))
    previous = store.directory
    restored = store.attach(PrefixCache(3), device="cpu")
    restored.add([3], SimpleNamespace(value=3), None)
    store.save(restored)
    assert not previous.exists() and not codec.loads
    assert store.snapshot()["saved_prefixes"] == 3
    restored = store.attach(PrefixCache(3), device="cpu")
    restored.add([4], SimpleNamespace(value=4), None)
    store.save(restored)
    assert store.snapshot()["saved_prefixes"] == 3
    assert store.snapshot()["omitted_prefixes"] == 1
    assert len(list(store.directory.parent.iterdir())) == 1


def test_corruption_is_rejected_on_wake_and_becomes_a_miss_if_changed_later(store):
    store, _codec = store
    store.save(cache([1, 2]))
    restored = store.attach(PrefixCache(), device="cpu")
    next(store.directory.glob("*.safetensors")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="invalid snapshot"):
        store.verify()
    assert restored.longest([1, 2, 3]) is None
    assert store.snapshot()["load_failures"] == 1


def test_memory_admission_can_choose_resident_prefix_without_loading(store):
    store, codec = store
    store.save(cache([1, 2]))
    calls = []
    restored = store.attach(cache([1]), device="cpu",
                            admit=lambda size, prompt: calls.append((size, prompt)) or False)
    assert restored.longest([1, 2, 3])[0] == [1]
    assert calls == [(64, [1, 2, 3])] and not codec.loads


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_unrequested_disk_prefix_does_not_prevent_preserving_live_state(store, damage):
    store, codec = store
    store.save(cache([1], [2]))
    restored = store.attach(PrefixCache(4), device="cpu")
    damaged = store.directory / store.records[0]["file"]
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"corrupt")
    restored.add([3], SimpleNamespace(value=3), None)
    store.save(restored)
    store.verify()
    assert store.snapshot()["saved_prefixes"] == 2
    assert store.snapshot()["unavailable_prefixes"] == 1
    assert not codec.loads
    restored = store.attach(PrefixCache(4), device="cpu")
    assert restored.longest([1, 4]) is None
    assert restored.longest([2, 4])[0] == [2]
    assert restored.longest([3, 4])[0] == [3]


def test_copy_write_failure_preserves_original_generation(store, monkeypatch):
    import shutil

    store, _codec = store
    store.save(cache([1]))
    previous = store.directory
    restored = store.attach(PrefixCache(4), device="cpu")

    def disk_full(source, destination):
        destination.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", disk_full)
    with pytest.raises(OSError, match="disk full"):
        store.save(restored)
    assert store.directory == previous and len(list(previous.parent.iterdir())) == 1
    store.verify()
    assert restored.longest([1, 2])[0] == [1]


def test_close_removes_only_its_private_directory(tmp_path):
    unrelated = tmp_path / "other"
    unrelated.write_text("keep")
    store = PrefixStore(tmp_path, "identity", Codec())
    store.save(cache([1]))
    assert store.directory.stat().st_mode & 0o077 == 0
    store.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other"]
