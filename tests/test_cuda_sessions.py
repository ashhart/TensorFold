"""Prompt states below the device: PrefixCache's contract, spills to host memory and disk, checked compatible files."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tensorfold.cuda.session_disk import DiskTier
from tensorfold.cuda.sessions import HostTier, TieredCache, compat_hash, entry_key, strict_prefix

from cuda_lane_fakes import FailingTier


class Codec:
    def to_host(self, state):
        return {"x": np.array(state, dtype=np.int64)}

    def from_host(self, arrays):
        return [int(v) for v in arrays["x"]]


def test_strict_prefix_leaves_a_token():
    assert strict_prefix([1, 2], [1, 2, 3]) and not strict_prefix([1, 2, 3], [1, 2, 3])
    assert not strict_prefix([], [1]) and not strict_prefix([1, 3], [1, 2, 3])


def test_keys_change_with_the_compat_settings():
    a, b = compat_hash({"engine": "x", "kv": "bf16"}), compat_hash({"engine": "x", "kv": "fp8"})
    assert a != b and compat_hash({"kv": "bf16", "engine": "x"}) == a
    assert entry_key(a, [1, 2]) != entry_key(b, [1, 2]) and entry_key(a, [1, 2]) != entry_key(a, [1, 2, 3])


def test_the_device_tier_keeps_prefix_caches_contract_and_releases():
    gone = []
    c = TieredCache(2, release=gone.append)
    c.add([1, 2], "a", None)
    c.add([1, 2, 3], "b", None)
    assert c.longest([1, 2, 3, 4])[1] == "b" and c.named([1, 2, 3], 2)[1] == "a"
    c.add([1, 2], "dup", None)  # kept already: the new state is released
    assert gone == ["dup"] and len(c.entries) == 2
    c.add([5, 6], "c", None)
    assert gone == ["dup", "b"]  # both were resumed: the least recently used goes
    assert c.find([5, 6, 7]) == (2, -1) and c.find([9]) is None


def test_evicted_entries_spill_down_and_load_back(tmp_path):
    tiers = [HostTier(10_000, min_tokens=3), DiskTier(tmp_path, {"engine": "x"}, limit=1 << 20)]
    c = TieredCache(1, codec=Codec(), tiers=tiers)
    c.add([1, 2], [10], None)
    c.add([1, 2, 3, 4], [20], None)  # [1, 2] is too short for the host tier: the disk takes it
    c.add([7, 7, 7], [30], None)
    assert c.spilled == 2 and tiers[0].keys() == [c.key([1, 2, 3, 4])] and tiers[1].keys() == [c.key([1, 2])]
    assert c.find([1, 2, 3, 4, 5]) == (4, 0) and c.find([1, 2, 9]) == (2, 1)
    assert c.load([1, 2, 3, 4, 5], 4, 0) == [20] and c.load([1, 2, 9], 2, 1) == [10]
    assert c.victim() == 0
    c.drop_at(0)
    assert c.entries == [] and c.find([7, 7, 7, 1]) == (3, 0)


def test_victim_spares_the_entry_a_request_resumes():
    c = TieredCache(4)
    for ids in ([1], [1, 2], [3]):
        c.add(ids, None, None)
    assert c.victim() == 0 and c.victim(keep=([1],)) == 1 and c.victim(keep=([1], [1, 2])) == 2
    assert c.victim(keep=([1], [1, 2], [3])) is None


def test_victim_pins_the_resumed_entry_not_its_length():
    c = TieredCache(4)
    c.add([1, 2, 3], "a", None)
    c.add([4, 5, 6], "b", None)  # unrelated, the same length
    assert c.victim(keep=([1, 2, 3],)) == 1


def test_an_entry_the_host_tier_pushes_out_moves_to_disk(tmp_path):
    entry = 8 * 4  # one int64 array of four values
    tiers = [HostTier(2 * entry), DiskTier(tmp_path, {"engine": "x"}, limit=1 << 20)]
    c = TieredCache(0, codec=Codec(), tiers=tiers)
    for i, ids in enumerate(([1, 1], [2, 2], [3, 3])):
        c.add(ids, [i, i, i, i], None)
    assert tiers[0].keys() == [c.key([2, 2]), c.key([3, 3])] and tiers[1].keys() == [c.key([1, 1])]
    assert c.find([1, 1, 9]) == (2, 1) and c.load([1, 1, 9], 2, 1) == [0, 0, 0, 0]
    assert c.spilled == 3 and c.dropped == 0


def test_keep_zero_sends_every_entry_down():
    tier = HostTier(1 << 20)
    c = TieredCache(0, codec=Codec(), tiers=[tier])
    c.add([1, 2, 3], [5], None)
    assert c.entries == [] and c.find([1, 2, 3, 4]) == (3, 0)


def test_the_host_tier_drops_the_least_recently_used_past_its_limit():
    t = HostTier(3 * 8)
    for i in range(4):
        t.put(f"k{i}", [i, i], {"x": np.zeros(1, dtype=np.int64)})
    assert t.keys() == ["k1", "k2", "k3"] and t.used == 24
    t.get("k1")
    t.put("k4", [9, 9], {"x": np.zeros(1, dtype=np.int64)})
    assert t.keys() == ["k3", "k1", "k4"]
    assert not t.put("big", [1], {"x": np.zeros(4, dtype=np.int64)})


def arrays(n=3):
    return {"rows": np.arange(n * 5, dtype=np.uint8).reshape(n, 5), "f": np.array([1.5, -2.0], dtype=np.float32)}


def test_a_disk_entry_reads_back_bit_for_bit_after_a_restart(tmp_path):
    t = DiskTier(tmp_path, {"engine": "x"}, limit=1 << 20)
    assert t.put("abc", [4, 5, 6], arrays(5000))
    again = DiskTier(tmp_path, {"engine": "x"}, limit=1 << 20)
    ids, got = again.get("abc")
    assert ids == [4, 5, 6] and again.find([4, 5, 6, 7]) == ("abc", 3)
    for k, v in arrays(5000).items():
        assert got[k].dtype == v.dtype and np.array_equal(got[k], v)


def test_another_builds_entries_are_never_found(tmp_path):
    DiskTier(tmp_path, {"engine": "x", "kv": "bf16"}, limit=1 << 20).put("abc", [1, 2], arrays())
    other = DiskTier(tmp_path, {"engine": "x", "kv": "fp8"}, limit=1 << 20)
    assert other.find([1, 2, 3]) is None and other.keys() == []


def test_a_damaged_entry_is_refused_and_deleted(tmp_path):
    t = DiskTier(tmp_path, {}, limit=1 << 20)
    t.put("abc", [1, 2], arrays())
    path = t.path("abc")
    raw = bytearray(path.read_bytes())
    raw[-ALIGNED_TAIL] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        t.get("abc")
    assert not path.exists() and t.keys() == []


ALIGNED_TAIL = 4096 - 7  # inside the last segment's bytes, not its zero padding


def test_a_crash_mid_write_leaves_no_entry(tmp_path):
    t = DiskTier(tmp_path, {}, limit=1 << 20)
    t.put("good", [1, 2], arrays())
    (t.dir / "half.tmp").write_bytes(b"TFSTATE1")
    (t.dir / "cut.tfs").write_bytes(t.path("good").read_bytes()[:5000])
    again = DiskTier(tmp_path, {}, limit=1 << 20)
    assert again.keys() == ["good"] and sorted(p.name for p in again.dir.iterdir()) == ["compat.json", "good.tfs"]


def test_the_disk_budget_drops_the_least_recently_used(tmp_path):
    t = DiskTier(tmp_path, {}, limit=3 * 4 * 4096)  # a header and three segments of 4 KiB an entry
    for i in range(4):
        t.put(f"k{i}", [i, i], arrays())
        os.utime(t.path(f"k{i}"), ns=(i * 10**9, i * 10**9))
    assert t.keys() == ["k1", "k2", "k3"]
    t.get("k1")
    t.put("k4", [9, 9], arrays())
    assert t.keys() == ["k1", "k3", "k4"]
    assert not DiskTier(tmp_path / "small", {}, limit=4096).put("big", [1], arrays())


def test_ranks_retain_the_entries_they_all_hold(tmp_path):
    a = DiskTier(tmp_path, {}, limit=1 << 20, rank=0)
    b = DiskTier(tmp_path, {}, limit=1 << 20, rank=1)
    for k in ("x", "y"):
        a.put(k, [1, len(k)], arrays())
    b.put("y", [1, 1], arrays())
    both = sorted(set(a.keys()) & set(b.keys()))
    a.retain(both)
    assert a.keys() == ["y"] and not a.path("x").exists()


def test_a_host_tier_used_alone_keeps_nothing_past_its_limit():
    import gc
    import weakref

    import numpy as np

    tier = HostTier(3 * 8 * 4)  # three entries of one int64 array of four values
    refs = []
    for i in range(10):
        a = np.full(4, i, dtype=np.int64)
        refs.append(weakref.ref(a))
        tier.put(f"k{i}", [i, i], {"x": a}, owned=True)
        del a
    gc.collect()
    assert tier.used <= tier.limit and tier.keys() == ["k7", "k8", "k9"]
    assert sum(r() is not None for r in refs) == 3  # the seven pushed out are gone, not queued


def test_a_failing_tier_drops_the_entry_and_still_frees_its_state(tmp_path):
    freed = []
    c = TieredCache(0, codec=Codec(), tiers=[FailingTier()], release=freed.append)
    c.add([1, 2], [5], None)  # keep 0: straight to the tier, which fails
    assert freed == [[5]] and c.entries == [] and c.dropped == 1 and isinstance(c.error, OSError)


def test_a_state_that_cannot_reach_the_host_is_still_freed():
    class Broken(Codec):
        def to_host(self, state):
            raise RuntimeError("device copy failed")

    freed = []
    c = TieredCache(0, codec=Broken(), tiers=[HostTier(1 << 20)], release=freed.append)
    c.add([1, 2], [5], None)
    assert freed == [[5]] and c.dropped == 1 and isinstance(c.error, RuntimeError)


def test_one_hashing_pass_gives_each_prefix_its_entry_key():
    c = TieredCache(1, codec=Codec(), compat={"engine": "x"})
    prompt = list(range(3, 300))
    keys = c._prefix_keys(prompt, {1, 7, 64, 200, 296})
    assert keys == {n: c.key(prompt[:n]) for n in (1, 7, 64, 200, 296)}
