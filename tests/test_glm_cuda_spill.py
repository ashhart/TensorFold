"""GLM's spill decisions on the CPU: what is written, what fits a request, failed reads, copies of the live caches."""

from __future__ import annotations

import dataclasses
import os
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import spill  # noqa: E402
from tensorfold.families.glm5_next.cuda.engine import GlmEngine  # noqa: E402

pytestmark = pytest.mark.torch

MTP, DFLASH, AUTO = [0], [10], [4]                # policy codes: MTP drafts, DFlash2 drafts, either


@dataclasses.dataclass
class Snap:
    ids: list
    rec: object
    mtp_len: int
    drafter_end: int
    rows: list | None = None
    nbytes: int = 0
    drafter_rows: list | None = None


def _snap(n, *, mtp_len=3, drafter_end=None, rows=True, drafter_rows=None):
    return Snap(list(range(n)), torch.arange(4.0), mtp_len, n if drafter_end is None else drafter_end,
                rows=[torch.ones(n, 2)] if rows else None, drafter_rows=drafter_rows)


def _engine(drafter=None, store=None):
    e = SimpleNamespace(rank=0, drafter=drafter, spill=store, cache=[], live=[], cache_bytes=1 << 40)
    e._drafters = lambda code: GlmEngine._drafters(e, code)
    e._fit = lambda n, end, mtp_len, code: GlmEngine._fit(e, n, end, mtp_len, code)
    e._fits = lambda snap, code: GlmEngine._fits(e, snap, code)
    e._held_bytes = lambda: 0
    e._gather_ints = lambda v: [list(v), list(v)]                     # both ranks alike
    e._remember = e.cache.append
    return e


def test_draft_caches_fit_the_requests_drafters():
    with_mtp, with_window = _snap(16, drafter_end=-1), _snap(16, mtp_len=-1)
    plain, dflash = _engine(), _engine(drafter=SimpleNamespace())
    assert GlmEngine._fits(plain, with_mtp, MTP) and not GlmEngine._fits(plain, with_window, MTP)
    assert GlmEngine._fits(plain, with_mtp, AUTO)                    # no drafter: auto is MTP alone
    assert GlmEngine._fits(dflash, with_window, DFLASH) and not GlmEngine._fits(dflash, with_mtp, DFLASH)
    assert not GlmEngine._fits(dflash, with_mtp, AUTO) and not GlmEngine._fits(dflash, with_window, AUTO)
    assert GlmEngine._fits(dflash, _snap(16), AUTO)


def test_a_copy_is_written_only_with_the_rings_window():
    none, plain, ring = _engine(), _engine(drafter=SimpleNamespace()), _engine(drafter=SimpleNamespace(ring=64))
    window = [torch.ones(2)]
    assert GlmEngine._writable(none, _snap(16))                                  # no drafter: the rows suffice
    assert not GlmEngine._writable(plain, _snap(16, drafter_rows=window))        # no ring: save_rows drops it
    assert not GlmEngine._writable(ring, _snap(16))                              # a ring's window rows not taken
    assert not GlmEngine._writable(ring, _snap(16, drafter_end=8, drafter_rows=window))     # the drafter lagged
    assert GlmEngine._writable(ring, _snap(16, drafter_rows=window))
    assert not GlmEngine._writable(ring, _snap(16, rows=False, drafter_rows=window))        # its rows are gone
    ring.live = list(range(16))
    assert GlmEngine._writable(ring, _snap(16, rows=False, drafter_rows=window))            # still the live caches'


def _store(tmp_path, classes=(Snap,)):
    cfg = spill.SpillConfig(root=str(tmp_path), gib=1.0, min_tokens=8, min_free_gib=0.0, direct=False)
    return spill.SpillStore(cfg, device="cpu", signature="test", model_id="m", quiet=True, classes=classes)


def _stored(tmp_path, snap):
    store = _store(tmp_path)
    store.save(snap.ids, snap, skip=("ids", "nbytes")).done.wait(10)
    return store


def test_a_stored_prompt_comes_back_as_a_kept_snapshot(tmp_path):
    snap = _snap(32)
    e = _engine(store=_stored(tmp_path, snap))
    back = GlmEngine._spill_load(e, snap.ids, MTP)
    assert back is not None and e.cache == [back] and back.ids == snap.ids
    assert back.nbytes == snap.rows[0].numel() * 4 and torch.equal(back.rows[0], snap.rows[0])


def test_a_stored_prompt_without_the_requests_draft_caches_prefills(tmp_path):
    snap = _snap(32, mtp_len=-1)                                     # kept without pending MTP rows
    e = _engine(store=_stored(tmp_path, snap))
    assert GlmEngine._spill_load(e, snap.ids, MTP) is None and e.cache == []
    assert e.spill.stats.load_unfit == 1 and not e.spill.has(snap.ids)


def test_a_cut_file_prefills_and_is_forgotten(tmp_path):
    snap = _snap(32)
    store = _stored(tmp_path, snap)
    path = os.path.join(store.dir, store.index[spill.ids_key(snap.ids)].name + ".safetensors")
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 8)
    e = _engine(store=store)
    assert GlmEngine._spill_load(e, snap.ids, MTP) is None and e.cache == []
    assert store.stats.load_failed == 1 and not store.has(snap.ids) and not os.path.exists(path)


def test_a_file_of_a_class_the_engine_does_not_read_prefills_and_is_forgotten(tmp_path):
    snap = _snap(32)
    _stored(tmp_path, snap)
    e = _engine(store=_store(tmp_path, classes=()))                  # the file names Snap: never decoded here
    assert GlmEngine._spill_load(e, snap.ids, MTP) is None and e.cache == []
    assert e.spill.stats.load_failed == 1 and not e.spill.has(snap.ids)


def test_one_ranks_failed_read_makes_both_prefill(tmp_path):
    snap = _snap(32)
    e = _engine(store=_stored(tmp_path, snap))
    e._gather_ints = lambda v: [list(v), [0, 0]]                     # the other rank could not read it
    assert GlmEngine._spill_load(e, snap.ids, MTP) is None and e.cache == []
    assert e.spill.stats.load_failed == 1 and not e.spill.has(snap.ids)


def test_a_copy_of_the_live_caches_is_off_them_before_another_prompt_writes_them(tmp_path):
    from tensorfold.families.glm5_next.cuda.decode import Snapshot

    n = 32
    store = _store(tmp_path, classes=(Snapshot,))
    write = store._write
    store._write = lambda *a: (time.sleep(0.3), write(*a))[1]   # a slow disk
    live = torch.arange(64 * 4, dtype=torch.float32).view(64, 4)                   # one layer's attention rows
    want = live[:n].clone()
    e = GlmEngine.__new__(GlmEngine)
    e.rank, e.drafter = 0, SimpleNamespace(ring=64)
    e.e = SimpleNamespace(st=SimpleNamespace(kc=[live], vc=[None], index=None))
    e.spill, e.spill_jobs, e.cache_bytes, e.cache_entries = store, [], 0, 8         # no room to save its rows
    snap = Snapshot(list(range(n)), torch.zeros(2), torch.zeros(2), None, -1, n, drafter_rows=[torch.ones(3)])
    e.cache, e.live = [snap], list(range(n)) + [7]
    GlmEngine._take_over(e, [5, 6])                  # another conversation: the kept prompt is written as it leaves
    live.fill_(-1.0)                                 # that conversation's prefill writes the caches from row 0
    assert e.cache == [] and store.drain(10)
    back, _ = store.load(spill.ids_key(snap.ids))
    assert back.drafter_end == n and torch.equal(back.rows[0], want)


def test_an_older_turn_a_kept_one_extends_is_not_written(tmp_path):
    store = _store(tmp_path)
    e = GlmEngine.__new__(GlmEngine)
    e.rank, e.drafter, e.spill, e.spill_jobs, e.live = 0, None, store, [], []
    e._share, e._held_bytes, e.cache_bytes, e.cache_entries = (lambda v: v), (lambda: 1 << 30), 1, 8
    first, second, other = _snap(16), _snap(24), _snap(16)          # second goes on from first's conversation
    other.ids = [100 + t for t in other.ids]
    e.cache = [first, second, other]
    GlmEngine._spill_behind(e)                                        # early writes pass over the older turn
    assert store.drain(10) and store.has(second.ids) and store.has(other.ids) and not store.has(first.ids)
    GlmEngine._drop(e, first)                                         # and so does its eviction
    assert store.drain(10) and not store.has(first.ids) and store.stats.unwritten_evicted == 1
    GlmEngine._drop(e, second)
    GlmEngine._drop(e, other)
    assert store.stats.unwritten_evicted == 1 and len(store.index) == 2


def test_a_stored_prefix_is_looked_up_under_the_same_condition_as_a_kept_one(tmp_path):
    snap = _snap(32)
    e = _engine(store=_stored(tmp_path, snap))
    prompt = snap.ids + [7, 8]
    assert GlmEngine._spill_cached(e, prompt, None, True, MTP) == -32             # read the 32 stored tokens
    assert GlmEngine._spill_cached(e, prompt, _snap(16), True, MTP) == -32
    assert GlmEngine._spill_cached(e, prompt, None, False, MTP) == 0              # no resume: neither is looked up


def test_ranks_started_with_different_spill_settings_are_refused(tmp_path):
    from tensorfold.families.glm5_next.cuda.spill_hooks import GlmSpill
    eng = SimpleNamespace(_gather_ints=lambda v: [list(v), [0] * len(v)])     # rank 1 without --spill-gib
    with pytest.raises(RuntimeError, match="different --spill-gib settings"):
        GlmSpill._spill_setup(eng, spill.SpillConfig(root=str(tmp_path), gib=64.0))
    off = SimpleNamespace(_gather_ints=lambda v: [list(v), list(v)])
    GlmSpill._spill_setup(off, None)                                                # both off: no store, no error
    assert off.spill is None and off.spill_cfg is None


def test_a_stored_prefix_whose_draft_caches_do_not_fit_is_not_read(tmp_path):
    snap = _snap(32, mtp_len=-1)                                     # stored without pending MTP rows
    e = _engine(store=_stored(tmp_path, snap))
    prompt = snap.ids + [7, 8]
    assert GlmEngine._spill_cached(e, prompt, None, True, MTP) == 0              # its fields say so: not read
    e.drafter = SimpleNamespace(ring=64)
    assert GlmEngine._spill_cached(e, prompt, None, True, DFLASH) == -32         # its DFlash2 window fits
    assert e.spill.stats.loaded == 0 and e.spill.stats.load_unfit == 0


def _bare(store, drafter):
    e = GlmEngine.__new__(GlmEngine)
    e.rank, e.drafter, e.spill, e.spill_jobs, e.live, e.cache_entries = 0, drafter, store, [], [], 8
    return e


def test_take_over_writes_a_kept_state_with_the_window_its_saved_copy_drops(tmp_path):
    from tensorfold.families.glm5_next.cuda.decode import Snapshot

    n = 32
    store = _store(tmp_path, classes=(Snapshot,))
    live = torch.arange(64 * 4, dtype=torch.float32).view(64, 4)
    want = live[:n].clone()
    e = _bare(store, SimpleNamespace(ring=64))
    e.e = SimpleNamespace(st=SimpleNamespace(kc=[live], vc=[None], index=None))
    e.cache_bytes = 1 << 30                                           # room: its rows are saved in memory
    snap = Snapshot(list(range(n)), torch.zeros(2), torch.zeros(2), None, -1, n, drafter_rows=[torch.ones(3)])
    e.cache, e.live = [snap], list(range(n)) + [7]
    GlmEngine._take_over(e, [5, 6])                  # another conversation: the copy kept in memory has no window
    live.fill_(-1.0)
    assert e.cache == [snap] and (snap.drafter_end, snap.drafter_rows) == (-1, None) and store.drain(10)
    back, _ = store.load(spill.ids_key(snap.ids))
    assert back.drafter_end == n and back.drafter_rows is not None and torch.equal(back.rows[0], want)


def test_an_older_turn_is_written_when_the_longer_one_cannot_be(tmp_path):
    store = _store(tmp_path)
    e = _bare(store, SimpleNamespace(ring=64))
    first, second = _snap(16, drafter_rows=[torch.ones(2)]), _snap(24)     # the longer turn kept no window
    e.cache = [first, second]
    GlmEngine._drop(e, first)
    assert store.drain(10) and store.has(first.ids) and store.stats.unwritten_evicted == 0


def test_reading_the_oldest_stored_prompt_does_not_evict_it_first(tmp_path):
    from tensorfold.families.glm5_next.cuda.decode import Snapshot

    n = 64
    store = _store(tmp_path, classes=(Snapshot,))
    target, other = (Snapshot(list(range(b, b + n)), torch.zeros(2), torch.zeros(2), None, 0, -1,
                              rows=[torch.randn(n, 4)], nbytes=n * 16) for b in (1000, 2000))
    for snap in (target, other):
        store.save(snap.ids, snap, skip=GlmEngine._SPILL_SKIP).done.wait(10)
    size = store.entry(spill.ids_key(target.ids)).size
    store.cap = store._total() + size // 2           # the store is full: a third entry evicts the oldest, ``target``
    e = _bare(store, None)
    e.e = SimpleNamespace(st=SimpleNamespace(kc=[torch.randn(256, 4)], vc=[None], index=None))
    e._gather_ints = lambda v: [list(v), list(v)]
    live = Snapshot(list(range(3000, 3000 + n)), torch.zeros(2), torch.zeros(2), None, 0, -1)
    e.cache, e.live, e.cache_bytes = [live], list(live.ids) + [7], 0       # making room writes the live state first
    back = GlmEngine._spill_load(e, list(target.ids), MTP)
    assert back is not None and store.stats.load_failed == 0 and store.has(live.ids)


def test_early_writes_never_evict_copies_of_snapshots_still_kept(tmp_path):
    store = _store(tmp_path)
    e = _bare(store, None)                                            # no drafter: saved rows can be written
    e._share, e._held_bytes, e.cache_bytes = (lambda v: v), (lambda: 1 << 30), 1    # past the high-water mark
    snaps = [_snap(16) for _ in range(3)]
    for i, snap in enumerate(snaps):
        snap.ids = [1000 * i + t for t in range(16)]
    e.cache = list(snaps)
    GlmEngine._spill_behind(e)                                        # two early writes a call
    assert store.drain(10) and store.has(snaps[0].ids) and store.has(snaps[1].ids)
    store.cap = store._total() + 1                                    # a third fits only by evicting one of them
    GlmEngine._spill_behind(e)                                        # so it is not written (they would take turns)
    assert store.drain(10) and store.has(snaps[0].ids) and store.has(snaps[1].ids) and not store.has(snaps[2].ids)
    assert store.stats.full == 1


def test_the_flush_writes_the_newest_snapshot_under_a_full_cap(tmp_path):
    from tensorfold.families.glm5_next.cuda.decode import Snapshot

    store = _store(tmp_path, classes=(Snapshot,))
    e = _bare(store, None)
    e._share, e._ring, e.e = (lambda v: v), (lambda: None), None
    snaps = [Snapshot([1000 * i + t for t in range(16)], torch.zeros(2), torch.zeros(2), None, 0, -1,
                      rows=[torch.randn(16, 4)], nbytes=256) for i in range(3)]     # oldest first, as kept
    for snap in snaps[:2]:                                            # the two oldest are on disk already
        store.save(snap.ids, snap, skip=GlmEngine._SPILL_SKIP).done.wait(10)
    store.cap = store._total() + 1                                    # and the store is full
    e.cache = list(snaps)
    assert GlmEngine.flush_spill(e, 10.0) is not False
    assert store.has(snaps[2].ids) and not store.has(snaps[0].ids)   # the newest replaces the oldest

