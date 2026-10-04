"""The NVMe tier of DeepSeek-V4.1's kept prompts (``kvdisk.KeptDisk``), host side: numpy views stand in for the
engine's device views (no torch or triton needed)."""

import os

import numpy as np
import pytest

from tensorfold.families.deepseek_v41.cuda import kvdisk as D

IDENT = {"model": "test", "layout": [1, 2, 3]}
OK = lambda m, plen, vs, vd: True


def views(seed: int, n: int, big: int = 0) -> list:
    """Stand-ins for ``kept_views``: an Fp8Rows-like triple, a bf16-like plane (as uint8) and a bank block."""

    rng = np.random.default_rng(seed)
    out = [("comp/2/q", rng.integers(0, 256, (n // 2, 448), dtype=np.uint8)),
           ("comp/2/r", rng.integers(0, 256, (n // 2, 128), dtype=np.uint8)),
           ("ik/2", rng.integers(0, 256, (n // 2, 130), dtype=np.uint8)),
           ("bank/0", rng.integers(0, 256, (256, 1024), dtype=np.uint8))]
    if big:
        out.append(("bank/1", rng.integers(0, 256, (big, 1 << 20), dtype=np.uint8)))
    return out


def blank(vs: list) -> list:
    return [(name, np.zeros_like(v)) for name, v in vs]


def disk(tmp_path, rank=0, ident=IDENT, **kw) -> D.KeptDisk:
    kw.setdefault("stage_mib", 64)
    d = D.KeptDisk(tmp_path, rank, quiet=True, threads=4, **kw)
    d.attach(ident)
    return d


def ids_of(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(1000 + seed).integers(2, 100000, n).astype(np.int64)


@pytest.mark.parametrize("direct", [True, False])
def test_round_trip(tmp_path, direct):
    d = disk(tmp_path, direct=direct)
    ids = ids_of(3000)
    src = views(1, 3000)
    key = d.key(ids)
    assert d.spill(key, 3000, 2800, 2810, ids, src)
    assert not d.spill(key, 3000, 2800, 2810, ids, src)           # indexed already: touched only
    d.drain()
    assert d.index[key].state == D.OK and d.path(key).exists()
    assert os.path.getsize(d.path(key)) % 4096 == 0
    dst = blank(src)
    got = d.restore(key, 3000, 2800, 2810, dst)
    assert got is not None and np.array_equal(got, ids) and got.dtype == np.int64
    for (_, a), (_, b) in zip(src, dst):
        assert np.array_equal(a, b)
    assert d.restore(key, 3000, 2800, 2811, blank(src)) is None   # not the state asked for
    assert d.restore(key, 3000, 2800, 2810, blank(src)[:-1]) is None   # views of another layout


def test_entry_larger_than_the_stage(tmp_path):
    d = disk(tmp_path, stage_mib=8)
    ids = ids_of(4096)
    src = views(2, 4096, big=20)                                  # ~22 MiB through an 8 MiB stage
    key = d.key(ids)
    assert d.spill(key, 4096, 3900, 3900, ids, src)
    assert d.index[key].state == D.OK                             # streamed, written before spill returned
    dst = blank(src)
    assert np.array_equal(d.restore(key, 4096, 3900, 3900, dst), ids)
    for (_, a), (_, b) in zip(src, dst):
        assert np.array_equal(a, b)


def test_many_queued_spills_wrap_the_stage(tmp_path):
    d = disk(tmp_path, stage_mib=8)
    src = {i: views(10 + i, 4000) for i in range(12)}            # ~2.4 MiB each: the ring wraps several times
    keys = {}
    for i in range(12):
        ids = ids_of(4000, i)
        keys[i] = d.key(ids)
        d.spill(keys[i], 4000, 3800, 3800, ids, src[i])
    d.drain()
    for i in range(12):
        dst = blank(src[i])
        assert d.restore(keys[i], 4000, 3800, 3800, dst) is not None
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(src[i], dst))


def test_corruption_is_detected(tmp_path):
    d = disk(tmp_path)
    ids = ids_of(3000)
    src = views(3, 3000)
    key = d.key(ids)
    d.spill(key, 3000, 2800, 2800, ids, src)
    d.drain()
    p = d.path(key)
    raw = bytearray(p.read_bytes())
    raw[-5000] ^= 0x40                                            # a byte of the bank block
    p.write_bytes(bytes(raw))
    assert d.restore(key, 3000, 2800, 2800, blank(src)) is None
    assert d.stats["bad"] == 1
    d.delete(key)
    assert not p.exists() and key not in d.index

    # header damage, truncation and leftovers are dropped at reconcile
    names = []
    for i in range(3):
        ids = ids_of(3000, 10 + i)
        k = d.key(ids)
        d.spill(k, 3000, 2800, 2800, ids, views(20 + i, 3000))
        names.append(k)
    d.drain()
    p0, p1 = d.path(names[0]), d.path(names[1])
    raw = bytearray(p0.read_bytes())
    raw[40] ^= 0x01                                               # inside the header JSON
    p0.write_bytes(bytes(raw))
    with open(p1, "r+b") as fh:
        fh.truncate(os.path.getsize(p1) - 8192)
    (d.dir / "deadbeef.tmp").write_bytes(b"x" * 100)
    (d.dir / ("00" * 16 + ".tfk")).write_bytes(b"junk")
    again = disk(tmp_path)
    assert again.keys() == [names[2]]
    assert sorted(x.name for x in again.dir.iterdir()) == [f"{names[2].hex()}.tfk"]


def test_budget_and_lru(tmp_path):
    src = views(4, 3000)
    size = 4096 * 3 + sum(-(-v.nbytes // 4096) * 4096 for _, v in src)   # ids (12 KB) + the views
    d = disk(tmp_path, budget_gib=3.5 * size / 2 ** 30)
    keys = []
    for i in range(3):
        ids = ids_of(3000, i)
        keys.append(d.key(ids))
        d.spill(keys[-1], 3000, 2800, 2800, ids, src)             # pending entries: trimmed at enqueue time
    d.touch(keys[0])                                              # most recent now
    ids = ids_of(3000, 3)
    k3 = d.key(ids)
    d.spill(k3, 3000, 2800, 2800, ids, src)
    assert d.keys() == [keys[2], keys[0], k3]                     # the least recent (keys[1]) went
    assert not d.path(keys[1]).exists()
    d.drain()
    assert all(d.path(k).exists() for k in d.index)
    small = disk(tmp_path / "small", budget_gib=size / 4 / 2 ** 30)
    assert not small.spill(k3, 3000, 2800, 2800, ids, src) and not small.index   # larger than the budget


def test_reconcile_after_restart_and_compat(tmp_path):
    d = disk(tmp_path)
    keys, srcs = [], []
    for i in range(3):
        ids = ids_of(3000, i)
        keys.append(d.key(ids))
        srcs.append(views(30 + i, 3000))
        d.spill(keys[-1], 3000, 2800, 2800, ids, srcs[-1])
        d.drain()
        os.utime(d.path(keys[-1]), ns=(10 ** 18 + i, 10 ** 18 + i))
    d.close()
    again = disk(tmp_path)
    assert again.keys() == keys                                   # oldest modified first
    dst = blank(srcs[1])
    assert np.array_equal(again.restore(keys[1], 3000, 2800, 2800, dst), ids_of(3000, 1))
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(srcs[1], dst))
    other = disk(tmp_path, ident={**IDENT, "knob": 1})            # another build: its own directory
    assert other.dir != d.dir and not other.keys()
    assert disk(tmp_path, rank=1).keys() == []                    # and each rank its own


def test_retain_keeps_the_given_order(tmp_path):
    d = disk(tmp_path)
    keys = []
    for i in range(4):
        ids = ids_of(3000, i)
        keys.append(d.key(ids))
        d.spill(keys[-1], 3000, 2800, 2800, ids, views(40, 3000))
    d.retain([keys[3], keys[0], b"\x01" * 16, keys[2]])
    assert d.keys() == [keys[3], keys[0], keys[2]]
    assert not d.path(keys[1]).exists()


def test_find(tmp_path):
    d = disk(tmp_path)
    base = ids_of(5000)
    for n, vs in ((3000, 2800), (4000, 3790), (200, 0)):          # B = 2560 and 3584, and B = 0 (vs < 256)
        ids = base[:n]
        d.spill(d.key(ids), n, vs, vs, ids, views(50, max(n, 2)))
    p = list(base[:3900]) + [7, 7, 7]
    m, key, e = d.find(p, OK, span=8192)
    assert (m, key) == (3900, d.key(base[:4000])) and e.B == 3584 and len(e.tail) == 416
    m, key, _ = d.find(list(base[:3000]), OK, span=8192)        # (one token left to prefill; 4000's B is past it)
    assert (m, key) == (2999, d.key(base[:3000]))
    assert d.find(list(base[:3000]), OK, span=2048)[:2] == (200, d.key(base[:200]))   # 3000 -> 4096 rows: too long
    assert d.find(list(base[:2000]), OK, span=8192, least=1000) is None  # shorter than every B: only B = 0 (n 200)
    assert d.find(list(base[:2000]), OK, span=8192)[0] == 200
    q = list(base[:1000]) + [1] + list(base[1001:4000])           # differs inside the prefix digest
    assert d.find(q, OK, span=8192)[0] == 200
    assert d.find(p, lambda m, plen, vs, vd: m - 130 >= vs and m < 3500, span=8192)[0] == 3000
    other = np.concatenate([base[:2900], [5] * 100])
    d.spill(d.key(other), 3000, 2800, 2800, other, views(51, 3000))
    assert d.find(list(base[:2900]) + [9], OK, span=8192)[:2] == (2900, d.key(other))   # ties: the most recent
    assert d.find(p, lambda *a: False, span=8192) is None


def test_failed_write_stays_indexed(tmp_path):
    d = disk(tmp_path)
    ids = ids_of(3000)
    key = d.key(ids)
    os.chmod(d.dir, 0o500)
    try:
        if os.access(d.dir, os.W_OK):
            pytest.skip("running as root: permissions are not enforced")
        assert d.spill(key, 3000, 2800, 2800, ids, views(5, 3000))
        d.drain()
        assert d.index[key].state == D.FAILED                     # a miss later, the same on both ranks' indexes
        assert d.find(list(ids) + [3], OK, span=8192) is None
        assert d.restore(key, 3000, 2800, 2800, blank(views(5, 3000))) is None
    finally:
        os.chmod(d.dir, 0o755)
    d.delete(key)
    assert key not in d.index


def test_key_words_round_trip():
    k = D.KeptDisk.key(np.arange(3000))
    assert D.ints_key(*D.key_ints(k)) == k and len(k) == 16
    assert D.chain(np.arange(600))[1] == D.ids_digest(np.arange(512))


def gathers():
    """Two ranks' all-gather of equal-length int lists, for two threads."""

    import threading

    bar, box = threading.Barrier(2), [None, None]

    def make(rank):
        def gather(values):
            box[rank] = list(values)
            bar.wait(timeout=30)
            out = [list(box[0]), list(box[1])]
            bar.wait(timeout=30)
            return out
        return gather

    return make(0), make(1)


def test_restart_keeps_what_both_ranks_hold(tmp_path):
    import threading

    d0, d1 = disk(tmp_path, 0), disk(tmp_path, 1)
    keys = []
    for i in range(5):
        ids = ids_of(3000, i)
        keys.append(D.KeptDisk.key(ids))
        for d in (d0, d1):
            d.spill(keys[-1], 3000, 2800, 2800, ids, views(60 + i, 3000))
    for d in (d0, d1):
        d.drain()
        d.close()
    d1.path(keys[1]).unlink()                                     # rank 1 died before writing this one
    for i, k in enumerate(keys):                                  # rank 1 wrote them in another order
        if d1.path(k).exists():
            os.utime(d1.path(k), ns=(10 ** 18 + 10 - i, 10 ** 18 + 10 - i))
        os.utime(d0.path(k), ns=(10 ** 18 + i, 10 ** 18 + i))
    a, b = disk(tmp_path, 0), disk(tmp_path, 1)
    assert a.keys() == keys and b.keys() == [keys[4], keys[3], keys[2], keys[0]]
    g0, g1 = gathers()
    out = [None, None]
    t = threading.Thread(target=lambda: out.__setitem__(1, D.intersect(b, g1)))
    t.start()
    out[0] = D.intersect(a, g0)
    t.join(30)
    assert out == [4, 4]
    assert a.keys() == b.keys() == [keys[0], keys[2], keys[3], keys[4]]   # rank 0's order on both
    assert not a.path(keys[1]).exists()
    empty0, empty1 = disk(tmp_path / "e", 0), disk(tmp_path / "e", 1)
    t = threading.Thread(target=lambda: D.intersect(empty1, g1))
    t.start()
    assert D.intersect(empty0, g0) == 0
    t.join(30)
