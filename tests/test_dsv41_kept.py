"""Kept prompts in the DeepSeek-V4.1 shared pool, host side: two MultiDecoders (rank 0 deciding, rank 1 replaying
its messages in its own thread) over a fake engine stay in step, resume kept states and evict them only when that
makes room; with the NVMe tier on (``kvdisk``), spill evicted states and restore them on both ranks or neither."""

import queue
import random
import threading

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import multi as M  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.pool import ALIGN, Pool  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.serial import DRING  # noqa: E402


@pytest.fixture(autouse=True)
def host_sampling(monkeypatch):
    monkeypatch.setattr(M, "sample_rows", lambda logits, positions, sampling: [0] * len(positions))


class FakeView:
    def __init__(self) -> None:
        self.ids: list[int] = []


class FakeEngine:
    """The surface MultiDecoder uses: slots with views and extents, a pool arena of token ids (row t of an extent
    holds the token its stream wrote there), window rings by bank entry (a slot's last DRING tokens, -1 padded), and
    ``kept_views`` of both as raw bytes."""

    def __init__(self, slots: int, pool_tokens: int, span: int, banks: int) -> None:
        self.slots, self.span, self.limit, self.pool_tokens = slots, span, span - 1, pool_tokens
        self.drafter = None
        self.c = type("C", (), {"eos_token_id": 1})()
        self.views = [FakeView() for _ in range(slots)]
        self.extents = [(0, 0)] * slots
        self.ring_from = [0] * slots
        self.deep_from = [0] * slots
        self.tail_min = 2702
        self.bank = [torch.full((banks * DRING, 1), -1, dtype=torch.int64)]
        self.arena = torch.full((pool_tokens,), -1, dtype=torch.int64)
        self.ring: dict[int, tuple] = {}
        self.slot = 0
        self.state = self.views[0]
        self.pool = None

    def bind(self, slot, base, size, ids=None):
        assert size <= self.span and base % ALIGN == 0
        self.extents[slot] = (base, size)
        self.views[slot] = FakeView()
        self.views[slot].ids = list(ids or [])
        if slot == self.slot:
            self.state = self.views[slot]
        ids = self.views[slot].ids                              # a resumed prefix: the rows must hold it
        assert self.arena[base:base + len(ids)].tolist() == ids, slot

    def select_slot(self, slot):
        self.slot = slot
        self.state = self.views[slot]

    def reset(self):
        self.state.ids.clear()
        self.ring_from[self.slot] = 0
        self.ring[self.slot] = ()

    def prefill(self, tokens, final=None):
        base, size = self.extents[self.slot]
        p0 = len(self.state.ids)
        assert p0 + len(tokens) <= size
        self.arena[base + p0:base + p0 + len(tokens)] = torch.tensor(list(tokens), dtype=torch.int64)
        self.state.ids.extend(tokens)
        self.ring[self.slot] = tuple(self.state.ids[-DRING:])
        self.ring_from[self.slot] = max(self.ring_from[self.slot], len(self.state.ids) - DRING)
        return torch.zeros((len(tokens), 8))

    def save_window(self, slot, k):
        block = self.bank[0][k * DRING:(k + 1) * DRING, 0]
        block.fill_(-1)
        ring = self.ring[slot]
        block[:len(ring)] = torch.tensor(list(ring), dtype=torch.int64)

    def load_window(self, k, slot):
        self.ring[slot] = self.window(k)

    def window(self, k):
        return tuple(t for t in self.bank[0][k * DRING:(k + 1) * DRING, 0].tolist() if t >= 0)

    def copy_rows(self, src, dst, n):
        self.arena[dst:dst + n] = self.arena[src:src + n].clone()

    def kept_views(self, base, n, bank):
        return [("comp/0", self.arena[base:base + n].view(torch.uint8)),
                ("bank/0", self.bank[0][bank * DRING:(bank + 1) * DRING].view(torch.uint8))]

    def reusable(self, prompt):
        return 0


IDENT = {"test": "dsv41-kept"}


def disks(root, budget_gib=1.0, min_tokens=1000):
    from tensorfold.families.deepseek_v41.cuda.kvdisk import KeptDisk

    out = []
    for rank in (0, 1):
        d = KeptDisk(root, rank, budget_gib=budget_gib, min_tokens=min_tokens, stage_mib=8, quiet=True, threads=2)
        d.attach(IDENT)
        out.append(d)
    return out


def pair(slots=4, rows=16 * ALIGN, span=4 * ALIGN, banks=6, disk=None):
    """Rank 0 here, rank 1 replaying its messages in a thread (``d1.sync()``: wait until it caught up); both agree
    through a barrier all-gather. ``disk``: the two ranks' ``KeptDisk`` (``disks``), or None."""

    q: queue.Queue = queue.Queue()
    acks: queue.Queue = queue.Queue()
    bar, box = threading.Barrier(2), [None, None]
    sent: list[list[int]] = []                                  # rank 0's messages (``ops``)

    def gather(rank):
        def g(values):
            box[rank] = list(values)
            bar.wait(timeout=30)
            out = [list(box[0]), list(box[1])]
            bar.wait(timeout=30)
            return out
        return g

    def send(v):
        if v:
            sent.append(list(v))
        q.put(list(v))
        return v

    def recv(_):
        return q.get(timeout=120)

    e0, e1 = FakeEngine(slots, rows, span, banks), FakeEngine(slots, rows, span, banks)
    d0 = M.MultiDecoder(e0, lambda v: send(v) if v is not None else None, rank=0, drafts=0, pool=Pool(rows),
                        gather=gather(0), disk=disk[0] if disk else None)
    d1 = M.MultiDecoder(e1, recv, rank=1, drafts=0, pool=Pool(rows), gather=gather(1),
                        disk=disk[1] if disk else None)

    def follow():
        while True:
            try:
                d1.follow()                                     # returns at each [] (a sync point)
            except BaseException as exc:                        # noqa: BLE001
                acks.put(exc)
                return
            acks.put(None)

    def caught_up():
        err = acks.get(timeout=120)
        if err is not None:
            raise err

    threading.Thread(target=follow, daemon=True).start()
    d1.sync = lambda: (q.put([]), caught_up())
    d1.caught_up = caught_up
    d0.sent = sent
    return d0, d1


def ops(d0, op: int) -> list[int]:
    """Where rank 0 sent message ``op`` (RESTORE: 11 words, ADMIT: 22, PERSIST: 1; prompts are longer)."""

    size = {M.RESTORE: 11, M.ADMIT: 12 + M.SAMPLING_WORDS, M.PERSIST: 1}[op]
    return [i for i, m in enumerate(d0.sent) if m[0] == op and len(m) == size]


def same(d0, d1):
    """Both ranks' pools, kept states, banks and disk indexes equal; every kept state's rows and bank entry hold its
    tokens (restored ones too); every live stream's rows hold its tokens."""

    d1.sync()
    assert d0.pool.digest() == d1.pool.digest()
    assert [(k.kid, k.n, k.vs, k.vd, k.bank, k.x.eid, k.disk) for k in d0.kept] == \
        [(k.kid, k.n, k.vs, k.vd, k.bank, k.x.eid, k.disk) for k in d1.kept]
    assert d0.banks == d1.banks and d0.next_kid == d1.next_kid
    for d in (d0, d1):
        e = d.e
        for k in d.kept:
            assert e.arena[k.x.base:k.x.base + k.n].tolist() == k.ids.tolist(), k.kid
            assert e.window(k.bank) == tuple(k.ids[-DRING:].tolist()), k.kid
        for s in d.streams.values():
            base, _ = e.extents[s.slot]
            assert e.arena[base:base + len(e.views[s.slot].ids)].tolist() == e.views[s.slot].ids
    if d0.disk is not None:
        assert d0.disk.keys() == d1.disk.keys()
        assert [(e.n, e.vs, e.vd, e.size) for e in d0.disk.index.values()] == \
            [(e.n, e.vs, e.vd, e.size) for e in d1.disk.index.values()]


def run(d0, s):
    """Admit ``s`` on rank 0 and prefill its prompt (FILL messages), decoding nothing."""

    d0.admit(s)
    while s in d0.filling:
        d0._fill()


def test_resume_takeover_and_copy():
    d0, d1 = pair()
    rng = random.Random(0)
    p = [rng.randrange(2, 1000) for _ in range(3000)]
    a = Stream(list(p), 50)
    run(d0, a)
    same(d0, d1)
    assert sorted(k.n for k in d0.kept) == [2048, 3000]       # its last chunk start and its end
    b = Stream(list(p), 50)                                     # while a still owns its extent: copied
    run(d0, b)
    assert b.cached == len(p) - 1 and d0.kstats["copies"] == 1
    same(d0, d1)
    d0.finish([a, b])
    same(d0, d1)
    c = Stream(list(p[:2950]) + [5] * 40, 50)                  # a's extent is free now: taken over
    run(d0, c)
    assert c.cached == 2950 and d0.kstats["takeovers"] == 1
    same(d0, d1)


def test_eviction_only_when_it_makes_room():
    d0, d1 = pair(rows=8 * ALIGN, span=4 * ALIGN, banks=40)
    rng = random.Random(1)
    prompts = [[rng.randrange(2, 1000) for _ in range(3000)] for _ in range(9)]
    for p in prompts:
        s = Stream(list(p), 20)
        run(d0, s)
        d0.finish([s])
        same(d0, d1)
    assert d0.kstats["evictions"] > 0 and d0.kept
    assert all(x.owner is None for x in d0.pool.extents)
    from tensorfold.cuda.memory_gate import NoRoom

    for p in prompts[:3]:                                       # long-lived streams: whole-span extents
        before = (len(d0.kept), d0.kstats["evictions"])
        try:
            run(d0, Stream(list(p) + [3], 4000))
        except NoRoom:                                          # no run even without every kept state:
            assert (len(d0.kept), d0.kstats["evictions"]) == before   # nothing was evicted for it
        same(d0, d1)
    assert any(x.owner is not None for x in d0.pool.extents)


def test_random_traffic_stays_in_step():
    from tensorfold.cuda.memory_gate import NoRoom

    d0, d1 = pair(slots=3, rows=12 * ALIGN, span=4 * ALIGN, banks=4)
    rng = random.Random(2)
    roots = [[rng.randrange(2, 1000) for _ in range(rng.randrange(500, 3500))] for _ in range(4)]
    live: list[Stream] = []
    for step in range(200):
        if live and (rng.random() < 0.4 or len(live) == 3):
            d0.finish([live.pop(rng.randrange(len(live)))])
        else:
            root = rng.choice(roots)
            cut = rng.randrange(len(root) // 2, len(root))
            p = root[:cut] + [rng.randrange(2, 1000) for _ in range(rng.randrange(1, 300))]
            s = Stream(p[:7000], rng.randrange(1, 1500))
            try:
                run(d0, s)
                live.append(s)
            except NoRoom:
                pass
        same(d0, d1)
    assert d0.kstats["hits"] > 0


def decode(d0, d1, s, k):
    """``k`` decoded tokens of stream ``s`` written on both ranks (the rows a round would write), after rank 0 made
    room for them (GROW/MOVE/EVICT replayed by rank 1)."""

    ended = d0._make_room([x for x in d0.streams.values() if not x.done])
    d1.sync()
    d0.ended = getattr(d0, "ended", []) + ended
    if s.waiting or s.done:
        return False
    for d in (d0, d1):
        e = d.e
        st = d.streams[s.sid]
        base, size = e.extents[st.slot]
        ids = e.views[st.slot].ids
        assert len(ids) + k <= size
        for j in range(k):
            e.arena[base + len(ids)] = 900 + (len(ids) % 50)
            ids.append(900 + (len(ids) % 50))
    return True


def test_growth_in_place_and_by_move(monkeypatch):
    monkeypatch.setattr(M, "GROW_AHEAD", 2048)
    d0, d1 = pair(slots=3, rows=16 * ALIGN, span=6 * ALIGN)
    rng = random.Random(3)
    a = Stream([rng.randrange(2, 1000) for _ in range(1500)], 9000)
    run(d0, a)
    b = Stream([rng.randrange(2, 1000) for _ in range(1500)], 9000)
    run(d0, b)                                                  # right after a: a must move to grow
    same(d0, d1)
    assert d0.ext[a.sid].size == 2 * ALIGN
    for _ in range(270):                                        # a round writes at most ROWS rows
        assert decode(d0, d1, a, 30) and decode(d0, d1, b, 30)
        same(d0, d1)
    assert d0.kstats.get("moves", 0) >= 1 and d0.kstats.get("grows", 0) >= 1
    for d in (d0, d1):                                          # the moved rows still hold the stream's tokens
        for s in d.streams.values():
            base, _ = d.e.extents[s.slot]
            assert d.e.arena[base:base + len(d.e.views[s.slot].ids)].tolist() == d.e.views[s.slot].ids


def test_no_room_to_grow_ends_the_newest_and_yield_for(monkeypatch):
    monkeypatch.setattr(M, "GROW_AHEAD", 2048)
    d0, d1 = pair(slots=3, rows=6 * ALIGN, span=6 * ALIGN)
    rng = random.Random(4)
    a = Stream([rng.randrange(2, 1000) for _ in range(1500)], 12000)
    b = Stream([rng.randrange(2, 1000) for _ in range(1500)], 12000)
    b.background = True
    run(d0, a)
    run(d0, b)
    same(d0, d1)
    fg = Stream([5] * 1500, 100)
    assert d0.yield_for(fg) == []                               # room already: nothing yields
    c = Stream([rng.randrange(2, 1000) for _ in range(1500)], 12000)
    run(d0, c)
    assert d0.yield_for(Stream([6] * 1500, 100)) == [b]         # the background stream would make room
    for _ in range(600):
        for st in [x for x in (a, b, c) if not x.done and x not in d0.yielded]:
            decode(d0, d1, st, 30)
        if d0.yielded or c.done:
            break
    assert d0.yielded == [b] and not b.done and c.error is None   # the background stream gives way first
    d0.finish([b])                                              # (the scheduler re-queues its replay)
    same(d0, d1)
    for _ in range(600):                                        # no background stream left
        for st in [x for x in (a, c) if not x.done]:
            decode(d0, d1, st, 30)
        if c.done:
            break
    ended = d0.ended
    assert ended == [c] and c.error is not None and not a.done  # the newest ends, alone
    d0.finish(ended)
    same(d0, d1)


# -- the NVMe tier (kvdisk): spills on eviction, restores on admission, on both ranks or neither ------------------------
def corrupt(d, key):
    """Flip a byte inside entry ``key``'s cache rows (a chunk checksum catches it)."""

    p = d.path(key)
    head, base, _ = d._read_header(p)
    seg = next(s for s in head["segments"] if s["name"] == "comp/0")
    raw = bytearray(p.read_bytes())
    raw[base + seg["offset"] + 10] ^= 0xFF
    p.write_bytes(bytes(raw))


def spilled(tmp_path, monkeypatch):
    """p (3000 tokens) kept at 2048 and 3000, then p[:1200] kept, then an unrelated prompt evicting p's states
    (spilled to disk on both ranks). Pool: two 4096-row extents."""

    monkeypatch.setattr(M, "DISK_GAIN", 500)
    disk = disks(tmp_path)
    d0, d1 = pair(slots=2, rows=4 * ALIGN, span=4 * ALIGN, banks=8, disk=disk)
    rng = random.Random(7)
    p = [rng.randrange(2, 1000) for _ in range(3000)]
    other = [rng.randrange(2, 1000) for _ in range(3000)]
    for prompt in (p, p[:1200], other):
        s = Stream(list(prompt), 20)
        run(d0, s)
        d0.finish([s])
        same(d0, d1)
    key = disk[0].key(p)
    assert key in disk[0].index and disk[0].key(p[:2048]) in disk[0].index and d0.kstats["spills"] == 2
    assert not any(k.n == 3000 and k.ids.tolist() == p for k in d0.kept)
    disk[0].drain()
    disk[1].drain()
    return d0, d1, disk, p, key


def test_disk_spill_on_eviction_and_restore_on_admission(tmp_path, monkeypatch):
    d0, d1, _, p, key = spilled(tmp_path, monkeypatch)
    d0.sent.clear()
    s = Stream(list(p) + [5] * 40, 20)
    run(d0, s)                                  # the pool holds p[:1200]; the disk p's 3000 tokens: restored
    assert len(ops(d0, M.RESTORE)) == 1 and ops(d0, M.RESTORE)[0] < ops(d0, M.ADMIT)[0]
    assert s.cached == 3000 and d0.kstats["disk_hits"] == d1.kstats["disk_hits"] == 1
    assert d0.kstats["takeovers"] == 1 and d0.kstats["disk_bad"] == 0
    restored = next(k for k in d0.kept if k.disk == key)
    assert restored.n == 3000 and restored.ids.tolist() == p     # (same(): its rows and window hold p, both ranks)
    same(d0, d1)
    d0.finish([s])
    same(d0, d1)


@pytest.mark.parametrize("bad", [0, 1])
def test_disk_failed_restore_on_one_rank(tmp_path, monkeypatch, bad):
    d0, d1, disk, p, key = spilled(tmp_path, monkeypatch)
    corrupt(disk[bad], key)
    d0.sent.clear()
    s = Stream(list(p) + [5] * 40, 20)
    run(d0, s)
    assert len(ops(d0, M.RESTORE)) == 1
    assert d0.kstats["disk_bad"] == d1.kstats["disk_bad"] == 1 and d0.kstats["disk_hits"] == 0
    assert key not in disk[0].index and key not in disk[1].index
    assert not disk[0].path(key).exists() and not disk[1].path(key).exists()
    # the pool's hit (p[:1200]) was protected from the restore's evictions: resumed from it
    assert s.cached == 1200 and any(k.ids.tolist() == p[:1200] for k in d0.kept)
    same(d0, d1)
    d0.finish([s])
    same(d0, d1)


def test_disk_regenerate_is_not_spilled(tmp_path):
    disk = disks(tmp_path)
    d0, d1 = pair(slots=2, rows=8 * ALIGN, span=4 * ALIGN, banks=8, disk=disk)
    rng = random.Random(8)
    p = [rng.randrange(2, 1000) for _ in range(3000)]
    a = Stream(list(p), 20)
    run(d0, a)
    d0.finish([a])
    same(d0, d1)
    b = Stream(list(p), 20)                     # the same prompt again (a regenerated reply): kept again, not spilled
    run(d0, b)
    assert d0.kstats["takeovers"] == 1 and d0.kstats["spills"] == 0 and not disk[0].keys()
    d0.finish([b])
    same(d0, d1)
    c = Stream(list(p[:2950]) + [5] * 100, 20)  # a branch inside p's state: p's rows are overwritten, so spilled
    run(d0, c)
    assert d0.kstats["takeovers"] == 2 and d0.kstats["spills"] == 1 and disk[0].keys() == [disk[0].key(p)]
    same(d0, d1)


def test_disk_persist_at_shutdown_and_restart(tmp_path, monkeypatch):
    from tensorfold.families.deepseek_v41.cuda import kvdisk

    monkeypatch.setattr(M, "DISK_GAIN", 500)
    disk = disks(tmp_path)
    d0, d1 = pair(slots=2, rows=8 * ALIGN, span=4 * ALIGN, banks=8, disk=disk)
    rng = random.Random(9)
    p = [rng.randrange(2, 1000) for _ in range(3000)]
    a = Stream(list(p), 20)
    run(d0, a)
    b = Stream([rng.randrange(2, 1000) for _ in range(2500)], 20)
    run(d0, b)
    d0.finish([b])
    same(d0, d1)
    d0.sent.clear()
    assert d0.shutdown() == [a]                 # live streams ended, every kept state written on both ranks
    d1.caught_up()                              # rank 1 left follow at the end message
    assert [m[0] for m in d0.sent] == [M.DONE, M.PERSIST]
    kept = sorted(disk[0].key(k.ids) for k in d0.kept)
    assert len(kept) == 3 and sorted(disk[0].keys()) == sorted(disk[1].keys()) == kept   # (b: no split keep)
    for d in disk:
        d.close()
    disk[1].path(disk[1].key(p[:2048])).unlink()                 # (rank 1 lost one: the restart drops it on both)

    again = disks(tmp_path)
    bar, box = threading.Barrier(2), [None, None]

    def gather(rank):
        def g(values):
            box[rank] = list(values)
            bar.wait(timeout=30)
            out = [list(box[0]), list(box[1])]
            bar.wait(timeout=30)
            return out
        return g

    t = threading.Thread(target=kvdisk.intersect, args=(again[1], gather(1)))
    t.start()
    assert kvdisk.intersect(again[0], gather(0)) == 2
    t.join(30)
    assert again[0].keys() == again[1].keys() and disk[0].key(p[:2048]) not in again[0].index
    d0, d1 = pair(slots=2, rows=8 * ALIGN, span=4 * ALIGN, banks=8, disk=again)
    s = Stream(list(p) + [5] * 40, 20)
    run(d0, s)                                  # an empty pool: resumed from the disk
    assert s.cached == 3000 and d0.kstats["disk_hits"] == 1
    same(d0, d1)


def test_random_traffic_with_disk_stays_in_step(tmp_path, monkeypatch):
    from tensorfold.cuda.memory_gate import NoRoom

    monkeypatch.setattr(M, "DISK_GAIN", 1)
    disk = disks(tmp_path, budget_gib=0.6 / 1024, min_tokens=300)      # ~12 entries: trimmed as it goes
    d0, d1 = pair(slots=3, rows=12 * ALIGN, span=4 * ALIGN, banks=4, disk=disk)
    rng = random.Random(2)
    roots = [[rng.randrange(2, 1000) for _ in range(rng.randrange(500, 3500))] for _ in range(6)]
    live: list[Stream] = []
    for step in range(200):
        if live and (rng.random() < 0.4 or len(live) == 3):
            d0.finish([live.pop(rng.randrange(len(live)))])
        else:
            root = rng.choice(roots)
            cut = rng.randrange(len(root) // 2, len(root))
            p = root[:cut] + [rng.randrange(2, 1000) for _ in range(rng.randrange(1, 300))]
            s = Stream(p[:7000], rng.randrange(1, 1500))
            try:
                run(d0, s)
                live.append(s)
            except NoRoom:
                pass
        same(d0, d1)
        if step % 37 == 36 and disk[1].keys():                  # rank 1 loses a file: a restore of it fails on both
            disk[1].drain()
            disk[1].path(rng.choice(disk[1].keys())).unlink(missing_ok=True)
    assert d0.kstats["spills"] > 0 and d0.kstats["disk_hits"] > 0
