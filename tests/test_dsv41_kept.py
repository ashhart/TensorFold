"""Kept prompts in the DeepSeek-V4.1 shared pool, host side: two MultiDecoders (rank 0 deciding, rank 1 replaying
its messages) over a fake engine stay in step, resume kept states and evict them only when that makes room."""

import random

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
    holds the token its stream wrote there), window rings by bank entry."""

    def __init__(self, slots: int, pool_tokens: int, span: int, banks: int) -> None:
        self.slots, self.span, self.limit, self.pool_tokens = slots, span, span - 1, pool_tokens
        self.drafter = None
        self.c = type("C", (), {"eos_token_id": 1})()
        self.views = [FakeView() for _ in range(slots)]
        self.extents = [(0, 0)] * slots
        self.bank = [torch.zeros((banks * DRING,))]
        self.arena = [-1] * pool_tokens
        self.saved: dict[int, tuple] = {}
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
        for t, tok in enumerate(self.views[slot].ids):         # a resumed prefix: the rows must hold it
            assert self.arena[base + t] == tok, (slot, t)

    def select_slot(self, slot):
        self.slot = slot
        self.state = self.views[slot]

    def reset(self):
        self.state.ids.clear()
        self.ring[self.slot] = ()

    def prefill(self, tokens):
        base, size = self.extents[self.slot]
        p0 = len(self.state.ids)
        assert p0 + len(tokens) <= size
        for j, t in enumerate(tokens):
            self.arena[base + p0 + j] = t
        self.state.ids.extend(tokens)
        self.ring[self.slot] = tuple(self.state.ids[-DRING:])
        return torch.zeros((len(tokens), 8))

    def save_window(self, slot, k):
        self.saved[k] = self.ring[slot]

    def load_window(self, k, slot):
        self.ring[slot] = self.saved[k]

    def copy_rows(self, src, dst, n):
        self.arena[dst:dst + n] = self.arena[src:src + n]

    def reusable(self, prompt):
        return 0


def pair(slots=4, rows=16 * ALIGN, span=4 * ALIGN, banks=6):
    q: list[list[int]] = []

    def send(v):
        q.append(list(v))
        return v

    def recv(_):
        return q.pop(0) if q else []

    e0, e1 = FakeEngine(slots, rows, span, banks), FakeEngine(slots, rows, span, banks)
    d0 = M.MultiDecoder(e0, lambda v: send(v) if v is not None else None, rank=0, drafts=0, pool=Pool(rows))
    d1 = M.MultiDecoder(e1, recv, rank=1, drafts=0, pool=Pool(rows))
    return d0, d1


def same(d0, d1):
    d1.follow()
    assert d0.pool.digest() == d1.pool.digest()
    assert [(k.kid, k.n, k.bank, k.x.eid) for k in d0.kept] == [(k.kid, k.n, k.bank, k.x.eid) for k in d1.kept]
    assert d0.banks == d1.banks and d0.e.arena == d1.e.arena


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
    assert len(d0.kept) == 1 and d0.kept[0].n == 2048         # kept at the prompt's last chunk start
    b = Stream(list(p), 50)                                     # while a still owns its extent: copied
    run(d0, b)
    assert b.cached == 2048 and d0.kstats["copies"] == 1
    same(d0, d1)
    d0.finish([a, b])
    same(d0, d1)
    c = Stream(list(p[:2950]) + [5] * 40, 50)                  # a's extent is free now: taken over
    run(d0, c)
    assert c.cached == 2048 and d0.kstats["takeovers"] == 1
    same(d0, d1)


def test_eviction_only_when_it_makes_room():
    d0, d1 = pair(rows=8 * ALIGN, span=4 * ALIGN)
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
    d1.follow()
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
            assert d.e.arena[base:base + len(d.e.views[s.slot].ids)] == d.e.views[s.slot].ids


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
