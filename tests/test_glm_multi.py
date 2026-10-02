"""GLM's concurrent streams (--parallel, ``glm5_next.cuda.multi``) on CPU, two ranks through recorded messages.

The model is a fake whose logits for a row are a hash of everything its stream committed, read back from the pool
(the arena rows of its extent: one row a token and one a pool of 4) and from its slot (a rolling "KDA state" with two
parities): a stream whose rows were misplaced, copied, moved or mixed with another's gives another reply. Each
concurrent request's reply must equal its request served alone with ``"draft": false`` (the real single-stream
``GlmEngine.generate`` on the same fake); both ranks must hold the same pool, kept prompts, slots and arenas after
every iteration. The fake DFlash2 drafter reads its stream's committed rows from its own context (checking they
arrive once each, in order) and drafts the fake model's greedy chain, every third draft wrong. The fake MTP head
keeps its rows in the pool too (a third plane), checks each absorbed hidden row sits at its row's position, and
drafts the fake model's chain from what it absorbed, every third draft wrong; a stream's MTP rows must hold its
committed tokens.

Covered: staggered admission of mixed requests (greedy, top-k, nucleus, serial, DFlash2 and MTP drafts),
cancellation at a round boundary, pause (no room to grow) and the youngest given back to the queue to replay (its
sent tokens owed, not sent again), resumes from kept prompts (in place, and copied while a live stream holds the
extent; DFlash2 windows and MTP rows), evictions and compaction moves, one message an iteration (plus the IDLE one when
rank 0 goes idle), the speed settings, and the threaded scheduler with a follower thread. The real kernels behind the
same seams are in tests/cuda/test_glm_multi.py."""

from __future__ import annotations

import importlib
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so decode imports)

pytestmark = pytest.mark.torch

V, EOS, M = 48, 47, 2**31 - 1
ALIGN = 2048


def fold(h: int, pos: int, tok: int) -> int:
    return (h * 31 + pos * 1009 + tok + 1) % M


def row_logits(torch, h: int, S: int, P: int):
    base = (h ^ (S * 7) ^ (P * 13)) % M
    v = torch.arange(V, dtype=torch.int64)
    return ((((base % 100003) * (v + 11) + v * v * 17 + base % 7919) % 1009).float() / 61.0)


def pure_logits(torch, seq: list[int]):
    """The fake model's logits after ``seq`` (the prompt and every committed token), from the tokens alone."""

    h = 0
    for i, t in enumerate(seq):
        h = fold(h, i, t)
    q = len(seq) - 1
    S = sum((i + 1) * (t + 1) for i, t in enumerate(seq)) % M
    P = sum(t + 1 for t in seq[:4 * ((q + 1) // 4)])
    return row_logits(torch, h, S, P)


# -- the fake engine -------------------------------------------------------------------------------------------------
class FakeCaches:
    """The pool: per token its token + 1 (the "latent"), per pool of 4 their sum (the "pooled key", 2 rows of pad),
    and per token the MTP head's row: the next token + 1 it absorbed with that position's hidden row."""

    def __init__(self, torch, rows: int) -> None:
        from tensorfold.families.glm5_next.cuda.pool import Arena, Plane

        self.rows = rows
        self.arena = Arena(rows, [Plane(torch.zeros((rows, 1), dtype=torch.int64)),
                                  Plane(torch.zeros((rows // 4 + 2, 1), dtype=torch.int64), 4, 2),
                                  Plane(torch.zeros((rows, 1), dtype=torch.int64))])


class FakeSlots:
    def __init__(self, torch, count: int, rows: int) -> None:
        self.count = count
        self.rec = torch.zeros((count, 2, 1), dtype=torch.int64)
        self.conv = torch.zeros((count, 3), dtype=torch.int64)
        self.window: list[list[int]] = [[] for _ in range(count)]      # the last window's tokens, for the commit


class FakeState:
    """forward.State's surface: views of an extent of the pool and of a slot."""

    def __init__(self, w, capacity: int, rows: int, *, caches=None, base: int = 0, slots=None, slot: int = 0) -> None:
        import torch

        self.caches = caches if caches is not None else FakeCaches(torch, capacity)
        self.slots = slots if slots is not None else FakeSlots(torch, 1, rows)
        self.slot = slot
        self.rec = self.slots.rec[slot]
        self.conv = self.slots.conv[slot]
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0
        self.index, self.vc = None, [None]
        self.bind(base, capacity)

    def bind(self, base: int, capacity: int) -> None:
        self.base, self.capacity = base, capacity
        a = self.caches.arena
        self.lat, self.pk, self.mtpv = a.view(0, base, capacity), a.view(1, base, capacity), a.view(2, base, capacity)
        self.kc = [self.lat]

    @property
    def graph_key(self):
        return self.slot, self.base

    def reset(self) -> None:
        self.rec.zero_()
        self.conv.zero_()
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def set_pos(self, n: int) -> None:
        self.pos = n

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n

    def h(self) -> int:
        return int(self.rec[self.cur[0]][0])

    def write(self, q: int, tok: int) -> None:
        """The row at position q: its latent, and the pooled key of a pool it completes."""

        if q >= self.capacity:
            raise ValueError("context past the cache capacity")
        self.lat[q, 0] = tok + 1
        if q % 4 == 3:
            self.pk[q // 4, 0] = self.lat[q - 3:q + 1, 0].sum()

    def logits(self, q: int, h: int):
        import torch

        S = int(((torch.arange(q + 1) + 1) * self.lat[:q + 1, 0]).sum()) % M
        P = int(self.pk[:(q + 1) // 4, 0].sum())
        return row_logits(torch, h, S, P)

    def absorb(self, next_tokens, hidden) -> None:
        """The MTP head's rows mtp_len.. : each hidden row (position, token) must sit at its row's position."""

        for i, t in enumerate(next_tokens):
            q = self.mtp_len + i
            if int(hidden[i][0]) != -1:
                assert int(hidden[i][0]) == q, (int(hidden[i][0]), q)
            if q >= self.capacity:
                raise ValueError("MTP context past the cache capacity")
            self.mtpv[q, 0] = int(t) + 1


def fake_mtp_forward(w, st, b, next_tokens, hidden):
    st.absorb(next_tokens, hidden)


class FakeEngine:
    """decode.Engine's surface for the single-stream paths and ``multi``."""

    def __init__(self, torch, *, pool: int, streams: int, prefill_rows: int, mtp: bool = False) -> None:
        self.torch = torch
        self.w = SimpleNamespace(cfg=SimpleNamespace(eos=(EOS,), vocab=V, dense_limit=10**9, quant="mlx"),
                                 mtp=object() if mtp else None, comm=None, world=1, vocab_offset=0,
                                 head=SimpleNamespace(n=V), draft_head=None)
        self.rows, self.prefill_rows = 16, prefill_rows
        self.caches = FakeCaches(torch, pool)
        self.slots = FakeSlots(torch, streams, self.rows)
        self.home = self.st = FakeState(self.w, pool, self.rows, caches=self.caches, slots=self.slots)
        self.buf = SimpleNamespace(prefill=False, ids=[], taps=None, fnormed=None)
        self.pbuf = SimpleNamespace(prefill=True, ids=[], taps=None, fnormed=None, engine=self)
        self.mbuf = SimpleNamespace(rows=self.rows)
        self.last_hidden = None
        self.constraint = self.window = None
        self.forwards = self.mtp_steps = 0
        self.graphs = SimpleNamespace(main={(r, p): None for r in range(1, 9) for p in (0, 1)})

    def reset(self) -> None:
        self.st.reset()

    def use(self, st):
        prev, self.st = self.st, st
        return prev

    def graphed(self, st=None) -> bool:
        """The fake's "graphs" (its forward either way) run a state at the home slot and base, as the engine's."""
        st = self.st if st is None else st
        return st.slots is self.slots and st.caches is self.caches and st.graph_key == (0, 0)

    def tap_rows(self, n: int, b=None):
        return (b or self.buf).taps[:n]

    def main_hidden(self, rows: slice):
        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int):
        return self.torch.tensor([[-1, -1]], dtype=self.torch.int64)      # the head's own row: no position to check

    def mtp(self, next_tokens, hidden):
        """The MTP head: rows absorbed at mtp_len.., then the fake model's logits after what it absorbed (the
        stream's first token from its latent row 0), every third position's argmax made wrong."""

        torch = self.torch
        st = self.st
        st.absorb(next_tokens, hidden)
        n = st.mtp_len + len(next_tokens)
        seq = [int(st.lat[0, 0]) - 1] + [int(v) - 1 for v in st.mtpv[:n, 0].tolist()]
        logits = pure_logits(torch, seq)
        if n % 3 == 2:
            logits = torch.roll(logits, 1)
        self.mtp_steps += 1
        return logits[None]

    def forward(self, tokens):
        """A verify window: every row's latent written (as the kernels do), logits of each row from the state
        folded through the window's rows before it."""

        torch = self.torch
        st = self.st
        R = len(tokens)
        if st.pos + R > st.capacity:
            raise ValueError("context past the cache capacity")
        h = st.h()
        out = []
        for r, t in enumerate(tokens):
            q = st.pos + r
            st.write(q, int(t))
            h = fold(h, q, int(t))
            out.append(st.logits(q, h))
        st.slots.window[st.slot] = [int(t) for t in tokens]
        self.buf.taps = torch.tensor([[st.pos + r, int(t)] for r, t in enumerate(tokens)], dtype=torch.int64)
        self.buf.fnormed = self.buf.taps.clone()
        self.forwards += 1
        return torch.stack(out)

    def sample(self, logits, positions, sampling, *, draft=False, probs=None):
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def verify_window(self, tokens):
        return tokens

    def follow(self, tokens) -> None:
        pass


def fake_stage(w, st, b, chunk):
    b.ids = [int(t) for t in chunk]
    return len(chunk)


def fake_compute(w, st, b, R, *, nch=None, host_pos=None, cut=None, **kw):
    """A prompt chunk: its rows written and folded into the state now (as the KDA layers commit a prompt chunk)."""

    torch = b.engine.torch
    assert host_pos == st.pos and b.prefill and cut is None
    h = st.h()
    for i, t in enumerate(b.ids):
        q = st.pos + i
        st.write(q, t)
        h = fold(h, q, t)
    cur = st.cur[0]
    st.rec[1 - cur][0] = h
    st.cur = [1 - cur]
    st.conv.copy_(torch.tensor(([0, 0, 0] + list(st.conv.tolist()) + b.ids)[-3:]))
    b.taps = torch.tensor([[st.pos + i, t] for i, t in enumerate(b.ids)], dtype=torch.int64)
    b.fnormed = b.taps.clone()
    return st.logits(st.pos + R - 1, h)[None]


def fake_commit(w, st, b, R, keep):
    torch = __import__("torch")
    if b.prefill:
        assert keep == R
        st.set_pos(st.pos + R)
        return
    window = st.slots.window[st.slot]
    assert len(window) == R and 1 <= keep <= R
    h = st.h()
    for i, t in enumerate(window[:keep]):
        h = fold(h, st.pos + i, t)
    cur = st.cur[0]
    st.rec[1 - cur][0] = h
    st.cur = [1 - cur]
    st.conv.copy_(torch.tensor(([0, 0, 0] + list(st.conv.tolist()) + window[:keep])[-3:]))
    st.set_pos(st.pos + keep)


class FakeContext:
    """dflash2_multi.DraftContext's surface: a flat context of (token + 1) rows, checked to arrive in order."""

    def __init__(self, torch, owner, slot: int, capacity: int) -> None:
        self.owner, self.slot = owner, slot
        self.ring, self.window, self.block, self.capacity = 0, 6, 8, capacity
        self.kc = [torch.zeros((1, capacity + 8, 1), dtype=torch.int64)]
        self.vc = [torch.zeros((1, capacity + 8, 1), dtype=torch.int64)]
        self.pos_dev = torch.zeros((1,), dtype=torch.int64)
        self.context_end = 0

    def reset(self) -> None:
        self.context_end = 0
        self.pos_dev.zero_()

    def add_taps(self, taps) -> None:
        self.owner.commit([(self, taps)])


class FakeMultiDrafter:
    def __init__(self, torch, streams: int, capacity: int) -> None:
        self.torch = torch
        self.contexts = [FakeContext(torch, self, i, capacity) for i in range(streams)]
        self.proposed = 0

    def commit(self, items) -> None:
        slots = [c.slot for c, _ in items]
        assert len(set(slots)) == len(slots)
        for c, taps in items:
            got = [int(p) for p in taps[:, 0].tolist()]
            assert got == list(range(c.context_end, c.context_end + len(got))), (c.slot, c.context_end, got[:3])
            for p, t in taps.tolist():
                c.kc[0][0, p, 0] = t + 1
                c.vc[0][0, p, 0] = 7 * t + 1
            c.context_end += len(got)
            c.pos_dev.add_(len(got))

    block = 8

    @property
    def d(self):
        return self

    def candidates(self, reqs):
        """(ctx, pending, depth) -> "candidates": the fake chain as a [depth, 1] id array (values, proj unused)."""
        from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftRequest

        chains = self.propose([DraftRequest(c, p, n) for c, p, n in reqs])
        return [(np.asarray(ch, dtype=np.int64)[:, None], None, None) for ch in chains]

    def chain(self, tokens, values, proj, anchor, first, sampling, confidence=0.0, *, confs=None):
        out = [int(t) for t in tokens[:, 0]]
        if confs is not None:
            confs.extend(0.97 if k % 3 != 2 else 0.2 for k in range(len(out)))
        return out

    def propose(self, reqs) -> list[list[int]]:
        torch = self.torch
        out = []
        for r in reqs:
            c = r.ctx
            depth = min(r.depth, 7)
            if depth < 1 or c.context_end == 0:
                out.append([])
                continue
            seq = [int(v) - 1 for v in c.kc[0][0, :c.context_end, 0].tolist()] + [int(r.pending)]
            chain = []
            for k in range(depth):
                t = int(torch.argmax(pure_logits(torch, seq)))
                if k % 3 == 2:
                    t = (t + 1) % V
                chain.append(t)
                seq.append(t)
            self.proposed += 1
            out.append(chain)
        return out


@pytest.fixture
def fake(allocations, monkeypatch):  # noqa: F811
    import torch

    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    forward = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    monkeypatch.setattr(decode, "stage", fake_stage)
    monkeypatch.setattr(decode, "compute", fake_compute)
    monkeypatch.setattr(decode, "commit", fake_commit)
    monkeypatch.setattr(decode, "chunks_for", lambda st, R: 0)
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)     # serial_decode's, on a CPU-only box
    monkeypatch.setattr(decode, "mtp_forward", fake_mtp_forward)
    monkeypatch.setattr(forward, "commit", fake_commit)
    monkeypatch.setattr(forward, "State", FakeState)
    monkeypatch.delenv("TF_GLM_MULTI_LONE", raising=False)
    return torch


# -- shells ------------------------------------------------------------------------------------------------------------
def shell(torch, rank: int, *, pool: int, streams: int, prefill_rows: int = 64, entries: int = 8,
          limit: int | None = None, mtp: bool = False):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    g = GlmEngine.__new__(GlmEngine)
    g.rank = rank
    g.e = FakeEngine(torch, pool=pool, streams=streams, prefill_rows=prefill_rows, mtp=mtp)
    g.w = g.e.w
    g.drafter = None
    g.cache, g.live, g.cache_bytes, g.cache_entries = [], [], 1 << 40, entries
    g.eos, g.serial_only, g.policy = (EOS,), False, "0"
    g.limit = limit if limit is not None else pool - 16
    g.request = SimpleNamespace(policy=None, stop_eos=True)
    g.comm = SimpleNamespace()
    g.costs, g.model_dir = None, None
    g._ring = lambda: None
    return g


def solo(torch, prompt: list[int], count: int, sampling, *, stop_eos: bool = True) -> list[int]:
    """The request served alone with draft: false on the single-stream engine (the reference)."""

    g = shell(torch, 0, pool=16 * ALIGN, streams=1)
    g._share = lambda values: list(values)
    g.request.stop_eos = stop_eos
    out: list[int] = []
    g.generate(list(prompt), count, sampling, out.extend, draft=False)
    return out


class Pair:
    """Rank 0's decoder deciding, rank 1's applying each message rank 0 sends."""

    def __init__(self, torch, *, streams: int = 4, blocks: int = 8, drafter: bool = True, tune=None,
                 row_ms=None, **kw) -> None:
        from tensorfold.families.glm5_next.cuda.multi import MultiDecoder
        from tensorfold.families.glm5_next.cuda.verify import SerialVerify

        self.torch, self.sent, self.messages = torch, [], 0
        self.g = [shell(torch, r, pool=blocks * ALIGN, streams=streams, **kw) for r in range(2)]
        self.g[0]._share = lambda values: (self.sent.append(list(values)), list(values))[1]
        self.g[1]._share = lambda values: self.sent.pop(0)
        self.d = []
        for g in self.g:
            drafts = FakeMultiDrafter(torch, streams, blocks * ALIGN) if drafter else None
            g.drafter = object() if drafter else None
            d = MultiDecoder(g, streams, drafts=drafts, verify=SerialVerify(g.e, taps=drafter), tune=tune,
                             row_ms=row_ms)
            g.multi = d
            self.d.append(d)
        self.emitted: dict[int, list[int]] = {}

    def submit(self, prompt, count, sampling=None, *, policy: str = "f3", draft: bool = True, stop_eos: bool = True,
               cancel_after: int | None = None):
        from tensorfold.cuda.streams import Stream
        from tensorfold.families.glm5_next.cuda.engine import encode_policy

        s = Stream(list(prompt), count, sampling, draft=draft, stop_eos=stop_eos)
        s.glm = {"code": self.g[0]._effective(encode_policy(policy)), "spec": policy}
        got: list[int] = []

        def emit(new):
            got.extend(new)
            return cancel_after is not None and len(got) >= cancel_after

        s.emit = emit
        s.got = got
        self.d[0].admit(s)
        return s

    def follow(self) -> None:
        while self.sent:
            self.messages += 1
            self.d[1].follow(once=True)

    def step(self) -> list:
        d = self.d[0]
        done = d.round()
        d.finish(done)
        self.follow()
        if not d.outbox:                  # ops rank 0 applied but sends with its next message: compare after that
            self.check()
        return done

    def check(self) -> None:
        """Both ranks hold the same pool, kept prompts, streams, arenas, slots and DFlash2 contexts; every MTP
        stream's head holds its committed tokens."""

        torch = self.torch
        a, b = self.d
        key = lambda d: [(x.eid, x.base, x.size, x.owner, [k.kid for k in x.kept]) for x in d.pool.extents]  # noqa
        assert key(a) == key(b)
        assert [(c.kid, c.ids) for c in a.kept] == [(c.kid, c.ids) for c in b.kept]
        assert {s: (l.st.pos, l.slot, l.s.out, l.depth, l.st.mtp_len) for s, l in a.lanes.items()} == \
            {s: (l.st.pos, l.slot, l.s.out, l.depth, l.st.mtp_len) for s, l in b.lanes.items()}
        for p, q in zip(a.arena.planes, b.arena.planes):
            assert torch.equal(p.tensor, q.tensor)
        assert torch.equal(a.e.slots.rec, b.e.slots.rec) and torch.equal(a.e.slots.conv, b.e.slots.conv)
        if a.drafts is not None:
            for c, k in zip(a.drafts.contexts, b.drafts.contexts):
                assert c.context_end == k.context_end and torch.equal(c.kc[0], k.kc[0])
        for d in self.d:
            for lane in d.lanes.values():
                if lane.mtp and lane.decoding:
                    st = lane.st
                    n = min(st.mtp_len - st.mtp_drafted, st.pos - 1)
                    assert torch.equal(st.mtpv[:n, 0], st.lat[1:n + 1, 0]), lane.sid

    def run(self, until=None, most: int = 5000) -> None:
        for _ in range(most):
            if until is not None and until():
                return
            if not self.d[0].lanes:
                return
            self.step()
        raise AssertionError("the streams did not finish")


def _prompt(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(0, 40, size=n)]


def _sampling(kind: str, seed: int):
    from tensorfold.engine.exact_sampling import Sampling

    return {"greedy": None, "topk": Sampling(seed, 1.0, 20, 0.95), "nucleus": Sampling(seed, 1.0, 0, 0.9),
            "minp": Sampling(seed, 0.8, 12, 1.0, 0.05)}[kind]


# -- settings ---------------------------------------------------------------------------------------------------------
def test_the_settings(allocations, monkeypatch):  # noqa: F811
    from tensorfold.families.glm5_next.cuda.multi import (MTP_GREEDY, MTP_SAMPLED, fill_rows, fill_share, keep_point,
                                                          multi_code, trim, verify_kind)

    assert fill_rows(2048, "") == 1024 and fill_rows(512, "") == 512 and fill_rows(2048, "256") == 256
    for bad in ("0", "100", "4096", "x"):
        with pytest.raises(ValueError, match="TF_GLM_FILL_ROWS"):
            fill_rows(2048, bad)
    assert verify_kind("") == "batched" and verify_kind("serial") == "serial"
    with pytest.raises(ValueError, match="TF_GLM_MULTI_VERIFY"):
        verify_kind("tree")
    assert fill_share("") == 0.5 and fill_share("1") == 1.0
    for bad in ("0", "1.5", "-1", "x"):
        with pytest.raises(ValueError, match="TF_GLM_FILL_SHARE"):
            fill_share(bad)
    df = [13, 5, 300000, 0]                                            # engine.DFLASH_POLICY, fc5:0.3
    assert multi_code([0, 0, 0, 0], True, True, df) == [0, 0, 0, 0]
    assert multi_code([4, 2, 8, 30000], True, True, df) == df          # auto: DFlash2 with the draft model
    assert multi_code([4, 2, 8, 30000], False, True, df) == MTP_GREEDY  # ... else the MTP head's auto rule
    assert multi_code([5, 2, 8, 30000], False, True, df, greedy=False) == MTP_SAMPLED
    assert multi_code([1, 3, 0, 0], True, False, df) == [11, 3, 0, 0]   # MTP 3 without the head: DFlash2 3
    assert multi_code([1, 3, 0, 0], True, True, df) == [1, 3, 0, 0]     # ... with it: MTP 3
    assert multi_code([3, 3, 350000, 0], False, True, df) == [3, 3, 350000, 0]
    assert multi_code([16, 7, 300000, 2000000], True, True, df) == [16, 7, 300000, 2000000]
    assert multi_code([12, 3, 0, 0], False, True, df) == [2, 3, 0, 0]   # DFlash2 without the draft model: MTP
    assert multi_code([11, 3, 0, 0], False, False, df) == [0, 0, 0, 0]  # no drafter at all: serial
    assert multi_code([4, 2, 8, 30000], False, False, df) == [0, 0, 0, 0]
    assert keep_point(100, 0) == 99 and keep_point(100, 99) is None and keep_point(1, 0) == 1
    assert keep_point(2, 0) == 1 and keep_point(2, 1) is None
    assert trim([[1] * 15, [2] * 15, [3] * 3, []], 32) == [[1] * 13, [2] * 12, [3] * 3, []]
    assert sum(1 + len(d) for d in trim([[1] * 15] * 4, 32)) == 32
    assert trim([[1, 2], [3]], 32) == [[1, 2], [3]]


# -- concurrent requests equal their solo serial replies --------------------------------------------------------------
CASES = [(_prompt(37, 1), 60, "topk", "f3", True), (_prompt(150, 2), 50, "greedy", "f5", True),
         (_prompt(90, 3), 45, "nucleus", "0", False), (_prompt(210, 4), 55, "minp", "2", True),
         (_prompt(64, 5), 40, "greedy", "auto", True), (_prompt(129, 6), 35, "topk", "c3:0.35", True)]


@pytest.mark.parametrize("drafters", ["dflash2", "both", "mtp"])
@pytest.mark.parametrize("streams", [2, 4])
def test_staggered_concurrent_requests_equal_their_solo_serial_replies(fake, streams, drafters):
    """Six requests admitted as they come; with the MTP head loaded its policies ("2", "c3:0.35") draft on it, beside
    DFlash2's (auto, f3, f5); with the head alone every drafting request does."""

    torch = fake
    p = Pair(torch, streams=streams, drafter=drafters != "mtp", mtp=drafters != "dflash2")
    admitted, pending = [], list(enumerate(CASES))
    starts = {0: 0, 1: 2, 2: 3, 3: 7, 4: 11, 5: 12}
    for it in range(4000):
        while pending and it >= starts[pending[0][0]] and p.d[0].live() < streams:
            i, (prompt, count, kind, policy, draft) = pending.pop(0)
            s = p.submit(prompt, count, _sampling(kind, 100 + i), policy=policy, draft=draft)
            admitted.append((i, s))
        if not p.d[0].lanes and not pending:
            break
        if p.d[0].lanes:
            for s in p.step():
                s.finished_at = it
        elif p.sent:
            p.follow()
    assert not pending and not p.d[0].lanes
    if drafters != "mtp":
        assert p.d[0].drafts.proposed > 0
    if drafters != "dflash2":
        assert p.g[0].e.mtp_steps > 0 and p.g[0].e.mtp_steps == p.g[1].e.mtp_steps
    else:
        assert p.g[0].e.mtp_steps == 0
    for i, s in admitted:
        prompt, count, kind, _, _ = CASES[i]
        want = solo(torch, prompt, count, _sampling(kind, 100 + i))
        assert s.got == want, (i, len(s.got), len(want))
        assert s.out[:len(want)] == want
    # drafted rounds took several rows at once, and some kept more than one
    assert any(s.drafted for _, s in admitted) and any(s.accepted for _, s in admitted)
    assert p.d[1].idle and not p.d[1].lanes


def test_one_message_an_iteration(fake):
    """Every iteration of rank 0 sends one message (and one more, IDLE, when it goes idle), so rank 1 never waits
    for a second collective of the same iteration; rank 1 idles after the last stream."""

    torch = fake
    p = Pair(torch, streams=2)
    p.submit(_prompt(80, 7), 30, None)
    p.submit(_prompt(20, 8), 25, _sampling("topk", 3))
    from tensorfold.families.glm5_next.cuda.multi import FILL, IDLE, MultiDecoder, ROUND

    kinds = []
    while p.d[0].lanes:
        before = len(p.sent)
        done = p.d[0].round()
        assert len(p.sent) - before == 1                     # the round's own message
        ops = [op for op, _ in MultiDecoder.parse(p.sent[-1])]
        assert ops[-1] in (FILL, ROUND)
        kinds.append(ops[-1])
        p.d[0].finish(done)
        extra = p.sent[before + 1:]
        assert all([op for op, _ in MultiDecoder.parse(m)][-1] == IDLE for m in extra)
        p.follow()
        if not p.d[0].outbox:
            p.check()
    assert FILL in kinds and ROUND in kinds
    assert p.d[0].idle and p.d[1].idle


def test_cancelled_streams_end_at_the_next_round_on_both_ranks(fake):
    """A request whose client leaves (its emit returns True) ends at that round boundary: FINISH(cancelled) reaches
    rank 1 in the next message, both ranks free its slot and extent (its prompt stays kept), and the other streams'
    replies are unchanged."""

    from tensorfold.families.glm5_next.cuda.multi import CANCELLED, FINISH, MultiDecoder

    torch = fake
    p = Pair(torch, streams=3)
    a = p.submit(_prompt(60, 9), 200, _sampling("topk", 1), stop_eos=False, cancel_after=12)
    b = p.submit(_prompt(70, 10), 40, None)
    c = p.submit(_prompt(50, 11), 40, _sampling("nucleus", 2))
    log = []
    real = p.g[0]._share
    p.g[0]._share = lambda values: (log.append(list(values)), real(values))[1]
    rounds_after = seen = None
    while p.d[0].lanes:
        done = p.step()
        if a in done:
            rounds_after, seen = a.rounds, len(log)
            assert a.sid not in p.d[0].lanes and a.sid in p.d[1].lanes      # rank 1 hears with the next message
        elif seen is not None and len(log) > seen:
            first = MultiDecoder.parse(log[seen])[0]
            assert first == (FINISH, [a.sid, CANCELLED])       # before anything else of that iteration
            assert a.sid not in p.d[1].lanes
        if rounds_after is not None:
            assert a.rounds == rounds_after                   # no round after the one it left in
    assert 12 <= len(a.got) < 12 + 16 and a.done
    assert b.got == solo(torch, b.prompt, 40, None)
    assert c.got == solo(torch, c.prompt, 40, _sampling("nucleus", 2))
    assert a.got == solo(torch, a.prompt, 200, _sampling("topk", 1), stop_eos=False)[:len(a.got)]
    assert any(k.ids == a.prompt[:-1] for k in p.d[0].kept)  # its prompt stays for a later turn
    assert all(x.owner is None or x.owner in p.d[0].lanes for x in p.d[0].pool.extents)


def test_a_cancel_message_names_the_stream(fake):
    from tensorfold.families.glm5_next.cuda.multi import CANCELLED, DONE, FINISH, MultiDecoder

    torch = fake
    p = Pair(torch, streams=2)
    a = p.submit(_prompt(30, 12), 100, None, stop_eos=False, cancel_after=5)
    b = p.submit(_prompt(30, 13), 8, None)
    log = []
    real = p.g[0]._share
    p.g[0]._share = lambda values: (log.append(list(values)), real(values))[1]
    p.run()
    finishes = [pl for m in log for op, pl in MultiDecoder.parse(m) if op == FINISH]
    assert [a.sid, CANCELLED] in finishes and [b.sid, DONE] in finishes


# -- room: pause, give back, evict, move ----------------------------------------------------------------------------
def test_a_stream_without_room_pauses_and_the_youngest_goes_back_when_all_wait(fake):
    """Two streams whose extents fill the pool both need a second block: they pause, the youngest leaves (its kept
    prompt dropped) and replays later from its prompt; its sent tokens are owed, not sent twice; both replies equal
    their solo ones."""

    torch = fake
    p = Pair(torch, streams=2, blocks=2)
    one = p.submit(_prompt(2000, 14), 80, None, policy="f3", stop_eos=False)
    two = p.submit(_prompt(2010, 15), 60, _sampling("topk", 4), policy="f3", stop_eos=False)
    requeued, gave_back, paused = [], 0, 0
    for _ in range(5000):
        if not p.d[0].lanes and not requeued:
            break
        if not p.d[0].lanes and requeued:
            again = requeued.pop(0)
            p.d[0].admit(again)
        p.step()
        paused = max(paused, p.d[0].health()["streams"]["paused"])
        for s in p.d[0].requeue:
            gave_back += 1
            again = s.continued()
            again.glm, again.emit, again.got = s.glm, s.emit, s.got
            requeued.append(again)
        p.d[0].requeue = []
    assert not p.d[0].lanes and not requeued
    assert gave_back == 1 and paused >= 1          # both waited (the survivor still paused as the youngest left)
    assert one.got == solo(torch, one.prompt, 80, None, stop_eos=False)
    assert two.got == solo(torch, two.prompt, 60, _sampling("topk", 4), stop_eos=False)   # each token sent once


def test_growth_moves_extents_and_keeps_every_row(fake):
    """A pool fragmented by kept prompts: growing streams move (and the pool compacts), evicting kept prompts least
    recently used first; every reply is its solo one, and the moves copied the right rows (the fake reads them)."""

    from tensorfold.families.glm5_next.cuda.multi import EVICT, MOVE, MultiDecoder

    torch = fake
    p = Pair(torch, streams=3, blocks=6, entries=8)
    log = []
    real = p.g[0]._share
    p.g[0]._share = lambda values: (log.append(list(values)), real(values))[1]
    # three short conversations leave kept prompts scattered over the pool
    for seed in (20, 21, 22, 23):
        p.submit(_prompt(100 + seed, seed), 10, None)
        p.run()
    kept_before = [k.ids for k in p.d[0].kept]
    assert len(kept_before) == 4
    # three long replies must grow past their first blocks
    reqs = [(_prompt(1900, 30), 400, None), (_prompt(1500, 31), 700, _sampling("topk", 5)),
            (_prompt(1000, 32), 1200, _sampling("minp", 6))]
    streams = [p.submit(pr, n, sp, stop_eos=False) for pr, n, sp in reqs]
    p.run()
    ops = [op for m in log for op, _ in MultiDecoder.parse(m)]
    assert EVICT in ops and MOVE in ops
    for s, (pr, n, sp) in zip(streams, reqs):
        assert s.got == solo(torch, pr, n, sp, stop_eos=False)


# -- kept prompts -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy,mtp", [("f3", False), ("2", True)], ids=["dflash2", "mtp"])
def test_resumes_take_kept_extents_over_or_copy_them(fake, policy, mtp):
    """A conversation's next turn resumes from its kept prompt (kept before the prompt's last token, as the
    single-stream engine keeps it): in place when no stream holds the extent, copied into a new extent while another
    stream still writes in it; both equal the fresh (solo, serial) reply, with DFlash2's window or the MTP head's rows
    resumed along."""

    torch = fake
    p = Pair(torch, streams=3, blocks=8, mtp=mtp)
    base = _prompt(300, 40)
    first = p.submit(base, 20, None, policy=policy)
    p.run()
    assert [k.ids for k in p.d[0].kept] == [base[:299]]
    assert all((k.pending is not None) == mtp and (k.drafter_rows is not None) != mtp for k in p.d[0].kept)
    turn = base + first.got + _prompt(30, 41)
    again = p.submit(turn, 25, _sampling("topk", 7), policy=policy)
    assert again.cached == 299                                   # the prompt's kept state, taken over in place
    held = p.d[0].lanes[again.sid].extent
    other = p.submit(base + _prompt(12, 42), 30, None, policy=policy)   # the same prefix while `again` writes there
    assert other.cached == 299 and p.d[0].lanes[other.sid].extent is not held
    p.run()
    assert again.got == solo(torch, turn, 25, _sampling("topk", 7))
    assert other.got == solo(torch, base + _prompt(12, 42), 30, None)
    assert any(s.accepted for s in (again, other))


def test_the_same_prompt_again_resumes_before_its_last_token(fake):
    """The same prompt sent again (a retry, or a benchmark's repeat) resumes at its kept point and prefills one row;
    its reply equals the first one and the solo one."""

    from tensorfold.families.glm5_next.cuda import decode

    torch = fake
    p = Pair(torch, streams=2, blocks=4)
    prompt = _prompt(150, 80)
    sampling = _sampling("topk", 81)
    cold = p.submit(prompt, 12, sampling, stop_eos=False)
    p.run()
    want = solo(torch, prompt, 12, sampling, stop_eos=False)
    assert cold.cached == 0 and cold.got == want
    chunks = []
    real = decode.prefill_chunk
    decode.prefill_chunk = lambda e, pr, start, R, **kw: (chunks.append((start, R)), real(e, pr, start, R, **kw))[1]
    try:
        warm = p.submit(prompt, 12, sampling, stop_eos=False)
        assert warm.cached == 149
        p.run()
    finally:
        decode.prefill_chunk = real
    assert chunks == [(149, 1), (149, 1)] and warm.got == want          # one row on each rank
    assert [len(k.ids) for k in p.d[0].kept] == [149]


def test_mtp_drafts_need_kept_prompts_with_the_heads_rows(fake):
    """A DFlash2 stream's kept prompt has no MTP rows (and an MTP stream's no DFlash2 window): a request drafting with
    the other drafter does not resume from it; a serial one resumes from either; an MTP stream's next turn resumes
    from its own kept prompt, the head's rows along."""

    torch = fake
    p = Pair(torch, streams=2, blocks=8, mtp=True)
    base = _prompt(200, 70)
    p.submit(base, 10, None, policy="f3")
    p.run()
    first = base + _prompt(20, 71)
    s = p.submit(first, 15, None, policy="2")
    assert s.cached == 0
    p.run()
    serial = base + _prompt(20, 72)
    r = p.submit(serial, 15, None, policy="0")
    assert r.cached == 199
    p.run()
    turn = first + s.got + _prompt(10, 73)
    t = p.submit(turn, 15, _sampling("topk", 3), policy="2")
    assert t.cached == len(first) - 1
    p.run()
    assert s.got == solo(torch, first, 15, None)
    assert r.got == solo(torch, serial, 15, None)
    assert t.got == solo(torch, turn, 15, _sampling("topk", 3))
    assert t.accepted


def test_references_keep_nothing_and_never_resume(fake):
    torch = fake
    p = Pair(torch, streams=2)
    prompt = _prompt(120, 60)
    s = p.submit(prompt, 15, None, draft=False)
    p.run()
    assert not p.d[0].kept and s.cached == 0
    p.submit(prompt, 15, None)
    p.run()
    t = p.submit(prompt + _prompt(10, 61), 15, None, draft=False)
    assert t.cached == 0
    p.run()
    assert t.got == solo(torch, prompt + _prompt(10, 61), 15, None)


# -- the threaded scheduler -------------------------------------------------------------------------------------------
def test_the_scheduler_with_a_follower_thread(fake):
    """GlmScheduler's worker on rank 0 and rank 1's ``follow`` in another thread, requests from four threads, one of
    which stops after a few tokens (a stop string): every other reply equals its solo one."""

    from tensorfold.families.glm5_next.cuda.engine import encode_policy
    from tensorfold.families.glm5_next.cuda.multi import GlmScheduler, MultiDecoder

    torch = fake
    wire: queue.Queue = queue.Queue()
    g0 = shell(torch, 0, pool=8 * ALIGN, streams=3)
    g1 = shell(torch, 1, pool=8 * ALIGN, streams=3)
    g0._share = lambda values: (wire.put(list(values)), list(values))[1]
    g1._share = lambda values: wire.get(timeout=60)
    bell = threading.Semaphore(0)                     # the doorbell: rank 1 idles on it, not on the wire
    g0._ring, g1._await_bell = bell.release, bell.acquire
    from tensorfold.families.glm5_next.cuda.verify import SerialVerify

    d0 = MultiDecoder(g0, 3, drafts=FakeMultiDrafter(torch, 3, 8 * ALIGN), verify=SerialVerify(g0.e, taps=True))
    d1 = MultiDecoder(g1, 3, drafts=FakeMultiDrafter(torch, 3, 8 * ALIGN), verify=SerialVerify(g1.e, taps=True))
    errors = []

    def follower():
        try:
            d1.follow()
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threading.Thread(target=follower, daemon=True).start()
    sched = GlmScheduler(d0, max_streams=3)
    reqs = [(_prompt(40 + 30 * i, 70 + i), 30 + 5 * i, _sampling(("topk", "greedy", "nucleus", "minp")[i], i))
            for i in range(4)]
    results: dict[int, list[int]] = {}

    def client(i):
        prompt, n, sp = reqs[i]
        got: list[int] = []

        def emit(new):
            got.extend(new)
            return i == 3 and len(got) >= 6

        sched.submit(prompt, n, sp, True, emit, glm={"code": encode_policy("f3"), "spec": "f3"})
        results[i] = got

    threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors and len(results) == 4
    for i in range(3):
        prompt, n, sp = reqs[i]
        assert results[i] == solo(torch, prompt, n, sp), i
    assert 6 <= len(results[3]) < reqs[3][1]
    assert results[3] == solo(torch, reqs[3][0], reqs[3][1], reqs[3][2])[:len(results[3])]
    for _ in range(200):                              # the follower applies the last message (IDLE)
        if d1.idle and not d1.lanes:
            break
        threading.Event().wait(0.01)
    assert d1.idle and not d1.lanes and d0.idle


def test_health_reports_the_streams_and_the_pool(fake):
    """/health (cuda.health) merges the decoder's own counts: paused streams, the pool's size and free tokens and the
    kept prompts, which work/lab/multi's tools read (pool_tokens, streams.decoding / prefilling / paused)."""

    from tensorfold.cuda import health

    torch = fake
    p = Pair(torch, streams=2, blocks=4)
    p.submit(_prompt(100, 90), 10, None)
    app = SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(decoder=p.d[0], max_streams=2)),
                          effective_context_window=4 * ALIGN - 16)
    h = health.Health().snapshot(app)
    assert h["streams"] == {"decoding": 0, "prefilling": 1, "max": 2, "filling": 1, "paused": 0}
    assert h["pool_tokens"] == 4 * ALIGN and h["pool_free_tokens"] == 3 * ALIGN and h["kept_prompts"] == 0
    p.run()
    h = health.Health().snapshot(app)
    assert h["streams"]["decoding"] == h["streams"]["filling"] == 0 and h["kept_prompts"] == 1


# -- speed settings (multi_tune): the same replies ------------------------------------------------------------------
def test_a_lone_stream_moves_home_and_a_second_one_joins(fake):
    """Two streams; the one in slot 0 at the pool's first rows ends, the other (slot 1, elsewhere in the pool) goes on
    alone: it moves into the one-stream graphs' home (SLOT, MOVE) and its rounds say so (ROUND's lone flag); a third
    stream joins it on the batched path. Every reply is its solo one and both ranks agree after every message."""

    from tensorfold.families.glm5_next.cuda.multi import MOVE, ROUND, SLOT, MultiDecoder

    from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings

    torch = fake
    p = Pair(torch, streams=2, blocks=6, tune=MultiSettings.from_env({"TF_GLM_MULTI_LONE": "1"}))
    log = []
    real = p.g[0]._share
    p.g[0]._share = lambda values: (log.append(list(values)), real(values))[1]
    a = p.submit(_prompt(2100, 100), 8, None, stop_eos=False)             # two blocks from 0: b sits past them
    b = p.submit(_prompt(300, 101), 120, _sampling("topk", 5), stop_eos=False)
    assert (p.d[0].lanes[a.sid].slot, p.d[0].lanes[b.sid].slot) == (0, 1)
    c = None
    while p.d[0].lanes:
        p.step()
        if b.sid in p.d[0].lanes and a.done and c is None and len(b.out) > 60:
            c = p.submit(_prompt(150, 102), 40, None, stop_eos=False)
    ops = [(op, pl) for m in log for op, pl in MultiDecoder.parse(m)]
    assert (SLOT, [b.sid, 0]) in ops and any(op == MOVE and pl[1] == 0 for op, pl in ops)
    flags = [pl[-1] for op, pl in ops if op == ROUND]
    assert 1 in flags                                                    # b's rounds alone, at home
    lone_after = [pl for op, pl in ops if op == ROUND and pl[0] == 2]
    assert lone_after and all(pl[-1] == 0 for pl in lone_after)          # two streams: never lone
    assert a.got == solo(torch, a.prompt, 8, None, stop_eos=False)
    assert b.got == solo(torch, b.prompt, 120, _sampling("topk", 5), stop_eos=False)
    assert c is not None and c.got == solo(torch, c.prompt, 40, None, stop_eos=False)


@pytest.mark.parametrize("env", [{"TF_GLM_MULTI_SAMPLER": "packed"}, {"TF_GLM_MULTI_DEPTH": "joint"},
                                 {"TF_GLM_MULTI_DEPTH": "scale:0.5"}, {"TF_GLM_MULTI_LONE": "1"},
                                 {"TF_GLM_MULTI_SAMPLER": "packed", "TF_GLM_MULTI_DEPTH": "joint",
                                  "TF_GLM_MULTI_PROFILE": "5"}],
                         ids=["packed", "joint", "scale", "lone", "all"])
def test_speed_settings_keep_every_reply(fake, env, capsys):
    from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings

    torch = fake
    tune = MultiSettings.from_env(env)
    row_ms = [10.0 + 0.8 * r for r in range(32)]
    p = Pair(torch, streams=3, tune=tune, row_ms=row_ms)
    cases = [(_prompt(60, 110), 50, "nucleus"), (_prompt(90, 111), 45, "topk"), (_prompt(40, 112), 40, "greedy"),
             (_prompt(70, 113), 35, "nucleus")]
    streams = [p.submit(pr, n, _sampling(kind, 7 + i)) for i, (pr, n, kind) in enumerate(cases[:3])]
    for _ in range(6):
        p.step()
    streams.append(p.submit(cases[3][0], cases[3][1], _sampling(cases[3][2], 10)) if p.d[0].live() < 3 else None)
    p.run()
    if streams[3] is None:
        streams[3] = p.submit(cases[3][0], cases[3][1], _sampling(cases[3][2], 10))
        p.run()
    for i, (s, (pr, n, kind)) in enumerate(zip(streams, cases)):
        assert s.got == solo(torch, pr, n, _sampling(kind, 7 + i)), (env, i)
    if "TF_GLM_MULTI_PROFILE" in env:
        assert "multi profile" in capsys.readouterr().out


def test_joint_allocation():
    from tensorfold.families.glm5_next.cuda.multi_tune import allocate, reach_of

    cost = lambda rows: 40.0 + 1.0 * rows                         # noqa: E731
    reach = [reach_of([0.9, 0.9, 0.9, 0.9]), reach_of([0.5, 0.2]), reach_of([])]
    take = allocate(reach, 3, cost, 10.0, 32)
    assert take[0] == 4 and take[1] >= 1 and take[2] == 0
    assert allocate(reach, 3, cost, 10.0, 5) == [2, 0, 0]          # the cap: the best reaches first
    expensive = lambda rows: 10.0 + 100.0 * rows                  # noqa: E731
    assert allocate(reach, 3, expensive, 0.0, 32) == [0, 0, 0]
    assert allocate([], 0, cost, 0.0, 32) == []


def _parts(torch, specs, width, seed):
    from tensorfold.engine.exact_sampling import Sampling

    g = torch.Generator().manual_seed(seed)
    parts = []
    for rows, kind, flat in specs:
        logits = (torch.randn((rows, width), generator=g) * (0.01 if flat else 3.0)).to(torch.bfloat16)
        sampling = {"greedy": None, "topk": Sampling(seed + rows, 1.0, 20, 0.95),
                    "nucleus": Sampling(seed + rows, 1.0, 0, 0.95), "minp": Sampling(seed + rows, 0.7, 0, 1.0, 0.05)}[kind]
        parts.append((logits, [100 + r for r in range(rows)], sampling))
    return parts


class _Twice:
    """Two ranks of the same shard: every all-gather hands back this rank's words twice."""

    rank, world = 0, 2

    def all_gather(self, send, recv) -> None:
        n = send.numel()
        recv.view(-1)[:n].copy_(send.reshape(-1))
        recv.view(-1)[n:2 * n].copy_(send.reshape(-1))


@pytest.mark.parametrize("two", [False, True], ids=["one-rank", "two-ranks"])
def test_the_packed_sampler_draws_what_each_part_draws_alone(fake, two):
    """Mixed parts (greedy, top-k, nucleus with top_p, min_p alone, one flat enough that its nucleus runs past the
    1,024 candidates and reads whole shards) sampled in two gathers equal each part through decode.sample_rows."""

    from tensorfold.families.glm5_next.cuda.decode import sample_rows
    from tensorfold.families.glm5_next.cuda.multi_tune import sample_packed

    torch = fake
    w = SimpleNamespace(comm=_Twice() if two else None, world=2 if two else 1, vocab_offset=0)
    for seed in (1, 2, 3):
        specs = [(3, "nucleus", False), (2, "topk", False), (1, "greedy", False), (4, "minp", False),
                 (2, "nucleus", True), (5, "nucleus", False)]
        parts = _parts(torch, specs, 5000, seed)
        got = sample_packed(w, parts)
        want = [sample_rows(w, logits, positions, sampling) for logits, positions, sampling in parts]
        assert got == want, seed
