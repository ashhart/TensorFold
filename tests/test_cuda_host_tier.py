"""The CUDA engine's host-RAM tier (``--ram-tier-gib``), checked on the host: CPU tensors, pageable chunks, no GPU.

- ``HostTier``: a spilled state and drafter snapshot come back bit for bit, in tensors of their own (attention buffers
  at their old row count, the rows below ``pos`` copied) or in given buffers; the budget holds whole chunks and the
  least recently spilled entries leave first; a lookup takes the longest entry the prompt strictly extends, as
  ``PrefixCache`` does; a host that refuses to pin gets one warning and pageable chunks.
- Attention rows are shared by prefix: a spill copies only rows no cached segment holds for its ids (none for a
  prefix, the new ones for an extension, all past the first differing id), chains of segments come back exact, idle
  segments make room before entries do, and a long random run stays in budget with every take exact.
- ``PrefixCache.on_evict`` receives every entry the cache lets go: evicted past ``keep``, or dropped because a
  shorter entry resumes into buffers they share.
- The one-stream engine and the concurrent decoder resume a returning conversation from the tier (stand-in prefills).
- ``--ram-tier-gib`` reaches Qwen3.8-27B's CUDA engine on one rank and is refused elsewhere before any download.
"""

from __future__ import annotations

import random
import sys
import types
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.cuda.streams import PrefixCache
from tests.test_qwen27_prompt_end_cache_host import CONTEXT, _ids_of, _stand_in_prefills
from tests.test_qwen27_prompt_end_cache_host import cuda_modules  # noqa: F401  (fixture: the 27B's CUDA modules)

NL, NL2 = 198, 271          # Qwen's "\n" and "\n\n"
THINK, END_THINK = 300, 301  # stand-ins for <think> and </think>
PKG = "tensorfold.families.qwen3_5.cuda"


class FakeState:
    """A committed state's layout: per layer DeltaNet rows and state (``conv``, ``rec``) or an attention cache."""


def _random(torch, gen, shape, dtype):
    """Random bytes as ``dtype``: NaN payloads, subnormals and infinities included."""

    size = torch.empty((), dtype=dtype).element_size()
    count = 1
    for n in shape:
        count *= n
    return torch.randint(0, 256, (count * size,), dtype=torch.uint8, generator=gen).view(dtype).reshape(shape)


def _bits(t):
    import torch

    return t.contiguous().reshape(-1).view(torch.uint8)


def _same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool((_bits(a) == _bits(b)).all())


def _layers(torch, gen, pos: int, rows: int) -> FakeState:
    st = FakeState()
    st.pos, st.limit, st.rope_delta = pos, 4096, 3
    st.conv, st.rec, st.kv = [], [], []
    for linear in (True, True, True, False, True, False):
        if linear:
            st.conv.append(_random(torch, gen, (3, 40), torch.bfloat16))
            st.rec.append(_random(torch, gen, (2, 4, 4), torch.float32))
            st.kv.append(None)
        else:
            st.conv.append(None)
            st.rec.append(None)
            st.kv.append(tuple(_random(torch, gen, (rows, 1, 8), torch.bfloat16) for _ in range(2)))
    return st


def _small(torch, ids, nbytes: int = 300) -> FakeState:
    """A state holding ``nbytes`` of DeltaNet bytes and no attention cache."""

    st = FakeState()
    st.pos, st.limit, st.conv, st.rec, st.kv = len(ids), 0, [torch.zeros(nbytes, dtype=torch.uint8)], [None], [None]
    return st


def _tier(budget: int, chunk: int, **kwargs):
    from tensorfold.cuda.host_tier import HostTier

    return HostTier(budget, "cpu", chunk=chunk, **kwargs)


@pytest.mark.torch
@pytest.mark.parametrize("chunk", [64, 1000, 1 << 20])
def test_a_spilled_state_and_snapshot_come_back_bit_for_bit(chunk):
    """Chunks smaller than a tensor, not a multiple of the alignment, and larger than the entry: after the GPU side is
    overwritten, the entry comes back as it left, attention rows below ``pos`` in buffers of the old row count."""

    import torch

    gen = torch.Generator().manual_seed(1)
    st = _layers(torch, gen, pos=7, rows=10)
    snap = ([_random(torch, gen, (2, 5, 8), torch.bfloat16), None], [_random(torch, gen, (2, 5, 8), torch.bfloat16),
                                                                    None], 5, 9)
    before = [None if t is None else t.clone() for t in st.conv + st.rec + snap[0] + snap[1]]
    keys = [None if kv is None else (kv[0][:7].clone(), kv[1][:7].clone()) for kv in st.kv]
    tier = _tier(1 << 22, chunk)
    assert tier.put(list(range(7)), st, snap)
    for t in [t for t in st.conv + st.rec + snap[0] + snap[1] if t is not None] + \
            [t for kv in st.kv if kv is not None for t in kv]:
        t.view(torch.uint8).fill_(0x5A)                   # the device side reused: the host copy stands alone
    ids, back, back_snap = tier.take(list(range(8)))
    assert ids == list(range(7)) and type(back) is FakeState and not tier.entries
    (seg,) = tier.segments                                  # the rows stay for a later spill to chain onto
    assert seg.refs == 0 and tier.used == len(seg.chunks) * chunk
    assert (back.pos, back.limit, back.rope_delta) == (7, 4096, 3)
    after = back.conv + back.rec + back_snap[0] + back_snap[1]
    assert [t is None for t in after] == [t is None for t in before]
    assert all(_same(a, b) for a, b in zip(after, before) if a is not None)
    assert all(a is not b for a, b in zip(after, st.conv + st.rec + snap[0] + snap[1]) if a is not None)
    for kv, want in zip(back.kv, keys):
        assert (kv is None) == (want is None)
        if kv is not None:
            assert isinstance(kv, tuple) and kv[0].shape == kv[1].shape == (10, 1, 8)
            assert _same(kv[0][:7], want[0]) and _same(kv[1][:7], want[1])
    assert isinstance(back_snap, tuple) and isinstance(back_snap[0], list) and back_snap[2:] == (5, 9)


@pytest.mark.torch
def test_the_budget_holds_whole_chunks_and_the_least_recently_spilled_leave_first():
    import torch

    tier = _tier(5 * 256, 256)                            # five chunks; each entry's 300 bytes take two
    assert tier.capacity == 5
    for ids in ([1], [2]):
        assert tier.put(ids, _small(torch, ids), None)
    assert tier.put([3], _small(torch, [3]), None)
    assert [e.ids for e in tier.entries] == [[2], [3]] and tier.held <= 5
    assert tier.put([2], _small(torch, [2]), None)       # the same ids again: one entry, now the newest
    assert [e.ids for e in tier.entries] == [[3], [2]]
    assert not tier.put([9], _small(torch, [9], nbytes=6 * 256), None)    # past the budget: not kept, none leave
    assert [e.ids for e in tier.entries] == [[3], [2]]
    held = tier.held
    assert tier.take([3, 4]) is not None
    assert tier.put([4], _small(torch, [4]), None) and tier.held == held      # a taken entry's chunks are reused
    assert [e.ids for e in tier.entries] == [[2], [4]] and tier.used == 4 * 256
    assert _tier(255, 256).capacity == 0 and not _tier(255, 256).put([1], _small(torch, [1]), None)


@pytest.mark.torch
@pytest.mark.parametrize("seed", range(4))
def test_a_lookup_takes_the_longest_strict_prefix_past_the_gpu_match(seed):
    """The rule of ``PrefixCache.longest``: an entry the prompt strictly extends (a whole-prompt entry would leave
    nothing to prefill), the longest one, and only past ``than`` (the GPU cache's own match)."""

    import torch

    rng = random.Random(seed)
    entries = [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 9], [7], [1, 2, 3, 4, 5, 6, 7]]
    tier, cache = _tier(1 << 20, 1024), PrefixCache(keep=len(entries))
    for ids in entries:
        tier.put(ids, _small(torch, ids, nbytes=8), None)
        cache.add(ids, None, None)
    for _ in range(200):
        prompt = [rng.choice([1, 2, 3, 4, 5, 6, 7, 9]) for _ in range(rng.randrange(0, 9))]
        if rng.random() < 0.5:
            prompt = rng.choice(entries)[:rng.randrange(1, 8)] + prompt[:rng.randrange(0, 3)]
        mine, gpu = tier.longest(prompt), cache.longest(prompt)
        assert (mine and mine.ids) == (gpu and gpu[0]), prompt
    assert tier.longest([1, 2, 3, 4, 5]).ids == [1, 2, 3]
    assert tier.longest([1, 2, 3, 4, 5, 6], than=3).ids == [1, 2, 3, 4, 5]
    assert tier.longest([1, 2, 3, 4, 5, 6], than=5) is None and tier.longest([7]) is None


@pytest.mark.torch
def test_a_host_that_refuses_to_pin_is_named_once_and_given_pageable_chunks(monkeypatch, capsys):
    import torch

    from tensorfold.cuda.host_tier import HostTier

    asked = []

    def alloc(self, pinned):
        asked.append(pinned)
        if pinned and asked.count(True) > 2:
            raise RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported")
        return torch.zeros(self.chunk, dtype=torch.uint8)

    monkeypatch.setattr(HostTier, "_alloc", alloc)
    tier = _tier(8 * 256, 256, pin=True)
    for ids in ([1], [2], [3]):
        assert tier.put(ids, _small(torch, ids), None)
    assert asked == [True, True, True, False, False, False, False] and not tier.pin
    out = capsys.readouterr().out
    assert out.count("[tensorfold] RAM tier") == 1 and "refused more (CUDA error: out of memory)" in out
    ids, back, _ = tier.take([2, 0])
    assert ids == [2] and _same(back.conv[0], torch.zeros(300, dtype=torch.uint8))


# ------------------------------------------------------------------------------------------ rows shared by prefix
def _prefix_rows(torch, ids, salt: int):
    """One row of 16 bytes per id as (rows, 1, 8) bf16, row p a function of ids[:p + 1] alone (a rolling hash)."""

    h, hs = salt + 1, []
    for t in ids:
        h = (h * 1000003 + t + 1) % (1 << 31)
        hs.append(h)
    x = torch.tensor(hs, dtype=torch.int64).reshape(-1, 1) * torch.arange(1, 33, 2) + torch.arange(16)
    return ((x ^ (x >> 11)) % 256).to(torch.uint8).view(torch.bfloat16).reshape(len(ids), 1, 8)


def _pure(torch, ids, spare: int = 3, seed: int = 0) -> FakeState:
    """A prompt-end state as prefill leaves it: attention rows below ``pos`` a function of the ids alone, the DeltaNet
    state and the rows past ``pos`` noise of ``seed``. Attention rows take 64 bytes each (two layers, K and V)."""

    gen = torch.Generator().manual_seed(seed)
    st = FakeState()
    st.pos, st.limit, st.rope_delta = len(ids), 4096, 0
    st.conv, st.rec, st.kv = [], [], []
    for layer, linear in enumerate((True, False, True, False)):
        st.conv.append(_random(torch, gen, (2, 16), torch.bfloat16) if linear else None)
        st.rec.append(_random(torch, gen, (2, 4), torch.float32) if linear else None)
        st.kv.append(None if linear else tuple(
            torch.cat([_prefix_rows(torch, ids, 2 * layer + part), _random(torch, gen, (spare, 1, 8), torch.bfloat16)])
            for part in (0, 1)))
    return st


def _own(st) -> int:
    """Bytes a spill copies whole: the DeltaNet state."""

    return sum(t.numel() * t.element_size() for t in st.conv + st.rec if t is not None)


def _kept(st) -> dict:
    """What a take must give back: rows below ``pos``, the DeltaNet state and every buffer's row count."""

    return {"pos": st.pos, "own": [None if t is None else t.clone() for t in st.conv + st.rec],
            "kv": [None if kv is None else tuple(t[:st.pos].clone() for t in kv) for kv in st.kv],
            "rows": [None if kv is None else tuple(t.shape[0] for t in kv) for kv in st.kv]}


def _exact(back, want: dict, rows: bool = True) -> bool:
    return (back.pos == want["pos"] and all((a is None) == (b is None) and (a is None or _same(a, b))
                                            for a, b in zip(back.conv + back.rec, want["own"]))
            and all((kv is None) == (w is None) and (kv is None or all(_same(t[:back.pos], u) for t, u in zip(kv, w)))
                    for kv, w in zip(back.kv, want["kv"]))
            and (not rows or [None if kv is None else tuple(t.shape[0] for t in kv) for kv in back.kv] == want["rows"]))


def _scrub(torch, st) -> None:
    """The device side reused after the spill: only the host copy holds the bytes now."""

    for t in [t for t in st.conv + st.rec if t is not None] + [t for kv in st.kv if kv is not None for t in kv]:
        t.view(torch.uint8).fill_(0x5A)


def _spill(torch, tier, ids, seed: int = 0) -> tuple[dict, int, int]:
    """Put a pure state; what a take must give back, and the attention and other bytes the spill copied."""

    st = _pure(torch, ids, seed=seed)
    want, before = _kept(st), tier.to_host
    assert tier.put(ids, st, None)
    _scrub(torch, st)
    return want, tier.to_host - before - _own(st), _own(st)


@pytest.mark.torch
def test_a_prefix_of_rows_on_the_host_copies_no_attention_bytes():
    """A prompt's message-start state spilled after its prompt end (and again once that entry was taken): only the
    DeltaNet state is copied; the prefix's rows come back from the longer entry's segment."""

    import torch

    rng = random.Random(1)
    ids = [rng.randrange(1, 50) for _ in range(40)]
    tier = _tier(1 << 20, 1000)
    whole, copied, _ = _spill(torch, tier, ids)
    assert copied == 40 * 64 and len(tier.segments) == 1
    head, copied, own = _spill(torch, tier, ids[:25], seed=1)
    assert copied == 0 and len(tier.segments) == 1 and tier.segments[0].refs == 2
    before = tier.from_host
    assert _exact(tier.take(ids[:25] + [7])[1], head) and tier.from_host - before == own + 25 * 64
    assert _exact(tier.take(ids + [7])[1], whole) and tier.segments[0].refs == 0
    for n in (40, 12):                                     # taken: the segment stays for the next spill to chain onto
        want, copied, _ = _spill(torch, tier, ids[:n], seed=n)
        assert copied == 0 and len(tier.segments) == 1
        assert _exact(tier.take(ids[:n] + [7])[1], want)


@pytest.mark.torch
@pytest.mark.parametrize("chunk", [64, 1000, 1 << 20])
def test_an_extension_copies_only_its_new_rows_and_chains_come_back_exact(chunk):
    """A conversation's turns: each extends the last, so each spill copies its new rows only; then a prompt agreeing
    with the second turn partway into its rows, and one agreeing with that partway, chain three segments and use the
    last two partly. Every entry comes back byte for byte, whatever the chunk boundaries."""

    import torch

    rng = random.Random(2)
    more = lambda n: [rng.randrange(1, 50) for _ in range(n)]       # noqa: E731
    a = more(30)
    b = a + more(20)
    c = b[:42] + [99] + more(14)                           # agrees with b through row 41
    d = c[:50] + [98] + more(9)                            # agrees with c through row 49
    tier, wants = _tier(1 << 24, chunk), {}
    for ids, new in ((a, 30), (b, 20), (c, 15), (d, 10)):
        wants[tuple(ids)], copied, _ = _spill(torch, tier, ids, seed=len(ids))
        assert copied == new * 64, len(ids)
    (entry,) = [e for e in tier.entries if e.ids == d]
    sa, sb, sc, sd = tier.segments
    assert [(s, i, j) for s, i, j in entry.pieces] == [(sa, 0, 30), (sb, 30, 42), (sc, 42, 50), (sd, 50, 60)]
    assert [(s.start, s.end) for s in tier.segments] == [(0, 30), (30, 50), (42, 57), (50, 60)]
    for ids in (c, a, d, b):
        got, back, _ = tier.take(ids + [0])
        assert got == ids and _exact(back, wants[tuple(ids)])
    assert not tier.entries and len(tier.segments) == 4 and all(s.refs == 0 for s in tier.segments)


@pytest.mark.torch
def test_diverging_ids_never_share_rows():
    """Rows past the first differing id are copied again, as are the rows of a cache of another shape; rows before it
    are shared."""

    import torch

    tier = _tier(1 << 20, 256)
    base = list(range(1, 31))
    first, copied, _ = _spill(torch, tier, base)
    fork = base[:10] + [77] + base[11:]                    # differs at row 10 only: rows 10 on differ too
    other, copied, _ = _spill(torch, tier, fork, seed=1)
    assert copied == 20 * 64
    stranger, copied, _ = _spill(torch, tier, [5] + base[1:], seed=2)
    assert copied == 30 * 64
    wide = _pure(torch, base[:20])
    wide.kv = [None if kv is None else tuple(torch.cat([t, t], dim=2) for t in kv) for kv in wide.kv]
    rows, before = _kept(wide), tier.to_host
    assert tier.put(base[:20], wide, None) and tier.to_host - before == _own(wide) + 20 * 128
    for ids, want in ((base, first), (fork, other), ([5] + base[1:], stranger), (base[:20], rows)):
        assert _exact(tier.take(ids + [0])[1], want)


@pytest.mark.torch
def test_a_restore_into_given_buffers_fills_their_rows_below_pos_and_hands_them_back():
    import torch

    tier = _tier(1 << 20, 1000)
    ids = list(range(3, 43))
    want, _, _ = _spill(torch, tier, ids[:25])
    _spill(torch, tier, ids, seed=1)                       # the entry at 25 now reads another entry's segment
    gen = torch.Generator().manual_seed(4)
    into = [None if kv is None else (_random(torch, gen, (32, 1, 8), torch.bfloat16),
                                     _random(torch, gen, (32, 1, 8), torch.bfloat16)) for kv in want["kv"]]
    past = [None if kv is None else tuple(t[25:].clone() for t in kv) for kv in into]
    for wrong in ([None] * 4, [None if kv is None else (kv[0][:24], kv[1][:24]) for kv in into],
                  [None if kv is None else (kv[0].float(), kv[1]) for kv in into]):
        with pytest.raises(ValueError, match="at least|match"):
            tier.take(ids[:26], into=wrong)
    assert len(tier.entries) == 2                          # a refused restore leaves the entry where it was
    got, back, _ = tier.take(ids[:26], into=into)
    assert got == ids[:25] and _exact(back, want, rows=False)
    assert all(a is b for a, b in zip(back.kv, into))
    assert all(kv is None or all(_same(t[25:], u) for t, u in zip(kv, p)) for kv, p in zip(back.kv, past))


@pytest.mark.torch
def test_idle_segments_make_room_before_entries_least_recently_used_first():
    """Each entry takes four chunks of DeltaNet state and four of rows. Taken entries leave idle segments; a spill
    needing room frees the least recently used of them first, then the oldest entry, whose segment goes next unless
    another entry holds rows of it."""

    import torch

    tier = _tier(24 * 256, 256)
    a, b, c, d, f, g = ([k] * 10 for k in range(1, 7))
    for ids in (a, b, c):
        _spill(torch, tier, ids)
    tier.take(b + [0])
    tier.take(a + [0])                                     # a's segment read last
    assert [s.ids for s in tier.segments] == [a, b, c] and [s.refs for s in tier.segments] == [0, 0, 1]
    _spill(torch, tier, d)                                 # 8 chunks free: nothing leaves
    assert [x.ids for x in tier.entries] == [c, d] and tier.used == 24 * 256
    want, copied, _ = _spill(torch, tier, c[:6])           # rows on the host: four chunks of state, b's segment goes
    assert copied == 0 and [s.ids for s in tier.segments] == [a, c, d]
    _spill(torch, tier, f)                                 # a's segment, then c (its segment holds c[:6]'s rows)
    assert [x.ids for x in tier.entries] == [d, c[:6], f] and [s.ids for s in tier.segments] == [c, d, f]
    _spill(torch, tier, g)                                 # d, then its segment
    assert [x.ids for x in tier.entries] == [c[:6], f, g] and [s.ids for s in tier.segments] == [c, f, g]
    assert tier.held == tier.capacity and _exact(tier.take(c[:6] + [0])[1], want)


@pytest.mark.torch
def test_a_cover_that_crowds_the_entry_out_is_let_go_and_its_rows_copied_anew():
    """Three rows of a 30-row idle segment and a drafter snapshot: with the segment the entry passes the budget, so the
    segment goes and the entry copies its own three rows."""

    import torch

    tier = _tier(12 * 256, 256)
    ids = list(range(1, 31))
    _spill(torch, tier, ids)                               # four chunks of state, eight of rows
    tier.take(ids + [0])
    st, snap = _pure(torch, ids[:3]), ([torch.arange(700, dtype=torch.int32).to(torch.uint8)], [None], 3, 4)
    want, before = _kept(st), tier.to_host
    assert tier.put(ids[:3], st, snap)
    assert tier.to_host - before == _own(st) + 700 + 3 * 64 and [(s.start, s.end) for s in tier.segments] == [(0, 3)]
    _, back, back_snap = tier.take(ids[:3] + [0])
    assert _exact(back, want) and _same(back_snap[0][0], snap[0][0]) and back_snap[1:] == ([None], 3, 4)


@pytest.mark.torch
@pytest.mark.parametrize("seed", range(3))
def test_a_long_random_run_stays_in_budget_and_every_take_is_exact(seed):
    """Conversations that branch and grow, spilled and taken in random order under a budget that forces evictions:
    the chunks never pass it, references match the entries' pieces, each entry's pieces cover [0, pos) with rows
    computed for its ids, and every take (into given buffers or not) gives back the bytes spilled."""

    import torch

    rng = random.Random(seed)
    chunk = rng.choice([64, 256, 1000])
    tier = _tier(rng.randrange(40, 120) * chunk, chunk)
    talks, wants, takes = [[]], {}, 0
    for step in range(400):
        if rng.random() < 0.6 or not tier.entries:
            base = rng.choice(talks)
            ids = base[:rng.randrange(0, len(base) + 1)] + [rng.randrange(1, 6) for _ in range(rng.randrange(1, 12))]
            st = _pure(torch, ids, seed=step)
            if tier.put(ids, st, None):
                wants[tuple(ids)] = _kept(st)
                talks.append(ids)
            _scrub(torch, st)
        else:
            entry = rng.choice(tier.entries)
            prompt = entry.ids + [rng.randrange(1, 6)]
            want = wants[tuple(tier.longest(prompt).ids)]
            into = None
            if rng.random() < 0.5:
                into = [None if kv is None else tuple(torch.zeros(want["pos"] + 2, 1, 8, dtype=torch.bfloat16)
                                                      for _ in kv) for kv in want["kv"]]
            _, back, _ = tier.take(prompt, into=into)
            assert _exact(back, want, rows=into is None) and all(a is b for a, b in zip(back.kv, into or []))
            takes += 1
        assert tier.held <= tier.capacity and tier.used <= tier.capacity * chunk
        pieces = [p for x in tier.entries for p in x.pieces]
        assert all(s.refs == sum(p[0] is s for p in pieces) for s in tier.segments)
        assert all(p[0] in tier.segments for p in pieces)
        for x in tier.entries:
            ends = [0] + [j for _, _, j in x.pieces]
            assert ends[-1] == len(x.ids) and all(i == ends[k] and i < j for k, (_, i, j) in enumerate(x.pieces))
            assert all(s.start <= i and j <= s.end and s.ids[:j] == x.ids[:j] for s, i, j in x.pieces)
    assert takes > 50 and tier.stats()["from_host"] > 0


def test_the_prefix_cache_hands_every_entry_it_lets_go_to_on_evict():
    gone = []
    cache = PrefixCache(2, on_evict=lambda ids, st, snap: gone.append((ids, st, snap)))
    cache.add([1], "s1", "p1")
    cache.add([1, 2], "s12", None)
    assert cache.longest([1, 2, 3])[0] == [1, 2]
    cache.add([5], "s5", None)                           # past keep: the oldest entry never resumed from
    assert gone == [([1], "s1", "p1")]
    cache.add([5], "s5b", None)                          # the same ids replace an entry: nothing leaves
    assert len(gone) == 1
    cache.add([1, 2, 3, 4], "s1234", None)
    assert gone[1:] == [([5], "s5b", None)]
    cache.add([1, 2, 7], "s127", None)
    del gone[:]
    cache.entries.insert(0, ([1, 2, 3], "s123", None))
    cache.drop([1, 2])                                   # every extension, oldest first
    assert gone == [([1, 2, 3], "s123", None), ([1, 2, 7], "s127", None)]
    assert [ids for ids, _, _ in cache.entries] == [[1, 2]]
    cache.drop([1, 2])
    assert len(gone) == 2
    silent = PrefixCache(1)
    silent.add([1], None, None)
    silent.add([2], None, None)
    silent.drop([])
    assert silent.entries == []


# ------------------------------------------------------------------------------------ the one-stream engine
def _held_state(torch, ids, spare: int = 5) -> FakeState:
    """A committed state whose attention buffer holds its ids (``spare`` rows past them) beside a DeltaNet state."""

    st = FakeState()
    st.pos, st.limit = len(ids), 1 << 20
    keys = torch.zeros((len(ids) + spare, 1, 1))
    keys[:len(ids), 0, 0] = torch.tensor([float(t) for t in ids])
    st.conv, st.rec, st.kv = [torch.ones(3, 4).bfloat16(), None], [torch.ones(2, 2), None], [None, (keys, -keys)]
    return st


def _held(st) -> list[int]:
    return [] if st is None else [int(x) for x in st.kv[1][0][:st.pos, 0, 0].tolist()]


def _one_gpu(monkeypatch, *, tier: bool = True, points=None):
    import torch

    from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, Qwen27Engine

    engine = object.__new__(Qwen27Engine)
    engine.tp, engine.rank, engine.max_rows, engine.allow_copy = 1, 0, 12, True
    engine.w = SimpleNamespace(norm=SimpleNamespace(device="cpu"))
    engine.draft, engine.eos, engine.points = None, (0,), points
    engine.context_window, engine.scheduler, engine.multi = 1 << 20, None, None
    engine.cache = PrefixCache(KEEP_ONE)
    if tier:
        engine.tier = _tier(1 << 24, 4096)
        engine.cache.on_evict = engine.tier.put
    starts = []
    fake = types.ModuleType(PKG + ".decode")

    def prefill(w, prompt, sampling, drafter=None, *, state=None, limit=0, stops=(), keep=None, keep_at=None,
                vision=None):
        start = state.pos if state is not None else 0
        assert _held(state) == list(prompt[:start]) and start < len(prompt)
        starts.append(start)
        for p in stops:
            if start < p < len(prompt) and keep is not None:
                keep(p, _held_state(torch, prompt[:p]), None)
        st = _held_state(torch, prompt)
        return (st, 7) if keep_at is None else (st, 7, (_held_state(torch, prompt[:keep_at]), None))

    def draft_decode(w, st, prompt, pending, count, sampling, draft, *, max_rows, allow_copy, on_tokens, inplace,
                     stop_eos=True):
        on_tokens([8, 9])
        return SimpleNamespace(seconds=0.1, rounds=2, widths=[1, 1], drafted_rows=0, accepted_drafts=0)

    fake.prefill, fake.draft_decode = prefill, draft_decode
    monkeypatch.setitem(sys.modules, PKG + ".decode", fake)
    return engine, starts


def _run(engine, prompt):
    return engine.generate(list(prompt), 4, None, lambda tokens: False)


def _entries(cache) -> list[list[int]]:
    return [ids for ids, _, _ in cache.entries]


def _turn(head: list[int], seed: int) -> tuple[list[int], list[int]]:
    """A prompt ending in the generation prompt, and its next turn (the history renders an empty reasoning block)."""

    rng = random.Random(seed)
    first = head + [THINK, NL]
    return first, head + [THINK, NL2, END_THINK, NL2] + [rng.randrange(20, 90) for _ in range(7)] + [THINK, NL]


@pytest.mark.torch
def test_a_conversation_dropped_under_a_shared_system_prompt_resumes_from_the_tier(monkeypatch):
    """Two conversations after one system prompt kept at its message start: the second resumes from the system prompt
    and drops the first's entry (it would write into the buffer they share), which the tier keeps; the first's next
    turn resumes where it ended, not from the system prompt, and its entry is back in the GPU cache."""

    system = list(range(1000, 1300))
    points = lambda prompt: [len(system)] if len(prompt) > len(system) else []      # noqa: E731
    rng = random.Random(3)
    a1, a2 = _turn(system + [rng.randrange(20, 90) for _ in range(300)], seed=4)
    b1, _ = _turn(system + [rng.randrange(20, 90) for _ in range(300)], seed=5)
    for tier in (True, False):
        engine, starts = _one_gpu(monkeypatch, tier=tier, points=points)
        _run(engine, a1)
        assert _entries(engine.cache) == [system, a1[:-1]]
        _run(engine, b1)
        assert starts[-1] == len(system) and a1[:-1] not in _entries(engine.cache)
        stats = _run(engine, a2)
        if not tier:
            assert starts[-1] == stats["cached"] == len(system)        # without the tier: from the system prompt
            continue
        assert [e.ids for e in engine.tier.entries] == []
        assert starts[-1] == stats["cached"] == len(a1) - 1
        assert _entries(engine.cache) == [b1[:-1], system, a1[:-1], a2[:-1]]      # the system prompt hit too
        for ids, st, _ in engine.cache.entries:
            assert _held(st) == ids


@pytest.mark.torch
def test_entries_evicted_past_keep_come_back_and_each_resume_takes_the_longest_match(monkeypatch):
    """Unrelated conversations past the GPU cache's ``KEEP_ONE`` entries: evicted ones go to the tier, and each
    request resumes from the longest strict prefix held on either side."""

    from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE

    engine, starts = _one_gpu(monkeypatch)
    rng = random.Random(9)
    talks = [[rng.randrange(20, 90) for _ in range(rng.randrange(3, 30))] for _ in range(KEEP_ONE + 3)]
    for step in range(60):
        k = rng.randrange(len(talks))
        prompt, following = _turn(talks[k], seed=step)
        both = _entries(engine.cache) + [e.ids for e in engine.tier.entries]
        want = max((len(ids) for ids in both if len(ids) < len(prompt) and prompt[:len(ids)] == ids), default=0)
        assert _run(engine, prompt)["cached"] == starts[-1] == want
        assert len(engine.cache.entries) <= KEEP_ONE
        talks[k] = following[:-2]
    assert engine.tier.entries and any(start for start in starts)
    for ids, st, _ in engine.cache.entries:
        assert _held(st) == ids


# ---------------------------------------------------------------------------------- the concurrent decoder
@pytest.mark.torch
def test_the_concurrent_decoder_admits_a_returning_conversation_from_the_tier(cuda_modules, monkeypatch):  # noqa: F811
    m = cuda_modules
    rec = _stand_in_prefills(m, monkeypatch)[0]
    w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=8), norm=m.torch.zeros(1), head=SimpleNamespace(n=8),
                        layers=[], rank=0)
    tier = _tier(1 << 22, 4096)
    dec = m.multi.MultiDecoder(w, None, keep=1, context=CONTEXT, tier=tier)
    a1, a2 = _turn([40, 41, 42, 43], seed=1)
    b1, _ = _turn([50, 51], seed=2)
    for prompt, cached in ((a1, 0), (b1, 0), (a2, len(a1) - 1)):
        s = m.multi.Stream(list(prompt), 3, None)
        dec.admit(s)
        dec._fill()
        assert s.cached == rec.prefills[-1].start == cached
        dec.finish([s])
    assert _entries(dec.cache) == [a2[:-1]] and [e.ids for e in tier.entries] == [b1[:-1], a1[:-1]]
    ids, st, _ = dec.cache.entries[0]
    assert _ids_of(st) == ids


# ------------------------------------------------------------------------------------------------- the CLI
def test_serve_parses_the_ram_tier():
    assert cli.build_parser().parse_args(["serve", "owner/model"]).ram_tier_gib == 0.0
    assert cli.build_parser().parse_args(["serve", "owner/model", "--ram-tier-gib", "24.5"]).ram_tier_gib == 24.5


@pytest.mark.parametrize("flags,backend,family,message", [
    (["--ram-tier-gib", "8"], "mlx", "qwen3_5", "on MLX, --spill-gib writes them to disk"),
    (["--ram-tier-gib", "8"], "cuda", "qwen4_exp", "has no host RAM tier"),
    (["--ram-tier-gib", "8"], "cuda", "nemotron_h", "has no host RAM tier"),
    (["--ram-tier-gib", "8"], "cuda", "glm5_next", "has no host RAM tier"),
    (["--ram-tier-gib", "8", "--tp", "2"], "cuda", "qwen3_5", "drop it with --tp 2"),
    (["--ram-tier-gib", "-1"], "cuda", "qwen3_5", "is a GiB count"),
])
def test_the_ram_tier_is_refused_before_any_download(tmp_path, monkeypatch, flags, backend, family, message):
    import importlib

    from tensorfold import families, hub

    module = importlib.import_module(f"tensorfold.families.{family}")
    found = SimpleNamespace(title=module.TITLE, package=module, model_type=family)
    monkeypatch.setattr(families, "detect", lambda path: found)
    monkeypatch.setattr(cli, "_backend", lambda choice, fam: backend)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights were fetched before the refusal"))
    monkeypatch.setattr(families, "require_readable",
                        lambda *a: pytest.fail("the checkpoint was read before the refusal"))
    command = ["serve", str(tmp_path), "--no-update-check"] + flags
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(cli.build_parser().parse_args(command))


@pytest.mark.parametrize("flags", [[], ["--ram-tier-gib", "0"], ["--ram-tier-gib", "24"]])
def test_qwen27_on_cuda_takes_the_ram_tier(tmp_path, flags):
    from tensorfold.families import qwen3_5

    args = cli.build_parser().parse_args(["serve", str(tmp_path)] + flags)
    family = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    assert cli._check_serve_options(args, family, "cuda") is None
    if not flags:
        assert cli._check_serve_options(args, family, "mlx") is None           # off: no refusal anywhere


@pytest.mark.parametrize("gib", [0.0, 2.5])
def test_the_ram_tier_reaches_the_engine_only_when_asked(tmp_path, monkeypatch, gib):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import build

    monkeypatch.setattr(build, "hip", lambda: False)
    made = []
    family = SimpleNamespace(title="Test family", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: made.append(k) or
                                                     SimpleNamespace(context_window=4096)))
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=4096))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--ram-tier-gib", str(gib)]
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("ram_tier_gib") == (gib or None)


@pytest.mark.torch
def test_the_27b_engine_gets_the_tier_in_bytes(tmp_path, monkeypatch):
    from tensorfold.families import qwen3_5
    from tensorfold.families.qwen3_5.cuda import engine as q27_engine

    made = []
    monkeypatch.setattr(q27_engine, "Qwen27Engine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path), ram_tier_gib=2.5)
    qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path))
    assert [k["ram_tier"] for k in made] == [int(2.5 * 1024**3), 0]



def test_reserve_carves_the_budget_into_free_chunks_up_front():
    """``reserve`` takes the whole budget in power-of-two slabs (none past it); puts then use those chunks."""

    import torch

    tier = _tier(5 * 64 + 32, 32, pin=True)
    tier._alloc = lambda pinned: pytest.fail("a put allocated a chunk after reserve")
    real_empty = torch.empty
    sizes = []

    def empty(size, **kwargs):
        sizes.append(size)
        return real_empty(size, dtype=kwargs["dtype"])          # CPU tensors: no pinning in a unit test

    torch.empty = empty
    try:
        assert tier.reserve(slab=128) == 11 * 32
    finally:
        torch.empty = real_empty
    assert sizes == [128, 128, 64, 32] and tier.held == tier.capacity == 11 and len(tier.free) == 11
    assert all(c.numel() == 32 for c in tier.free) and tier.used == 0


@pytest.mark.torch
@pytest.mark.parametrize("budget, refused, kept, alive", [
    (8 * 1024, 3, 8, [True, True, False]),               # the last 1 KiB slab goes back
    (2 * 1024 + 512 + 256, 3, 4, [True, False, False]),  # a 512-byte tail slab is not a slab's worth: one more goes
    (8 * 1024, 0, 0, None),                              # the first slab refused: nothing to give back
])
def test_a_refused_reserve_gives_a_slab_back_for_other_pinned_buffers(monkeypatch, capsys, budget, refused, kept,
                                                                      alive):
    """When the host refuses a slab, ``reserve`` unpins at least a slab's worth of the last ones (the decode path pins
    staging buffers every round), no chunk of them left anywhere when the host cache empties; puts fill the rest of
    the budget with pageable chunks."""

    import weakref

    import torch

    from tensorfold.cuda import host_tier

    real_empty, slabs, pageable, emptied = torch.empty, [], [], []

    def empty(size, **kwargs):
        if not kwargs.get("pin_memory"):
            pageable.append(size)
        elif len(slabs) == refused:
            raise RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported")
        block = real_empty(size, dtype=kwargs["dtype"])      # CPU tensors: no pinning in a unit test
        if kwargs.get("pin_memory"):
            slabs.append(weakref.ref(block))
        return block

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(host_tier, "_empty_host_cache", lambda: emptied.append([s() is not None for s in slabs]))
    tier = _tier(budget, 256, pin=True)
    assert tier.reserve(slab=1024) == kept * 256
    assert tier.held == len(tier.free) == kept and not tier.pin
    assert emptied == ([] if alive is None else [alive])
    out = capsys.readouterr().out
    assert out.count("[tensorfold] RAM tier") == 1 and "refused more (CUDA error: out of memory)" in out
    assert ("goes back for other pinned buffers" in out) is (alive is not None)
    for i in range(tier.capacity):                       # one 256-byte chunk an entry: the whole budget
        assert tier.put([i + 1], _small(torch, [i + 1], 256), None)
    assert tier.held == tier.capacity and pageable == [256] * (tier.capacity - kept)
