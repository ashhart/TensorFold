"""Host checks for the 27B CUDA engine's prompt-end cache entry, kept one token before the prompt's end (no GPU).

A chat prompt ends in the generation prompt, ``<think>`` and a newline (Qwen token 198); a next request that sends
that turn back without its reasoning renders an empty reasoning block, ``<think>`` and two newlines (token 271). An
entry for the whole prompt then differs from the next prompt in its last token and does not match; the entry at
``len(prompt) - 1`` does.

- The one-stream engine's bookkeeping, on one GPU and on both ranks of two, with stand-ins for ``decode`` and
  ``decode_tp`` (no PyTorch needed): which entry each prompt resumes from and what each entry holds.
- The concurrent decoder's admissions (``MultiDecoder``), one GPU and both ranks, with a stand-in prefill; with a
  stand-in drafter on ``prefill_state`` and a stand-in chunk, the drafter contexts allocated around a stream's first
  commit.
- ``prefill_state(keep_at=...)`` with a stand-in ``prefill_chunk``: the spans, where a chunk is cut, the kept state's
  buffers, and the drafter's calls and snapshots.
- The whole prefill on row-wise CPU stand-ins for the kernels: the prompt's state, last logits and first token are a
  prefill's without ``keep_at``, the kept state is a fresh prefill of the prefix, and the next turn resumed from it is
  a fresh prefill. This checks the Python around the kernels, not the kernels
  (``tests/cuda/test_qwen27_prompt_end_cache.py`` does, on a GPU).

Where Triton is missing, the PyTorch parts import the CUDA modules with an import-only stand-in; no kernel runs.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
import weakref
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.cuda.streams import PrefixCache
from tensorfold.families.qwen3_5.cuda.engine import KEEP, KEEP_ONE, Qwen27Engine, entry_end

NL, NL2 = 198, 271          # Qwen's "\n" and "\n\n"
THINK, END_THINK = 300, 301  # stand-ins for <think> and </think> (the real ids are past the toy vocabularies)
PKG = "tensorfold.families.qwen3_5.cuda"


def _chat(turns):
    """A toy chat rendering: each past turn's reply follows an empty reasoning block, the new turn opens one."""

    ids = [1, 2, 3]
    for user, reply in turns[:-1]:
        ids += [10] + user + [11, THINK, NL2, END_THINK, NL2] + reply + [12]
    return ids + [10] + turns[-1][0] + [11, THINK, NL]


def _reply(prompt, count=4):
    """A deterministic reply for a prompt (both ranks of a two-rank engine produce the same)."""

    seed = sum((i + 1) * t for i, t in enumerate(prompt)) % 9973
    return [1 + (seed * 31 + 7 * i) % 97 for i in range(count)]


def test_the_entry_ends_one_token_before_the_prompt_end():
    assert [entry_end(list(range(n))) for n in (1, 2, 3, 10)] == [1, 1, 2, 9]
    first = _chat([([40, 41], None)])
    second = _chat([([40, 41], [50]), ([42], None)])
    assert (first[-1], second[len(first) - 1]) == (NL, NL2)
    assert second[:entry_end(first)] == first[:entry_end(first)]


# --------------------------------------------------------------------------- the one-stream engine's bookkeeping
class FakeState:
    """A committed state that knows which tokens it holds."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.pos = len(self.ids)


class FakeDrafter:
    layers = 2

    def __init__(self):
        self.restored = []

    def restore(self, snap):
        self.restored.append(snap)


class Recorder:
    def __init__(self):
        self.prefills = []

    def prefill(self, w, prompt, sampling, drafter=None, *, state=None, keep_at=None, rank=None, limit=0, stops=(),
                keep=None, vision=None):
        start = state.pos if state is not None else 0
        if state is not None:
            assert list(prompt[:start]) == state.ids, "resumed from a state that is not a prefix of the prompt"
        assert start < len(prompt)
        self.prefills.append(SimpleNamespace(prompt=list(prompt), start=start, keep_at=keep_at, drafter=drafter,
                                             rank=rank))
        pending = _reply(prompt)[0]
        if keep_at is None:
            return FakeState(prompt), pending
        assert start <= keep_at <= len(prompt)
        snap = ("drafter at", keep_at) if drafter is not None else None
        return FakeState(prompt), pending, (FakeState(prompt[:keep_at]), snap)

    @staticmethod
    def decode(st, prompt, pending, max_tokens):
        assert st.ids == list(prompt)
        tokens = ([pending] + _reply(prompt)[1:])[:max(1, max_tokens)]
        return SimpleNamespace(tokens=tokens, seconds=0.1, rounds=len(tokens), widths=[1] * len(tokens), drafted_rows=0,
                               accepted_drafts=0)


def _bare_engine(tp=1, rank=0, drafter=None):
    engine = object.__new__(Qwen27Engine)
    engine.tp, engine.rank, engine.max_rows, engine.allow_copy = tp, rank, 12, True
    engine.w = SimpleNamespace(norm=SimpleNamespace(device="cpu"))
    engine.draft, engine.eos, engine.cache, engine.points = drafter, (0,), PrefixCache(KEEP_ONE), None
    engine.context_window, engine.scheduler, engine.multi = 1 << 20, None, None
    return engine


def _one_gpu(monkeypatch, drafter=None):
    rec = Recorder()
    engine = _bare_engine(drafter=drafter)
    fake = types.ModuleType(PKG + ".decode")
    fake.prefill = rec.prefill

    def draft_decode(w, st, prompt, pending, count, sampling, draft, *, max_rows, allow_copy, on_tokens, inplace,
                     stop_eos=True):
        # the decode may commit into the prompt state: no entry holds it
        assert inplace and all(st is not entry for _, entry, _ in engine.cache.entries)
        result = rec.decode(st, prompt, pending, count)
        on_tokens(result.tokens[1:])
        return result

    fake.draft_decode = draft_decode
    monkeypatch.setitem(sys.modules, PKG + ".decode", fake)
    return engine, rec


def _run(engine, prompt, max_tokens=4, **kwargs):
    return engine.generate(list(prompt), max_tokens, None, lambda tokens: False, **kwargs)


def _entries(engine):
    return [ids for ids, _, _ in engine.cache.entries]


def _check_entries(engine):
    """At most ``KEEP_ONE`` entries, each holding the state for exactly its ids."""

    assert len(engine.cache.entries) <= KEEP_ONE
    for ids, st, _ in engine.cache.entries:
        assert st.ids == ids and st.pos == len(ids)


def test_a_request_keeps_the_state_before_its_last_prompt_token(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    prompt = list(range(20, 30))
    stats = _run(engine, prompt)
    assert [(p.start, p.keep_at) for p in rec.prefills] == [(0, 9)]
    assert _entries(engine) == [prompt[:9]] and stats["cached"] == 0
    _check_entries(engine)


def test_the_next_chat_turn_resumes_one_token_before_the_last_prompt_end(monkeypatch):
    """The '<think>\\n' / '<think>\\n\\n' case: the whole last prompt is no prefix of the next one; all but its last
    token is."""

    engine, rec = _one_gpu(monkeypatch)
    first = _chat([([40, 41, 42], None)])
    _run(engine, first)
    second = _chat([([40, 41, 42], [50, 51]), ([43, 44], None)])
    assert second[:len(first)] != first and second[:len(first) - 1] == first[:-1]
    stats = _run(engine, second)
    assert rec.prefills[-1].start == len(first) - 1 == stats["cached"]
    assert _entries(engine) == [first[:-1], second[:-1]]
    _check_entries(engine)


def test_a_prompt_that_extends_the_whole_prompt_resumes_one_token_earlier(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    first = list(range(20, 30))
    _run(engine, first)
    _run(engine, first + _reply(first)[:-1] + [90, 91])
    assert rec.prefills[-1].start == len(first) - 1
    _check_entries(engine)


def test_an_identical_prompt_resumes_and_prefills_one_token(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    prompt = list(range(20, 30))
    _run(engine, prompt)
    stats = _run(engine, prompt)
    assert rec.prefills[-1].start == len(prompt) - 1 == stats["cached"]
    assert _entries(engine) == [prompt[:-1]]                  # the same ids: one entry, the newer state
    _check_entries(engine)


def test_a_one_token_prompt_keeps_its_whole_prompt(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    _run(engine, [7])
    assert rec.prefills[-1].keep_at == 1 and _entries(engine) == [[7]]
    _run(engine, [7, 8])
    assert rec.prefills[-1].start == 1 and rec.prefills[-1].keep_at == 1
    _check_entries(engine)


def test_the_serial_reference_keeps_nothing_and_leaves_the_cache_alone(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    _run(engine, list(range(20, 30)))
    before = list(engine.cache.entries)
    _run(engine, list(range(20, 30)), draft=False)
    assert rec.prefills[-1].start == 0 and rec.prefills[-1].keep_at is None
    assert engine.cache.entries == before


def test_a_client_stop_after_the_first_token_keeps_the_prompt_entry(monkeypatch):
    engine, rec = _one_gpu(monkeypatch)
    prompt = list(range(20, 30))
    stats = engine.generate(prompt, 8, None, lambda tokens: True)
    assert set(stats) == {"prefill_s", "cached"}
    assert _entries(engine) == [prompt[:-1]]


def test_the_drafter_resumes_from_the_entrys_own_snapshot(monkeypatch):
    drafter = FakeDrafter()
    engine, rec = _one_gpu(monkeypatch, drafter)
    first = _chat([([40, 41, 42], None)])
    _run(engine, first)
    assert engine.cache.entries[0][2] == ("drafter at", len(first) - 1)
    second = _chat([([40, 41, 42], [50, 51]), ([43, 44], None)])
    _run(engine, second)
    empty = ([None] * drafter.layers, [None] * drafter.layers, 0, 0)
    assert drafter.restored == [empty, ("drafter at", len(first) - 1)]
    assert rec.prefills[-1].drafter is drafter
    assert [snap for _, _, snap in engine.cache.entries] == [("drafter at", len(first) - 1),
                                                            ("drafter at", len(second) - 1)]


def _expected_cache(cache, prompt):
    """The engine's rule on a copy of its cache: resume from the longest strict prefix, drop its extensions, add."""

    model = PrefixCache(cache.keep)
    model.entries, model.hit = [(ids, None, None) for ids, _, _ in cache.entries], set(cache.hit)
    hit = model.longest(prompt)
    if hit is not None:
        model.entries = [e for e in model.entries if len(e[0]) <= len(hit[0]) or e[0][:len(hit[0])] != hit[0]]
    model.add(prompt[:entry_end(prompt)], None, None)
    return (len(hit[0]) if hit else 0), [ids for ids, _, _ in model.entries]


@pytest.mark.parametrize("seed", range(4))
def test_entries_stay_two_and_each_resume_takes_the_longest_match(monkeypatch, seed):
    """Chat turns, retries, extensions and unrelated prompts: at most ``KEEP_ONE`` entries, each ending one token before
    its prompt's end and holding its own ids, and every prefill resumes from the longest strict prefix."""

    engine, rec = _one_gpu(monkeypatch)
    rng = random.Random(seed)
    turns = [([rng.randrange(20, 90) for _ in range(rng.randrange(1, 5))], None)]
    prompt = _chat(turns)
    for _ in range(60):
        start, cache = _expected_cache(engine.cache, prompt)
        stats = _run(engine, prompt, max_tokens=rng.randrange(1, 5))
        assert rec.prefills[-1].start == start == stats["cached"]
        assert rec.prefills[-1].keep_at == entry_end(prompt)
        assert _entries(engine) == cache
        _check_entries(engine)
        roll = rng.random()
        if roll < 0.5:                                       # the next chat turn
            turns[-1] = (turns[-1][0], [rng.randrange(20, 90) for _ in range(rng.randrange(1, 4))])
            turns.append(([rng.randrange(20, 90) for _ in range(rng.randrange(1, 5))], None))
            prompt = _chat(turns)
        elif roll < 0.6:                                     # the same prompt again
            pass
        elif roll < 0.8:                                     # the reply appended as it was generated
            prompt = prompt + _reply(prompt)[:-1] + [rng.randrange(20, 90)]
        else:                                                # something else
            turns = [([rng.randrange(20, 90) for _ in range(rng.randrange(1, 5))], None)]
            prompt = _chat(turns)


class _Done(Exception):
    pass


def _two_ranks(monkeypatch):
    rec = Recorder()
    shares, rank1_caches = [], []
    engines = {}
    fake = types.ModuleType(PKG + ".decode_tp")

    def share(values, rank, device):
        if rank == 0:
            shares.append(list(values))
            return list(values)
        if len(shares) % 2 == 0:                             # rank 1 asks for the next header: the last request is done
            rank1_caches.append(_entries(engines[1]))
        if not shares:
            raise _Done
        return shares.pop(0)

    def prefill_tp(w, prompt, sampling, rank, drafter=None, *, state=None, keep_at=None, **stops):
        return rec.prefill(w, prompt, sampling, drafter, state=state, keep_at=keep_at, rank=rank)

    def decode_tp(w, st, prompt, pending, count, sampling, rank, drafter, *, max_rows, allow_copy=True,
                  on_tokens=None, inplace=False, stop_eos=True):
        assert inplace and all(st is not entry for _, entry, _ in engines[rank].cache.entries)
        result = rec.decode(st, prompt, pending, count)
        if on_tokens is not None:
            on_tokens(result.tokens[1:])
        return result

    def one_gpu_only(*args, **kwargs):
        raise AssertionError("a two-rank engine ran the one-GPU path")

    fake._share, fake.prefill_tp, fake.decode_tp = share, prefill_tp, decode_tp
    fake.pack_sampling, fake.unpack_sampling = (lambda sampling: [0] * 14), (lambda words: None)
    monkeypatch.setitem(sys.modules, PKG + ".decode_tp", fake)
    single = types.ModuleType(PKG + ".decode")              # generate imports it before choosing the path
    single.prefill = single.draft_decode = one_gpu_only
    monkeypatch.setitem(sys.modules, PKG + ".decode", single)
    for rank in (0, 1):
        engines[rank] = _bare_engine(tp=2, rank=rank)
    return engines, rec, rank1_caches


def test_both_ranks_keep_the_same_entries_and_resume_from_the_same_one(monkeypatch):
    engines, rec, rank1_caches = _two_ranks(monkeypatch)
    rng = random.Random(7)
    turns = [([40, 41, 42], None)]
    prompts = []
    for i in range(12):
        prompts.append(_chat(turns))
        if i % 4 == 3:
            prompts.append(list(prompts[-1]))                # a retry of the same prompt
        turns[-1] = (turns[-1][0], [rng.randrange(20, 90) for _ in range(2)])
        turns.append(([rng.randrange(20, 90) for _ in range(3)], None))
    rank0_caches = []
    for prompt in prompts:
        _run(engines[0], prompt)
        rank0_caches.append(_entries(engines[0]))
    rank0_prefills = list(rec.prefills)
    with pytest.raises(_Done):
        engines[1].follow()
    assert rank1_caches[1:] == rank0_caches                  # [0] is before the first request
    rank1_prefills = rec.prefills[len(rank0_prefills):]
    assert [(p.start, p.keep_at, p.prompt) for p in rank1_prefills] == \
        [(p.start, p.keep_at, p.prompt) for p in rank0_prefills]
    # each chat turn, and each retry, resumes one token before the previous prompt's end
    assert [p.start for p in rank0_prefills] == [0] + [len(prompts[i - 1]) - 1 for i in range(1, len(prompts))]
    assert all(p.keep_at == len(p.prompt) - 1 for p in rank0_prefills)


# ------------------------------------------------------------------------------------------ the PyTorch modules
@pytest.fixture
def cuda_modules(monkeypatch):
    """The 27B's CUDA modules, imported with an import-only Triton stand-in where Triton is missing; modules first
    imported here are dropped afterwards, as in ``test_cuda_geometry``."""

    torch = pytest.importorskip("torch")
    before = set(sys.modules)
    if importlib.util.find_spec("triton") is None:
        lang = ModuleType("triton.language")
        lang.constexpr = object
        triton = ModuleType("triton")
        triton.language = lang
        triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
        triton.cdiv = lambda a, b: (a + b - 1) // b
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setitem(sys.modules, "triton.language", lang)
    from tensorfold.families.qwen3_5.cuda import decode, decode_tp, forward, multi, prefill

    try:
        yield SimpleNamespace(torch=torch, decode=decode, decode_tp=decode_tp, forward=forward, multi=multi,
                              prefill=prefill)
    finally:
        added = [(name, module) for name, module in list(sys.modules.items())
                 if name not in before and name.startswith("tensorfold.")
                 and (".cuda" in name or name.startswith("tensorfold.cuda"))]
        for name, module in sorted(added, key=lambda item: len(item[0]), reverse=True):
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child, None) is module:
                delattr(parent, child)
            sys.modules.pop(name, None)


# ----------------------------------------------------------------------------- the concurrent decoder's admissions
def _ids_state(m, ids):
    """A committed state whose one attention layer's keys hold its ids."""

    st = object.__new__(m.forward.State)
    st.pos, st.limit, st.conv, st.rec = len(ids), 0, [None], [None]
    keys = m.torch.tensor(list(ids), dtype=m.torch.float32).view(-1, 1, 1)
    st.kv = [(keys, keys.clone())]
    return st


def _ids_of(st):
    return [int(x) for x in st.kv[0][0][:st.pos].reshape(-1).tolist()] if st.kv else []


CONTEXT = 1 << 16


def _snap(n):
    """A stand-in drafter snapshot at ``n`` (its per-layer lists, then its lengths)."""

    return ([("drafter at", n)], [], n, n)


def _stand_in_prefills(m, monkeypatch):
    """``prefill_state`` and ``first_token`` for ``multi``: states that hold their ids, one record per rank."""

    recs = {rank: SimpleNamespace(prefills=[], kept=[], last=None) for rank in (0, 1)}

    def prefill_state(w, prompt, st, *, tp=False, draft=None, keep_at=None, vision=None):
        rec, start = recs[w.rank], st.pos
        assert _ids_of(st) == list(prompt[:start]) and st.limit >= len(prompt)
        rec.prefills.append(SimpleNamespace(prompt=list(prompt), start=start, keep_at=keep_at))
        rec.last = list(prompt)
        st.pos, st.kv = len(prompt), _ids_state(m, prompt).kv       # the prompt committed into the stream's state
        if keep_at is None:
            return "normed"
        kept = _ids_state(m, prompt[:keep_at])
        rec.kept.append(kept)
        return "normed", (kept, _snap(keep_at))

    monkeypatch.setattr(m.multi, "prefill_state", prefill_state)
    monkeypatch.setattr(m.multi, "first_token", lambda w, normed, n, *rest: _reply(recs[w.rank].last)[0])
    return recs


def _decoder(m, *, rank=0, world=1):
    w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=8), norm=m.torch.zeros(1), head=SimpleNamespace(n=8),
                        layers=[], rank=rank)
    return m.multi.MultiDecoder(w, None, keep=KEEP, rank=rank, world=world, context=CONTEXT)


def _admitted(dec, s):
    """Admit ``s`` and run its prompt's prefill step (no stream decodes, so one step takes the whole prompt)."""

    dec.admit(s)
    dec._fill()


def _cache_ids(dec):
    return [ids for ids, _, _ in dec.cache.entries]


def _prompts(seed, count=40):
    rng = random.Random(seed)
    turns = [([rng.randrange(20, 90) for _ in range(3)], None)]
    out = []
    for _ in range(count):
        prompt = _chat(turns)
        out.append((prompt, rng.random() >= 0.15))            # (prompt, draft): some serial reference requests
        roll = rng.random()
        if roll < 0.6:
            turns[-1] = (turns[-1][0], [rng.randrange(20, 90) for _ in range(2)])
            turns.append(([rng.randrange(20, 90) for _ in range(rng.randrange(1, 4))], None))
        elif roll < 0.75:
            pass
        else:
            turns = [([rng.randrange(20, 90) for _ in range(3)], None)]
    return out


@pytest.mark.parametrize("seed", range(3))
def test_concurrent_admissions_keep_prompt_entries_one_token_short(cuda_modules, monkeypatch, seed):
    """Each drafting admission adds the private state at ``len(prompt) - 1`` (at most ``KEEP`` entries) and resumes
    from the longest entry its prompt strictly extends; a serial reference admission neither resumes nor adds."""

    m = cuda_modules
    rec = _stand_in_prefills(m, monkeypatch)[0]
    dec = _decoder(m)
    for prompt, draft in _prompts(seed):
        model = PrefixCache(KEEP)                         # the decoder's cache rule, on a copy of its entries
        model.entries, model.hit = [(ids, None, None) for ids in _cache_ids(dec)], set(dec.cache.hit)
        hit = model.longest(prompt) if draft else None
        if draft:
            model.add(prompt[:-1], None, None)
        s = m.multi.Stream(list(prompt), 3, None, draft=draft)
        _admitted(dec, s)
        assert s.cached == rec.prefills[-1].start == (len(hit[0]) if hit else 0)
        assert rec.prefills[-1].keep_at == (entry_end(prompt) if draft else None)
        assert _cache_ids(dec) == [ids for ids, _, _ in model.entries]
        for ids, st, snap in dec.cache.entries:
            assert _ids_of(st) == ids and st.pos == len(ids) and snap == _snap(len(ids))
        assert not s.st.kv or all(s.st.kv[0][0].data_ptr() != st.kv[0][0].data_ptr()      # the stream's own buffers
                                  for _, st, _ in dec.cache.entries)
        dec.finish([s])


def test_both_ranks_admit_from_and_keep_the_same_entries(cuda_modules, monkeypatch):
    m = cuda_modules
    queue = []

    def share(values, rank, device):
        if rank == 0:
            queue.append(list(values))
            return list(values)
        return queue.pop(0)

    monkeypatch.setattr(m.multi, "_share", share)
    rec0, rec1 = _stand_in_prefills(m, monkeypatch).values()
    dec0, dec1 = _decoder(m, rank=0, world=2), _decoder(m, rank=1, world=2)
    prompts = _prompts(11, count=24)
    caches0 = []
    for prompt, draft in prompts:
        s = m.multi.Stream(list(prompt), 3, None, draft=draft)
        _admitted(dec0, s)
        caches0.append(_cache_ids(dec0))
        dec0.finish([s])
    queue.append([])                                         # rank 0 ends the follow loop
    caches1, done = [], dec1._finish

    def finish_and_record(sid):
        caches1.append(_cache_ids(dec1))
        done(sid)

    dec1._finish = finish_and_record
    dec1.follow()
    assert caches1 == caches0
    assert [(p.start, p.keep_at, p.prompt) for p in rec1.prefills] == \
        [(p.start, p.keep_at, p.prompt) for p in rec0.prefills]
    assert any(p.start for p in rec1.prefills)


# ----------------------------------------------------------------- the concurrent decoder's drafter contexts
WINDOW = 16


class ContextDraft:
    """DFlash2's drafter contexts on its fused path: each layer's keys and values, ``[heads, rows, dim]`` tensors of at
    most ``window`` rows. ``add_taps`` concatenates and keeps the last ``window`` rows, ``snapshot`` and ``restore``
    copy the per-layer lists, ``skip`` empties them, ``add_taps_streams`` puts each stream's new tensors in its
    snapshot's own lists, and ``launch_blocks`` reads only the snapshots it is given (and proposes nothing). Every
    tensor put in a context is tracked."""

    layers, heads, dim = 2, 2, 4

    def __init__(self, torch, window, world=1):
        self.torch, self.window, self.world = torch, window, world
        self.kc, self.vc = [None] * self.layers, [None] * self.layers
        self.context_len, self.context_end = 0, 0
        self.made = []

    def _made(self, t):
        self.made.append(weakref.ref(t))
        return t

    def alive(self):
        return [t for t in (ref() for ref in self.made) if t is not None]

    def snapshot(self):
        return (list(self.kc), list(self.vc), self.context_len, self.context_end)

    def restore(self, snap):
        kc, vc, self.context_len, self.context_end = snap
        self.kc, self.vc = list(kc), list(vc)

    def skip(self, n):
        self.kc, self.vc = [None] * self.layers, [None] * self.layers
        self.context_len, self.context_end = 0, self.context_end + n

    def add_taps(self, taps):
        n = taps.shape[0]
        for cache in (self.kc, self.vc):
            for layer in range(self.layers):
                new, old = self.torch.zeros(self.heads, n, self.dim), cache[layer]
                joined = new if old is None else self.torch.cat((old, new), dim=1)
                cache[layer] = self._made(joined[:, -self.window:].contiguous())
        self.context_len = min(self.window, self.context_len + n)
        self.context_end += n

    def add_taps_streams(self, snaps, taps):
        sizes = [t.shape[0] for t in taps]
        for snap, n in zip(snaps, sizes):
            for cache in snap[:2]:
                for layer in range(self.layers):
                    rows = min(self.window, (0 if cache[layer] is None else cache[layer].shape[1]) + n)
                    cache[layer] = self._made(self.torch.zeros(self.heads, rows, self.dim))
        return [(snap[0], snap[1], min(self.window, snap[2] + n), snap[3] + n) for snap, n in zip(snaps, sizes)]

    def launch_blocks(self, snaps, pendings, max_nodes, block=None):
        assert all(t is not None for snap in snaps for t in snap[0] + snap[1])
        return [None] * len(snaps)


def _tapped_prefills(m, monkeypatch):
    """``MultiDecoder`` on the real ``prefill_state``, ``private``, ``kept`` and ``viewed``, over ``_ids_state`` states:
    a stand-in ``prefill_chunk`` writes each row's id into the attention buffers and returns one tap row a row; a
    stand-in first token."""

    def chunk(w, tokens, st, *, tp=False, capture_taps=False, last=True, every=False, cut=0, vision=None):
        p0, rows = st.pos, int(tokens.shape[0])
        assert 0 <= cut < rows
        kbuf, vbuf = m.prefill._grow(st, 0, p0 + rows)
        kbuf[p0:p0 + rows, 0, 0] = tokens.float()
        vbuf[p0:p0 + rows, 0, 0] = tokens.float()
        st.pos = p0 + rows
        normed = ("normed", st.pos) if last else None
        tapped = m.torch.zeros(rows, 1) if capture_taps else None
        if not cut:
            return normed, tapped
        part = m.decode.clone_state(st)
        part.pos = p0 + cut
        return normed, tapped, part

    monkeypatch.setattr(m.prefill, "prefill_chunk", chunk)
    monkeypatch.setattr(m.multi, "State", lambda w: _ids_state(m, []))
    monkeypatch.setattr(m.multi, "first_token", lambda w, normed, n, *rest: _reply([n])[0])


def _context_rows_in(tensors):
    """Drafter context rows in these tensors' storages, each storage once (a row: keys and values, every layer)."""

    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors if t is not None}
    return sum(storages.values()) // (2 * ContextDraft.layers * ContextDraft.heads * ContextDraft.dim * 4)


def _held(dec):
    """The context rows the cache entries and the decoding streams hold."""

    snaps = [snap for _, _, snap in dec.cache.entries] + [s.snap for s in dec.streams.values()]
    return _context_rows_in([t for snap in snaps if snap is not None for t in snap[0] + snap[1]])


def _rounds(m, monkeypatch):
    """A round's forward, sampling and commit as stand-ins: each window is its pending token alone, which samples 5
    (no end token), so a round commits one row a stream and gives the drafter that row's taps."""

    def forward(w, wins, *, full_logits, tp, capture_taps):
        starts = [0]
        for tokens, _, _ in wins:
            starts.append(starts[-1] + len(tokens))
        return None, None, m.torch.zeros(starts[-1], 1), starts

    monkeypatch.setattr(m.multi, "multi_tree_forward", forward)
    monkeypatch.setattr(m.multi, "sample_streams", lambda logits, starts, positions, samplings:
                        [[5] * len(p) for p in positions])
    monkeypatch.setattr(m.multi, "path_indices", lambda record, rows: [(None, None, m.torch.tensor(r)) for r in rows])
    monkeypatch.setattr(m.multi, "commit_streams", lambda states, record, rows, indices, in_place: None)


def _drafter_rows(m, monkeypatch, turns, *, whole_prompt, rank):
    """Serve each (prompt, reply tokens) once the one before has finished, drafting with a ``WINDOW``-row drafter; for
    each, (context rows allocated, rows the entries and decoding streams hold) when its decode starts and after its
    first round's commit. ``whole_prompt``: the entry at the whole prompt, whose snapshot is the stream's context, as
    before the entry moved to ``len(prompt) - 1``. ``rank`` None: one GPU; 0 or 1: that rank of two, rank 1 following
    rank 0's messages."""

    with monkeypatch.context() as mp:
        _tapped_prefills(m, mp)
        _rounds(m, mp)
        if whole_prompt:
            mp.setattr(m.multi, "entry_end", len)
        queue = []

        def share(values, r, device):
            if r == 0:
                queue.append(list(values))
                return list(values)
            return queue.pop(0)

        mp.setattr(m.multi, "_share", share)
        world = 1 if rank is None else 2
        w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=8), norm=m.torch.zeros(1),
                            head=SimpleNamespace(n=8), layers=[])
        drafts = [ContextDraft(m.torch, WINDOW, world) for _ in range(world)]
        decs = [m.multi.MultiDecoder(w, d, keep=KEEP, rank=r, world=world, context=CONTEXT, allow_copy=False)
                for r, d in enumerate(drafts)]
        dec, draft, seen = decs[rank or 0], drafts[rank or 0], []
        step, commit = dec._step, dec._commit

        def look():
            seen.append((_context_rows_in(draft.alive()), _held(dec)))

        def step_and_look(s, stop):
            first = step(s, stop)
            if s.sid in dec.streams:                          # the step that reached the prompt's end
                look()
            return first

        def commit_and_look(*args):
            commit(*args)
            look()

        dec._step, dec._commit = step_and_look, commit_and_look
        for prompt, count in turns:
            s = m.multi.Stream(list(prompt), count, None)
            _admitted(decs[0], s)
            assert not decs[0].filling and not s.done
            assert decs[0].round() == ([s] if count == 2 else [])
            decs[0].finish([s])
        if world == 2:
            queue.append([])                                  # rank 0 ends the follow loop
            decs[1].follow()
        assert len(seen) == 2 * len(turns)
        return [seen[i:i + 2] for i in range(0, len(seen), 2)]


@pytest.mark.parametrize("rank", [None, 0, 1], ids=["one GPU", "rank 0 of two", "rank 1 of two"])
def test_the_entrys_own_drafter_context_is_held_only_until_the_first_commit(cuda_modules, monkeypatch, rank):
    """The entry at ``n - 1`` holds a drafter context of its own; a whole-prompt entry shares its stream's. After a
    prefill step the drafter keeps no reference of its own, so every context allocated is an entry's or a decoding
    stream's. The stream's first commit replaces its context and frees the prompt-end one: from then on no more rows
    are allocated than with the whole-prompt entry. Before it, at most the entry's own context more."""

    m = cuda_modules
    first = _chat([([40, 41, 42], None)])                                     # 10 tokens: an entry below the window
    second = _chat([([40, 41, 42], [50, 51]), ([43, 44], None)])              # 21, resumed from first's entry here
    turns = [(first, 3), (second, 2), (second, 3), ([7], 3)]    # 2: the reply ends in its first round; KEEP entries
    here = _drafter_rows(m, monkeypatch, turns, whole_prompt=False, rank=rank)
    whole = _drafter_rows(m, monkeypatch, turns, whole_prompt=True, rank=rank)
    for (prompt, _), (start, after), (whole_start, whole_after) in zip(turns, here, whole):
        assert start[0] == start[1] and after[0] == after[1], (prompt, start, after)
        own = min(len(prompt) - 1, WINDOW) if len(prompt) > 1 else 0
        assert start[0] <= whole_start[0] + own, (prompt, start, whole_start)
        assert after[0] <= whole_after[0], (prompt, after, whole_after)
    assert any(start[0] > whole_start[0] for (start, _), (whole_start, _) in zip(here, whole))


@pytest.mark.parametrize("case", ["message start", "image"])
def test_without_a_prompt_end_entry_the_first_commit_frees_the_prompt_end_context(cuda_modules, monkeypatch, case):
    """Where no entry is kept at the prompt's end (a message start less than ``MIN_GAP`` before it covers it, or the
    prompt has images), after the stream's first commit every context allocated is an entry's or the stream's."""

    m = cuda_modules
    _tapped_prefills(m, monkeypatch)
    _rounds(m, monkeypatch)
    monkeypatch.setattr(m.multi, "MIN_GAP", 3)
    image = case == "image"
    w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=8), norm=m.torch.zeros(1), head=SimpleNamespace(n=8),
                        layers=[])
    draft = ContextDraft(m.torch, WINDOW)
    vision = SimpleNamespace(encode=lambda prepared, prompt: SimpleNamespace(rope_delta=0)) if image else None
    dec = m.multi.MultiDecoder(w, draft, keep=KEEP, context=CONTEXT, allow_copy=False,
                               points=lambda ids: [4] if len(ids) > 4 else [], vision=vision)
    prompt = list(range(60, 66))
    s = m.multi.Stream(prompt, 3, None, vision=("image",) if image else None)
    dec.admit(s)
    while dec.filling:
        dec._fill()
    assert _cache_ids(dec) == ([] if image else [prompt[:4]])
    assert dec.round() == [] and not s.done
    assert _context_rows_in(draft.alive()) == _held(dec)


# ------------------------------------------------------------------ prefill_state(keep_at=...), a stand-in chunk
class TapDraft:
    """DFlash2's context bookkeeping: the last ``window`` rows it absorbed, ``skip``, snapshots and restores."""

    def __init__(self, window, start):
        self.window, self.rows, self.end, self.calls = window, [], start, []

    def skip(self, n):
        self.rows, self.end = [], self.end + n

    def add_taps(self, taps):
        got = [int(r) for r in taps[:, 0].tolist()]
        assert got and got == list(range(self.end, self.end + len(got))), "taps out of order"
        self.calls.append(got)
        self.rows, self.end = (self.rows + got)[-self.window:], self.end + len(got)

    def snapshot(self):
        return tuple(self.rows), self.end

    def restore(self, snap):
        self.rows, self.end = list(snap[0]), snap[1]


def _fake_chunks(m, monkeypatch):
    calls = []

    def fake_chunk(w, tokens, st, *, tp=False, capture_taps=False, last=True, cut=0, vision=None):
        rows, p0 = int(tokens.shape[0]), st.pos
        assert 0 <= cut < rows
        calls.append((p0, rows, cut, capture_taps, last))
        st.pos = p0 + rows
        st.rec[0] = ("rec at", st.pos)
        st.kv[1] = ("buffers after", st.pos)                  # a commit may replace (grow) the buffers
        normed = ("normed", st.pos) if last else None
        taps = (p0 + m.torch.arange(rows, dtype=m.torch.float32))[:, None] if capture_taps else None
        if not cut:
            return normed, taps
        part = m.decode.clone_state(st)
        part.pos, part.rec = p0 + cut, [("rec at", p0 + cut), None]
        return normed, taps, part

    monkeypatch.setattr(m.prefill, "prefill_chunk", fake_chunk)
    return calls


def _start_state(m, base):
    st = object.__new__(m.forward.State)
    st.pos, st.limit, st.conv, st.rec, st.kv = base, 0, ["conv", None], [("rec at", base), None], [None, ("start",)]
    return st


CASES = [(0, 1, 1), (0, 2, 1), (0, 300, 299), (0, 300, 0), (0, 300, 300), (0, 300, 150), (0, 300, 64), (0, 300, 65),
         (0, 300, 63), (0, 129, 128), (5, 133, 132), (100, 101, 100), (100, 300, 299), (100, 300, 100), (64, 200, 128)]


@pytest.mark.parametrize("base,n,keep_at", CASES)
@pytest.mark.parametrize("size", [64, 4096])
@pytest.mark.parametrize("window", [None, 16, 1000])
def test_keep_at_cuts_only_the_chunk_that_holds_the_point(cuda_modules, monkeypatch, base, n, keep_at, size, window):
    """The spans are those of a prefill without ``keep_at``; only the chunk strictly holding the point is cut, there.
    The kept state stops at the point on the prompt state's buffers. The drafter sees every tapped row once, in
    order, ends where it ends without ``keep_at``, and the kept snapshot is the one a prefill of the prefix leaves."""

    m = cuda_modules
    calls = _fake_chunks(m, monkeypatch)
    prompt = list(range(n))

    def run(keep, until=n):
        calls.clear()
        draft = TapDraft(window, base) if window else None
        st = _start_state(m, base)
        out = m.prefill.prefill_state(SimpleNamespace(norm=m.torch.zeros(1)), prompt[:until], st, draft=draft,
                                      size=size, keep_at=keep)
        return out, st, draft, list(calls)

    ref, ref_st, ref_draft, ref_calls = run(None)
    (normed, (kept, snap)), st, draft, got = run(keep_at)
    spans = m.prefill.chunks(base, n, size)
    assert [(p0, rows) for p0, rows, *_ in got] == [(a, b - a) for a, b in spans] == [(p0, r) for p0, r, *_ in ref_calls]
    assert [cut for _, _, cut, _, _ in got] == [keep_at - a if a < keep_at < b else 0 for a, b in spans]
    assert normed == ref == ("normed", n) and st.pos == n
    assert kept.pos == keep_at and kept.kv == st.kv and kept.kv is not st.kv
    assert kept.rec[0] == ("rec at", keep_at)
    if draft is None:
        assert snap is None
        return
    assert (draft.rows, draft.end) == (ref_draft.rows, ref_draft.end)
    if keep_at > base:
        prefix = run(None, keep_at)[2]
        assert snap == prefix.snapshot()
    else:
        assert snap == ((), base)


@pytest.mark.parametrize("keep_at", [4, 11, -1])
def test_a_point_outside_the_prefilled_range_is_refused_before_any_work(cuda_modules, monkeypatch, keep_at):
    m = cuda_modules
    calls = _fake_chunks(m, monkeypatch)
    draft = TapDraft(2, 5)
    with pytest.raises(ValueError, match="keep_at"):
        m.prefill.prefill_state(SimpleNamespace(norm=m.torch.zeros(1)), list(range(10)), _start_state(m, 5),
                                draft=draft, keep_at=keep_at)
    assert calls == [] and draft.end == 5


# ------------------------------------------------------------------ the whole prefill on row-wise CPU stand-ins
# Every stand-in works one row at a time and the chain steps row by row on fp32 state, as the CUDA kernels do.
VOCAB = 512


def _cpu_kernels(torch, monkeypatch, prefill, forward):
    F = torch.nn.functional
    chains = []

    def rows(fn, *xs):
        return torch.stack([fn(*(x[r] for x in xs)) for r in range(xs[0].shape[0])])

    def rmsnorm(a, w, eps):
        return (a.float() * torch.rsqrt(a.float().pow(2).mean() + eps) * w.float()).bfloat16()

    def pg_add_rmsnorm(x, r, w, eps):
        h = x if r is None else rows(lambda a, b: (a.float() + b.float()).bfloat16(), x, r)
        return h, rows(lambda a: rmsnorm(a, w, eps), h)

    def glue_add_rmsnorm(x, r, w, eps):
        h, y = pg_add_rmsnorm(x, r, w, eps)
        return h, y, None

    def gdn_pre(qkv, conv_state, conv_w, windows, a, b, A_log, dt_bias, *, kh, vh, dk):
        inp = torch.cat([conv_state, qkv]).float()
        conv = rows(lambda win: F.silu((inp[win.long()] * conv_w.float().t()).sum(0)), windows)
        W = qkv.shape[0]
        q = rows(lambda x: F.normalize(x[:kh * dk].reshape(kh, dk), dim=-1).bfloat16(), conv)
        k = rows(lambda x: F.normalize(x[kh * dk:2 * kh * dk].reshape(kh, dk), dim=-1).bfloat16(), conv)
        v = conv[:, 2 * kh * dk:].reshape(W, vh, dk).bfloat16()
        g = rows(lambda x: torch.exp(-torch.exp(A_log) * F.softplus(x.float() + dt_bias)), a)
        beta = rows(lambda x: torch.sigmoid(x.float()), b)
        return q, k, v, g, beta

    def chain(q, k, v, g, beta, state, final):
        chains.append(q.shape[0])
        ratio = v.shape[1] // q.shape[1]
        s, ys = state.clone(), []
        for r in range(q.shape[0]):
            kk = k[r].float().repeat_interleave(ratio, 0)[:, None, :]
            s = s * g[r][:, None, None]
            delta = (v[r].float() - (s * kk).sum(-1)) * beta[r][:, None]
            s = s + kk * delta[..., None]
            ys.append((s * q[r].float().repeat_interleave(ratio, 0)[:, None, :]).sum(-1).bfloat16())
        final.copy_(s)
        return torch.stack(ys)

    def gated_norm(y, z, w, eps):
        return rows(lambda a, b: (a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()
                                  * F.silu(b.float())).reshape(-1).bfloat16(), y, z)

    def attn_prep(qg, key, q_norm, k_norm, pos, inv_freq, eps, *, heads, kv_heads, head_dim, mrope_section=None):
        q = rows(lambda a, p: (a.float().reshape(heads, 2 * head_dim)[:, :head_dim] * q_norm.float()
                               + 1e-3 * p.float()).bfloat16(), qg, pos)
        k = rows(lambda a, p: (a.float().reshape(kv_heads, head_dim) * k_norm.float() + 1e-3 * p.float()).bfloat16(),
                 key, pos)
        return q, k

    def attention(q, kbuf, vbuf, p0, *, scale):
        out = []
        for r in range(q.shape[0]):
            keys, values = kbuf[:p0 + r + 1].float(), vbuf[:p0 + r + 1].float()
            group = q.shape[1] // keys.shape[1]
            out.append(torch.stack([F.softmax(keys[:, h // group] @ q[r, h].float() * scale, 0) @ values[:, h // group]
                                    for h in range(q.shape[1])]).bfloat16())
        return torch.stack(out)

    def gate_mul(o, qg, *, heads, head_dim):
        return rows(lambda a, b: (a.float() * torch.sigmoid(b.float().reshape(heads, 2 * head_dim)[:, head_dim:]))
                    .reshape(-1).bfloat16(), o, qg)

    def swiglu(gate, up):
        return rows(lambda a, b: (F.silu(a.float()) * b.float()).bfloat16(), gate, up)

    def matmul(x, q, *args, **kwargs):
        return rows(lambda a: (q.weight @ a.float()).bfloat16(), x)

    glue = SimpleNamespace(embedding=lambda ids, q: q.weight[ids.long()].bfloat16(),
                           add_rmsnorm=glue_add_rmsnorm, gdn_pre=gdn_pre, attn_prep=attn_prep)
    pg = SimpleNamespace(add_rmsnorm=pg_add_rmsnorm, gated_norm=gated_norm, gate_mul=gate_mul, swiglu=swiglu)
    monkeypatch.setattr(prefill, "glue", glue)
    monkeypatch.setattr(prefill, "prefill_glue", pg)         # the MLX checkpoint's prompt glue (w.quant "mlx")
    monkeypatch.setattr(prefill, "_mm", matmul)
    monkeypatch.setattr(prefill, "deltanet", SimpleNamespace(chain=chain))
    monkeypatch.setattr(prefill, "attention", attention)
    monkeypatch.setattr(forward, "_mm", matmul)
    return chains


def _cpu_model(torch, layers=4):
    from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights

    gen = torch.Generator().manual_seed(3)
    c = Config(hidden=128, intermediate=128, layers=layers, heads=2, kv_heads=1, head_dim=64, vocab=VOCAB, k_heads=1,
               v_heads=2, dk=64, dv=64, conv_kernel=4, interval=4, eps=1e-6, rope_dims=16, rope_theta=1e6, eos=(0,))

    def lin(n, k):                                          # bf16 scale stand-ins: the 4-bit g64 (FP8 prompt) path
        return QLinear(torch.randn(n, k, generator=gen) * k ** -0.5, torch.empty(0, dtype=torch.bfloat16),
                       torch.empty(0, dtype=torch.bfloat16))

    def norm(d):
        return (1 + 0.1 * torch.randn(d, generator=gen)).bfloat16()

    out = []
    for i in range(c.layers):
        gdn = attn = None
        if c.is_linear(i):
            cd = 2 * c.k_heads * c.dk + c.v_heads * c.dv
            gdn = GDN(lin(cd, c.hidden), lin(c.v_heads * c.dv, c.hidden), lin(c.v_heads, c.hidden),
                      lin(c.v_heads, c.hidden), lin(c.hidden, c.v_heads * c.dv),
                      (0.5 * torch.randn(cd, c.conv_kernel, generator=gen)).bfloat16(),
                      0.5 * torch.randn(c.v_heads, generator=gen), 0.5 * torch.randn(c.v_heads, generator=gen),
                      norm(c.dv))
        else:
            attn = Attention(lin(2 * c.heads * c.head_dim, c.hidden), lin(c.kv_heads * c.head_dim, c.hidden),
                             lin(c.kv_heads * c.head_dim, c.hidden), lin(c.hidden, c.heads * c.head_dim),
                             norm(c.head_dim), norm(c.head_dim))
        out.append(Layer(c.is_linear(i), norm(c.hidden), norm(c.hidden), gdn, attn, lin(c.intermediate, c.hidden),
                         lin(c.intermediate, c.hidden), lin(c.hidden, c.intermediate)))
    embed = QLinear(torch.randn(c.vocab, c.hidden, generator=gen), torch.empty(0, dtype=torch.bfloat16),
                    torch.empty(0, dtype=torch.bfloat16))
    return Weights(c, embed, out, norm(c.hidden), lin(c.vocab, c.hidden), torch.ones(c.rope_dims // 2))


def _bits_equal(torch, a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def _assert_same_state(torch, a, b):
    assert a.pos == b.pos
    for i in range(len(a.rec)):
        if a.rec[i] is None:
            (ka, va), (kb, vb) = a.kv[i], b.kv[i]
            assert _bits_equal(torch, ka[:a.pos], kb[:b.pos]) and _bits_equal(torch, va[:a.pos], vb[:b.pos]), i
        else:
            assert _bits_equal(torch, a.rec[i], b.rec[i]), f"layer {i} GDN state"
            assert _bits_equal(torch, a.conv[i], b.conv[i]), f"layer {i} conv rows"


def _shares_kv(kept, st):
    """The kept state holds the returned state's key/value buffers, not buffers of its own."""

    return all(kv is None or kv is st.kv[i] for i, kv in enumerate(kept.kv))


@pytest.fixture(params=[4096, 64], ids=["one-span", "spans-of-64"])
def cpu(cuda_modules, monkeypatch, request):
    m = cuda_modules
    torch = m.torch
    chains = _cpu_kernels(torch, monkeypatch, m.prefill, m.forward)
    w = _cpu_model(torch)
    seen = []

    def argmax(logits, positions, sampling):
        seen.append(logits)
        return [int(x) for x in logits.argmax(-1).tolist()]

    monkeypatch.setattr(m.decode, "sample_rows", argmax)
    size, chunks = request.param, m.prefill.chunks
    monkeypatch.setattr(m.prefill, "chunks", lambda start, end, _=None: chunks(start, end, size))

    def run(prompt, **kw):
        seen.clear()
        out = m.decode.prefill(w, prompt, None, **kw)
        (logits,) = seen
        return out, logits

    return SimpleNamespace(torch=torch, m=m, w=w, run=run, chains=chains, size=size)


def _prompt(n, seed):
    gen = random.Random(seed)
    return [gen.randrange(1, VOCAB) for _ in range(n)]


@pytest.mark.parametrize("cached", [0, 5, 37])
@pytest.mark.parametrize("cut", [1, 2, 3, 4, 17, 38, 39])
def test_a_cut_chunk_commits_as_one_and_returns_the_state_at_the_cut(cpu, cached, cut):
    torch, m, w = cpu.torch, cpu.m, cpu.w
    ids = torch.tensor(_prompt(cached + 40, 21), dtype=torch.int32)

    def start():
        st = m.forward.State(w)
        if cached:
            m.prefill.prefill_chunk(w, ids[:cached], st)
        return st

    one, two, prefix = start(), start(), start()
    h_one, _ = m.prefill.prefill_chunk(w, ids[cached:], one)
    cpu.chains.clear()
    h_two, _, part = m.prefill.prefill_chunk(w, ids[cached:], two, cut=cut)
    assert cpu.chains == [cut, 40 - cut] * 3                  # two launches in each of the three GDN layers
    assert _bits_equal(torch, h_one, h_two)
    _assert_same_state(torch, two, one)
    m.prefill.prefill_chunk(w, ids[cached:cached + cut], prefix)
    assert part.pos == cached + cut and _shares_kv(part, two)
    _assert_same_state(torch, part, prefix)
    for bad in (-1, 40):
        with pytest.raises(ValueError, match="cut"):
            m.prefill.prefill_chunk(w, ids[cached:], start(), cut=bad)


@pytest.mark.parametrize("length,cached", [(n, c) for n in (1, 2, 3, 4, 63, 64, 65, 130) for c in (0, 1, 64) if c < n])
def test_prefill_keeping_a_point_equals_one_without_and_keeps_the_prefix_state(cpu, length, cached):
    torch = cpu.torch
    prompt = _prompt(length, length)

    def prefix():
        """A prefix of its own for each run: states resumed from one prefix share its key/value buffers."""
        return cpu.run(prompt[:cached])[0][0] if cached else None

    (ref, ref_pending), ref_logits = cpu.run(prompt, state=prefix())
    points = sorted({cached, cached + 1, length - 1, length, 64, 65, (cached + length) // 2} &
                    set(range(cached, length + 1)))
    for point in points:
        (st, pending, (kept, snap)), logits = cpu.run(prompt, state=prefix(), keep_at=point)
        assert all(kv is None or kv[0] is not ref.kv[i][0] for i, kv in enumerate(st.kv))
        assert _bits_equal(torch, logits, ref_logits) and pending == ref_pending
        _assert_same_state(torch, st, ref)
        assert kept.pos == point and snap is None and _shares_kv(kept, st)
        if point:
            (fresh, _), _ = cpu.run(prompt[:point])
            _assert_same_state(torch, kept, fresh)


@pytest.mark.parametrize("base", [0, 1, 2, 61, 62, 63, 64, 100, 126])
def test_the_next_turn_resumed_from_the_kept_state_equals_a_fresh_prefill(cpu, base):
    """``...<think>\\n`` then ``...<think>\\n\\n</think>\\n\\n...``: the state kept at n - 1 resumes the second prompt,
    which then ends in the state, logits and first token of a fresh prefill."""

    torch = cpu.torch
    head = _prompt(base, 40 + base)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(9, 41) + [THINK, NL]
    (_, _, (kept, _)), _ = cpu.run(first, keep_at=entry_end(first))
    (fresh, fresh_pending), fresh_logits = cpu.run(second)
    (st, pending), logits = cpu.run(second, state=kept)
    assert kept.pos == len(first) - 1
    assert _bits_equal(torch, logits, fresh_logits) and pending == fresh_pending
    _assert_same_state(torch, st, fresh)


def test_the_same_prompt_again_resumes_from_its_kept_state_with_one_token(cpu):
    torch = cpu.torch
    prompt = _prompt(70, 5)
    (_, _, (kept, _)), _ = cpu.run(prompt, keep_at=entry_end(prompt))
    (fresh, fresh_pending), fresh_logits = cpu.run(prompt)
    (st, pending, (again, _)), logits = cpu.run(prompt, state=kept, keep_at=entry_end(prompt))
    assert _bits_equal(torch, logits, fresh_logits) and pending == fresh_pending
    _assert_same_state(torch, st, fresh)
    # ``again`` is ``kept`` resumed with nothing prefilled, so it holds kept's tensors: compare it with a fresh prefix
    (fresh_prefix, _), _ = cpu.run(prompt[:entry_end(prompt)])
    _assert_same_state(torch, again, fresh_prefix)


@pytest.mark.parametrize("cached,length,point", [(0, 1025, 1024), (1000, 1025, 1024), (1000, 1030, 1024)])
def test_the_kept_state_holds_the_buffers_grown_after_the_point(cpu, cached, length, point):
    """Key/value buffers grow to at least 1,024 rows and double. A prefill resumed at 1,000 tokens grows them past
    1,024 inside the chunk that holds the point: the kept state then holds the grown buffers, whose first rows are
    the old ones' copies, and no buffer of its own."""

    torch = cpu.torch
    prompt = _prompt(length, 50)
    # two separately built prefixes: states resumed from one prefix share its key/value buffers
    prefix_a, prefix_b = (cpu.run(prompt[:cached])[0][0] for _ in range(2)) if cached else (None, None)
    (ref, ref_pending), ref_logits = cpu.run(prompt, state=prefix_a)
    (st, pending, (kept, _)), logits = cpu.run(prompt, state=prefix_b, keep_at=point)
    assert _bits_equal(torch, logits, ref_logits) and pending == ref_pending
    _assert_same_state(torch, st, ref)
    if cached:
        assert prefix_b.kv[3][0].shape[0] == 1024 < st.kv[3][0].shape[0]    # grown inside the chunk
    assert _shares_kv(kept, st)
    (fresh, _), _ = cpu.run(prompt[:point])
    _assert_same_state(torch, kept, fresh)
    other = prompt[:point] + [NL2] + _prompt(3, 51)
    (again, again_pending), again_logits = cpu.run(other, state=kept)
    (fresh_other, fresh_pending), fresh_logits = cpu.run(other)
    assert _bits_equal(torch, again_logits, fresh_logits) and again_pending == fresh_pending
    _assert_same_state(torch, again, fresh_other)


@pytest.mark.parametrize("where", ["prompt end", "last span start"])
def test_keeping_a_point_holds_no_buffer_that_a_grow_replaces(cpu, monkeypatch, where):
    """A fresh prefill frees each key/value buffer as soon as it grows it (the last span of 1,030 tokens grows them
    past 1,024 rows). The state kept at the point takes the final buffers after the chunk and never holds the ones a
    grow replaced, so keeping a point holds no extra buffers while the prefill runs."""

    m = cpu.m
    prompt = _prompt(1030, 52)
    point = len(prompt) - 1 if where == "prompt end" else m.prefill.chunks(0, len(prompt))[-1][0]
    grow, attention, replaced, alive = m.prefill._grow, m.prefill.attention, [], []

    def tracked_grow(st, i, need):
        before = st.kv[i]
        out = grow(st, i, need)
        if out is not before:
            replaced.append(weakref.ref(before[0]))
        return out

    def checked_attention(q, kbuf, vbuf, p0, *, scale):
        alive.append(sum(ref() is not None for ref in replaced))
        return attention(q, kbuf, vbuf, p0, scale=scale)

    monkeypatch.setattr(m.prefill, "_grow", tracked_grow)
    monkeypatch.setattr(m.prefill, "attention", checked_attention)
    (st, _, (kept, _)), _ = cpu.run(prompt, keep_at=point)
    assert replaced and alive and not any(alive), alive
    assert kept.pos == point and _shares_kv(kept, st)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("length,cached", [(1, 0), (3, 0), (65, 0), (130, 1), (130, 64)])
def test_the_two_rank_prefill_keeps_the_point_as_prefill_does(cpu, monkeypatch, rank, length, cached):
    """``prefill_tp`` on one process holding every weight (its rank sum and its share are the identity): with
    ``keep_at`` it returns ``prefill``'s prompt state, first token and kept state; without, two items as before."""

    torch, m = cpu.torch, cpu.m
    from tensorfold.families.qwen3_5.cuda import distributed

    prompt = _prompt(length, 70 + length)

    def prefix():
        return cpu.run(prompt[:cached])[0][0] if cached else None

    (_, ref_pending), _ = cpu.run(prompt, state=prefix())
    monkeypatch.setattr(distributed, "gather_rank_partials", lambda local: local)
    monkeypatch.setattr(m.decode_tp, "sample_rows", m.decode.sample_rows)             # the fixture's argmax
    monkeypatch.setattr(m.decode_tp, "_share",
                        lambda values, r, device: list(values) if r == 0 else [ref_pending])
    st, pending = m.decode_tp.prefill_tp(cpu.w, prompt, None, rank, state=prefix())
    assert pending == ref_pending
    for point in sorted({cached, cached + 1, length - 1, length, 64, 65} & set(range(max(cached, 1), length + 1))):
        (ref, _, (ref_kept, _)), _ = cpu.run(prompt, state=prefix(), keep_at=point)
        st, pending, (kept, snap) = m.decode_tp.prefill_tp(cpu.w, prompt, None, rank, state=prefix(), keep_at=point)
        assert all(kv is None or kv[0] is not ref.kv[i][0] for i, kv in enumerate(st.kv))
        assert pending == ref_pending and kept.pos == point and snap is None and _shares_kv(kept, st)
        _assert_same_state(torch, st, ref)
        _assert_same_state(torch, kept, ref_kept)
