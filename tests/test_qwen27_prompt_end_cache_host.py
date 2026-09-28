"""Host checks for the 27B CUDA engine's prompt-end cache entry, kept one token before the prompt's end (no GPU).

A chat prompt ends in the generation prompt, ``<think>`` and a newline (Qwen token 198); a next request that sends
that turn back without its reasoning renders an empty reasoning block, ``<think>`` and two newlines (token 271). An
entry for the whole prompt then differs from the next prompt in its last token and does not match; the entry at
``len(prompt) - 1`` does.

- The one-stream engine's bookkeeping, on one GPU and on both ranks of two, with stand-ins for ``decode`` and
  ``decode_tp`` (no PyTorch needed): which entry each prompt resumes from and what each entry holds, beside the
  states kept at message starts.
- The concurrent decoder's admissions (``MultiDecoder``), one GPU and both ranks, on ``prefill_state`` with a
  stand-in chunk: the entries it keeps and that no stream decodes into them.
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
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.cuda.markers import MIN_GAP
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

    def prefill(self, w, prompt, sampling, drafter=None, *, state=None, limit=0, stops=(), keep=None, keep_at=None,
                rank=None):
        start = state.pos if state is not None else 0
        if state is not None:
            assert list(prompt[:start]) == state.ids, "resumed from a state that is not a prefix of the prompt"
        assert start < len(prompt)
        for p in stops:                                     # as decode.prefill_stops: a state after each stop
            if start < p < len(prompt) and keep is not None:
                keep(p, FakeState(prompt[:p]), ("drafter at", p) if drafter is not None else None)
        self.prefills.append(SimpleNamespace(prompt=list(prompt), start=start, keep_at=keep_at, drafter=drafter,
                                             rank=rank, stops=list(stops)))
        pending = _reply(prompt)[0]
        if keep_at is None:
            return FakeState(prompt), pending
        assert max([start, *stops]) <= keep_at <= len(prompt)
        snap = ("drafter at", keep_at) if drafter is not None else None
        return FakeState(prompt), pending, (FakeState(prompt[:keep_at]), snap)

    @staticmethod
    def decode(st, prompt, pending, max_tokens):
        assert st.ids == list(prompt)
        tokens = ([pending] + _reply(prompt)[1:])[:max(1, max_tokens)]
        return SimpleNamespace(tokens=tokens, seconds=0.1, rounds=len(tokens), widths=[1] * len(tokens))


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

    def draft_decode(w, st, prompt, pending, count, sampling, draft, *, max_rows, allow_copy, on_tokens, inplace):
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


def test_a_message_start_well_before_the_end_keeps_both_states(monkeypatch):
    """With message starts (``markers.resume_points``), the prefill keeps a state at each start ``MIN_GAP`` or more
    past the resumed prefix; the prompt's entry is kept at ``entry_end`` as well when the last start is ``MIN_GAP`` or
    more before the prompt's end."""

    engine, rec = _one_gpu(monkeypatch)
    engine.points = lambda ids: [MIN_GAP] if len(ids) > MIN_GAP else []
    prompt = list(range(1000, 1000 + 2 * MIN_GAP + 10))
    _run(engine, prompt)
    assert rec.prefills[-1].stops == [MIN_GAP] and rec.prefills[-1].keep_at == len(prompt) - 1
    assert _entries(engine) == [prompt[:MIN_GAP], prompt[:-1]]
    _check_entries(engine)
    follow = prompt[:-1] + [NL2] + list(range(2000, 2010))      # resumes at the entry; no start past it
    stats = _run(engine, follow)
    assert stats["cached"] == len(prompt) - 1 and rec.prefills[-1].stops == []
    assert _entries(engine) == [prompt[:MIN_GAP], prompt[:-1], follow[:-1]]


def test_a_message_start_just_before_the_end_keeps_no_prompt_entry(monkeypatch):
    """As without this change: a kept message start less than ``MIN_GAP`` before the prompt's end covers it (the
    last assistant start, ``<|im_start|>assistant``, sits a few tokens before it), and no entry is kept at the end."""

    engine, rec = _one_gpu(monkeypatch)
    engine.points = lambda ids: [len(ids) - 5] if len(ids) - 5 >= MIN_GAP else []
    prompt = list(range(1000, 1000 + MIN_GAP + 5))
    _run(engine, prompt)
    assert rec.prefills[-1].stops == [MIN_GAP] and rec.prefills[-1].keep_at is None
    assert _entries(engine) == [prompt[:MIN_GAP]]


def _expected_cache(shadow, prompt):
    """The engine's rule, restated on a ``PrefixCache`` of ids: resume from the longest strict prefix, drop the
    entries that extend it, add the entry at ``entry_end``."""

    hit = shadow.longest(prompt)
    if hit is not None:
        n = len(hit[0])
        shadow.entries = [e for e in shadow.entries if len(e[0]) <= n or e[0][:n] != hit[0]]
    shadow.add(prompt[:entry_end(prompt)], None, None)
    return (len(hit[0]) if hit else 0), [ids for ids, _, _ in shadow.entries]


@pytest.mark.parametrize("seed", range(4))
def test_entries_stay_within_keep_one_and_each_resume_takes_the_longest_match(monkeypatch, seed):
    """Chat turns, retries, extensions and unrelated prompts: at most ``KEEP_ONE`` entries, each ending one token
    before its prompt's end and holding its own ids, and every prefill resumes from the longest strict prefix."""

    engine, rec = _one_gpu(monkeypatch)
    shadow = PrefixCache(KEEP_ONE)
    rng = random.Random(seed)
    turns = [([rng.randrange(20, 90) for _ in range(rng.randrange(1, 5))], None)]
    prompt = _chat(turns)
    for _ in range(60):
        start, cache = _expected_cache(shadow, prompt)
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
        return rec.prefill(w, prompt, sampling, drafter, state=state, keep_at=keep_at, rank=rank, **stops)

    def decode_tp(w, st, prompt, pending, count, sampling, rank, drafter, *, max_rows, allow_copy=True,
                  on_tokens=None, inplace=False):
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
def _gdn(m, pos):
    return m.torch.full((1, 1, 1), float(pos))


def _conv(m, pos):
    return m.torch.full((3, 1), float(pos))


def _ids_state(m, ids):
    """A committed state: a GDN layer whose state and conv rows name the tokens committed, and an attention layer
    whose keys and values hold them (float32, so the ids are exact)."""

    st = object.__new__(m.forward.State)
    st.pos, st.limit = len(ids), 0
    keys = m.torch.tensor([float(t) for t in ids], dtype=m.torch.float32).view(-1, 1, 1)
    st.kv, st.rec, st.conv = [None, (keys, keys.clone())], [_gdn(m, len(ids)), None], [_conv(m, len(ids)), None]
    return st


def _ids_of(st):
    return [int(x) for x in st.kv[1][0][:st.pos].reshape(-1).tolist()]


CONTEXT = 1 << 16


def _concurrent(m, monkeypatch, size=4096):
    """``MultiDecoder`` on the real ``prefill_state``, ``private`` and ``kept``, over ``_ids_state`` states: a stand-in
    ``prefill_chunk`` writes each row's id into the attention buffers and replaces (never writes through) the GDN
    state and conv rows, as the real chunk does; spans of ``size`` rows; a stand-in first token. Returns the GDN
    states the chunk made."""

    made = []

    def chunk(w, tokens, st, *, tp=False, capture_taps=False, last=True, every=False, cut=0):
        p0, rows = st.pos, int(tokens.shape[0])
        assert 0 <= cut < rows and not capture_taps
        kbuf, vbuf = m.prefill._grow(st, 1, p0 + rows)
        kbuf[p0:p0 + rows, 0, 0] = tokens.float()
        vbuf[p0:p0 + rows, 0, 0] = tokens.float()
        st.pos = p0 + rows
        st.rec[0], st.conv[0] = _gdn(m, st.pos), _conv(m, st.pos)
        made.append(st.rec[0])
        normed = ("normed", st.pos) if last else None
        if not cut:
            return normed, None
        part = m.decode.clone_state(st)
        part.pos, part.rec[0], part.conv[0] = p0 + cut, _gdn(m, p0 + cut), _conv(m, p0 + cut)
        made.append(part.rec[0])
        return normed, None, part

    monkeypatch.setattr(m.prefill, "prefill_chunk", chunk)
    chunks = m.prefill.chunks
    monkeypatch.setattr(m.prefill, "chunks", lambda start, end, _=None: chunks(start, end, size))
    monkeypatch.setattr(m.multi, "State", lambda w: _ids_state(m, []))
    monkeypatch.setattr(m.multi, "first_token", lambda w, normed, n, *args: _reply([n])[0])
    return made


def _decoder(m, *, rank=0, world=1, points=None):
    w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=8), norm=m.torch.zeros(1), head=SimpleNamespace(n=8),
                        layers=[])
    return m.multi.MultiDecoder(w, None, keep=KEEP, rank=rank, world=world, context=CONTEXT, points=points)


def _serve(dec, s):
    """Admit ``s`` and run its prefill steps (no round decodes)."""

    dec.admit(s)
    while any(x is s for x in dec.filling):
        dec._fill()


def _cache_ids(dec):
    return [ids for ids, _, _ in dec.cache.entries]


def _decode_in_place(s):
    """What a stream's rounds do to its GDN state and conv rows (``commit_streams(..., in_place=True)``)."""

    s.st.rec[0].fill_(-1.0)
    s.st.conv[0].fill_(-1.0)


def _check_cache(dec):
    """Each entry holds its ids' rows, and a GDN state and conv rows for its own length."""

    for ids, st, snap in dec.cache.entries:
        assert st.pos == len(ids) and _ids_of(st) == ids and snap is None, ids
        assert st.rec[0].item() == len(ids) and bool((st.conv[0] == len(ids)).all()), ids


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


@pytest.mark.parametrize("size", [4096, 4])
@pytest.mark.parametrize("seed", range(3))
def test_concurrent_admissions_keep_prompt_entries_one_token_short(cuda_modules, monkeypatch, seed, size):
    """Each drafting admission keeps the state at ``len(prompt) - 1`` (at most ``KEEP`` entries) and resumes from the
    longest entry its prompt strictly extends; a serial reference admission neither resumes nor adds. No stream
    holds an entry's GDN state or conv rows: a stream that decodes into its own in place leaves every entry as it
    was."""

    m = cuda_modules
    made = _concurrent(m, monkeypatch, size)
    dec, shadow = _decoder(m), PrefixCache(KEEP)
    for prompt, draft in _prompts(seed):
        hit = shadow.longest(prompt) if draft else None
        if draft:
            shadow.add(prompt[:entry_end(prompt)], None, None)
        s = m.multi.Stream(list(prompt), 3, None, draft=draft)
        _serve(dec, s)
        assert s.cached == (len(hit[0]) if hit else 0) and s.st.pos == len(prompt) and _ids_of(s.st) == prompt
        assert _cache_ids(dec) == [ids for ids, _, _ in shadow.entries]
        if draft:                                            # the prefill's own GDN state, not a copy
            assert any(dec.cache.entries[-1][1].rec[0] is t for t in made)
        _decode_in_place(s)
        _check_cache(dec)
        dec.finish([s])


def test_a_one_token_prompts_entry_is_a_copy(cuda_modules, monkeypatch):
    """A one-token prompt's entry is its whole prompt, the stream's own state: it is copied, as on main."""

    m = cuda_modules
    made = _concurrent(m, monkeypatch)
    dec = _decoder(m)
    s = m.multi.Stream([7], 3, None)
    _serve(dec, s)
    assert _cache_ids(dec) == [[7]] and not any(dec.cache.entries[0][1].rec[0] is t for t in made)
    _decode_in_place(s)
    _check_cache(dec)


@pytest.mark.parametrize("n", [5, 6, 9])
def test_a_prompt_filled_in_steps_keeps_the_same_entry(cuda_modules, monkeypatch, n):
    """While another stream decodes, a prompt prefills ``STEP`` rows a step; the step that reaches the prompt's end
    keeps the entry, also when it starts at the entry's point (n = 5 and 9 with steps of 2)."""

    m = cuda_modules
    _concurrent(m, monkeypatch, size=4)
    monkeypatch.setattr(m.multi, "STEP", 2)
    dec = _decoder(m)
    _serve(dec, m.multi.Stream([1, 2, 3], 64, None))        # decoding from here on
    prompt = list(range(40, 40 + n))
    s = m.multi.Stream(prompt, 3, None)
    dec.admit(s)
    starts = []
    while any(x is s for x in dec.filling):
        starts.append(s.st.pos)
        dec._fill()
    assert starts == list(range(0, n, 2))
    assert _cache_ids(dec) == [[1, 2], prompt[:-1]]
    _decode_in_place(s)
    _check_cache(dec)


def test_concurrent_message_starts_and_the_entry_before_the_end(cuda_modules, monkeypatch):
    """With message starts, the decoder keeps a state at each (``kept``); the entry at ``entry_end`` too, unless the
    last start is less than ``MIN_GAP`` before the prompt's end, as without this change."""

    m = cuda_modules
    _concurrent(m, monkeypatch)
    monkeypatch.setattr(m.multi, "MIN_GAP", 3)
    dec = _decoder(m, points=lambda ids: [4] if len(ids) > 4 else [])
    far, near = list(range(40, 50)), list(range(60, 66))
    for prompt in (far, near):
        s = m.multi.Stream(prompt, 3, None)
        _serve(dec, s)
        _decode_in_place(s)
        dec.finish([s])
    assert _cache_ids(dec) == [far[:4], far[:-1], near[:4]]
    _check_cache(dec)
    s = m.multi.Stream(far[:-1] + [NL2, 90, 91], 3, None)     # the next turn resumes from the entry
    _serve(dec, s)
    assert s.cached == len(far) - 1 and s.stops == []


def test_both_ranks_admit_from_and_keep_the_same_entries(cuda_modules, monkeypatch):
    m = cuda_modules
    queue = []

    def share(values, rank, device):
        if rank == 0:
            queue.append(list(values))
            return list(values)
        return queue.pop(0)

    monkeypatch.setattr(m.multi, "_share", share)
    _concurrent(m, monkeypatch, size=4)
    dec0, dec1 = _decoder(m, rank=0, world=2), _decoder(m, rank=1, world=2)
    caches0, cached = [], []
    for prompt, draft in _prompts(11, count=24):
        s = m.multi.Stream(list(prompt), 3, None, draft=draft)
        _serve(dec0, s)
        caches0.append(_cache_ids(dec0))
        cached.append(s.cached)
        dec0.finish([s])
    queue.append([])                                         # rank 0 ends the follow loop
    caches1, done = [], dec1._finish

    def finish_and_record(sid):
        caches1.append(_cache_ids(dec1))
        done(sid)

    dec1._finish = finish_and_record
    dec1.follow()
    assert caches1 == caches0 and any(cached)
    _check_cache(dec1)


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

    def fake_chunk(w, tokens, st, *, tp=False, capture_taps=False, last=True, every=False, cut=0):
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
    st.pos, st.limit, st.rec, st.kv = base, 0, [("rec at", base), None], [None, ("start",)]
    st.conv = [m.torch.zeros(3, 2), None]                   # conv rows in storage of their own (prefill._own)
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

    def attn_prep(qg, key, q_norm, k_norm, pos, inv_freq, eps, *, heads, kv_heads, head_dim):
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
    monkeypatch.setattr(prefill, "prefill_glue", pg)         # the prompt glue, whichever prefill_chunk picks
    monkeypatch.setattr(prefill, "prefill_bf16", pg)
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

    def lin(n, k):
        return QLinear(torch.randn(n, k, generator=gen) * k ** -0.5, torch.empty(0), torch.empty(0))

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
    embed = QLinear(torch.randn(c.vocab, c.hidden, generator=gen), torch.empty(0), torch.empty(0))
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


@pytest.mark.parametrize("length,cached,stops", [(130, 0, [64]), (130, 0, [30, 100]), (130, 5, [40]), (70, 0, [65])])
def test_message_starts_and_a_point_past_them_keep_fresh_prefill_states(cpu, length, cached, stops):
    """``prefill(stops=..., keep=..., keep_at=...)``, as the engine calls it with message starts: the prompt's
    state, logits and first token are a plain prefill's, and the state at each stop and the kept state are fresh
    prefills of their prefixes."""

    torch = cpu.torch
    prompt = _prompt(length, 90 + length)

    def prefix():
        return cpu.run(prompt[:cached])[0][0] if cached else None

    (ref, ref_pending), ref_logits = cpu.run(prompt, state=prefix())
    at, point = {}, entry_end(prompt)
    (st, pending, (kept, _)), logits = cpu.run(prompt, state=prefix(), stops=stops, keep_at=point,
                                               keep=lambda p, state, snap: at.setdefault(p, state))
    assert _bits_equal(torch, logits, ref_logits) and pending == ref_pending
    _assert_same_state(torch, st, ref)
    assert sorted(at) == [p for p in stops if cached < p < length] and kept.pos == point
    for p, state in [*at.items(), (point, kept)]:
        (fresh, _), _ = cpu.run(prompt[:p])
        _assert_same_state(torch, state, fresh)


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


@pytest.mark.parametrize("cached", [0, 64])
def test_the_kept_state_holds_its_conv_rows_in_storage_of_their_own(cpu, cached):
    """A chunk's conv rows are the last three rows of its join ``[old rows | chunk rows]``, and ``.contiguous()`` on
    them returns a view that holds the whole join. The kept state's rows own storage of their own size, at a cut, at
    a span start and at the prompt's end: an entry holds its GDN states and three conv rows a layer, no more."""

    torch, c = cpu.torch, cpu.w.config
    prompt = _prompt(130, 53)
    row = (2 * c.k_heads * c.dk + c.v_heads * c.dv) * 2          # one bf16 conv row
    starts = [a for a, _ in cpu.m.prefill.chunks(cached, len(prompt))]
    for point in sorted({cached, cached + 1, starts[-1], 65, 129, 130} & set(range(cached, 131))):
        prefix = cpu.run(prompt[:cached])[0][0] if cached else None
        (st, _, (kept, _)), _ = cpu.run(prompt, state=prefix, keep_at=point)
        convs = [t for t in kept.conv if t is not None]
        assert convs and all(t.nbytes == t.untyped_storage().nbytes() == (c.conv_kernel - 1) * row for t in convs), \
            (point, [t.untyped_storage().nbytes() for t in convs])
        if point:
            (fresh, _), _ = cpu.run(prompt[:point])
            _assert_same_state(torch, kept, fresh)


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
