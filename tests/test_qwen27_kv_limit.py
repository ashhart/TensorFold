"""The 27B's one-stream attention caches stay within the admitted window, as its startup estimate assumes."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.test_cuda_geometry import allocations  # noqa: F401

Q = "tensorfold.families.qwen3_5.cuda."
WINDOW = 32768
END = WINDOW - 1                                     # the engine commits at most context_window - 1 rows
_gen = torch.Generator().manual_seed(27)
KEYS, VALUES = torch.randn(END, 1, 2, generator=_gen).bfloat16(), torch.randn(END, 1, 2, generator=_gen).bfloat16()


def _weights(pattern):
    """Enough of ``Weights`` for ``State``: ``pattern`` holds True for a GDN layer, False for an attention layer."""

    config = SimpleNamespace(k_heads=1, dk=2, v_heads=1, dv=2, conv_kernel=4, kv_heads=1, head_dim=2, vocab=8)
    return SimpleNamespace(config=config, layers=[SimpleNamespace(linear=x) for x in pattern],
                           norm=torch.ones(1, dtype=torch.bfloat16), head=SimpleNamespace(n=8))


def _host(monkeypatch, pattern=(True, False, True, False)):
    """The three prefill entries on the host, ``prefill_chunk`` cut to its cache writes, and every buffer size left."""

    forward = importlib.import_module(Q + "forward")
    prefill = importlib.import_module(Q + "prefill")
    decode = importlib.import_module(Q + "decode")
    decode_tp = importlib.import_module(Q + "decode_tp")
    sizes = []

    def chunk(w, tokens, st, *, tp=False, capture_taps=False, last=True, cut=0, vision=None):  # the kernels need a GPU
        a, b = st.pos, st.pos + int(tokens.shape[0])
        for i, kv in enumerate(st.kv):
            if kv is not None:
                kbuf, vbuf = prefill._grow(st, i, b)
                kbuf[a:b], vbuf[a:b] = KEYS[a:b], VALUES[a:b]
                sizes.append(kbuf.shape[0])
        st.pos = b
        return (torch.zeros((1, 2), dtype=torch.bfloat16) if last else None), None

    monkeypatch.setattr(prefill, "prefill_chunk", chunk)
    monkeypatch.setattr(forward, "_mm", lambda x, w, xs=None: torch.zeros((1, 8)))
    for mod in (decode, decode_tp):
        monkeypatch.setattr(mod, "sample_rows", lambda logits, positions, sampling: [3])
    monkeypatch.setattr(decode_tp, "_share", lambda values, rank, device: [3] if values is None else list(values))
    w = _weights(pattern)
    entries = {"one": lambda n, **kw: decode.prefill(w, range(1, n + 1), None, **kw)[0],
               "rank 0": lambda n, **kw: decode_tp.prefill_tp(w, range(1, n + 1), None, 0, **kw)[0],
               "rank 1": lambda n, **kw: decode_tp.prefill_tp(w, range(1, n + 1), None, 1, **kw)[0]}
    return entries, sizes, forward


def _intact(st, end):
    for kv in st.kv:
        if kv is not None:
            assert torch.equal(kv[0][:end], KEYS[:end]) and torch.equal(kv[1][:end], VALUES[:end])


@pytest.mark.parametrize("entry", ["one", "rank 0", "rank 1"])
@pytest.mark.parametrize("prompt,unbounded", [(29815, 59624), (31015, 62024), (31497, 62992)])
def test_a_fresh_prompt_stays_within_the_limit(allocations, monkeypatch, entry, prompt, unbounded):  # noqa: F811
    entries, sizes, _ = _host(monkeypatch)
    st = entries[entry](prompt)
    assert st.limit == 0 and max(sizes) == unbounded                     # without a limit: about twice the window
    sizes.clear()
    st = entries[entry](prompt, limit=WINDOW)
    assert st.limit == WINDOW and max(sizes) == WINDOW
    _intact(st, prompt)


def test_a_31015_token_prompt_doubles_on_its_last_three_rows(allocations, monkeypatch):  # noqa: F811
    entries, sizes, _ = _host(monkeypatch)
    entries["one"](31015)
    assert list(dict.fromkeys(sizes)) == [3876, 7753, 15506, 31012, 62024]
    sizes.clear()
    entries["one"](31015, limit=WINDOW)
    assert list(dict.fromkeys(sizes)) == [3876, 7753, 15506, 31012, 32768]


@pytest.mark.parametrize("entry", ["one", "rank 0", "rank 1"])
def test_a_resumed_prompt_keeps_its_state_limit(allocations, monkeypatch, entry):  # noqa: F811
    entries, sizes, _ = _host(monkeypatch)
    for limit, largest in ((0, 64000), (WINDOW, WINDOW)):
        kept = entries[entry](16000, limit=limit)
        sizes.clear()
        st = entries[entry](32100, state=kept)              # the engine's cached prompt ends carry the limit
        assert st.limit == limit and max(sizes) == largest
        _intact(st, 32100)
    other = entries[entry](4)
    other.limit = 777
    assert entries[entry](10, state=other, limit=WINDOW).limit == 777 and other.pos == 4


@pytest.mark.parametrize("limit,largest", [(0, 64000), (WINDOW, WINDOW)])
def test_a_reply_committed_to_the_window_end_stays_within_the_limit(allocations, monkeypatch,  # noqa: F811
                                                                    limit, largest):
    entries, sizes, forward = _host(monkeypatch, pattern=(False, False))    # commit replays GDN rows on a GPU
    st = entries["one"](16000, limit=limit)
    width = 12                                        # the one-stream verify window (max_rows)
    while st.pos < END:
        n = min(width, END - st.pos)
        record = [forward.AttentionRecord(KEYS[st.pos:st.pos + n], VALUES[st.pos:st.pos + n]) for _ in st.kv]
        rows = torch.arange(n, dtype=torch.int32)
        forward.commit(st, record, list(range(n)), (rows, torch.tensor([n], dtype=torch.int32), rows.long()))
        sizes += [kv[0].shape[0] for kv in st.kv]
    assert max(sizes) == largest
    _intact(st, END)


def _engine(tp, rank):
    from tensorfold.cuda.streams import PrefixCache
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine

    engine = object.__new__(Qwen27Engine)
    engine.tp, engine.rank, engine.max_rows, engine.allow_copy = tp, rank, 12, True
    engine.w = SimpleNamespace(norm=SimpleNamespace(device="cpu"))
    engine.draft, engine.cache, engine.multi, engine.scheduler = None, PrefixCache(4), None, None
    engine.points = None
    engine.context_window = WINDOW
    return engine


class _Followed(Exception):
    pass


def test_engine_prefills_within_its_context_window(allocations, monkeypatch):  # noqa: F811
    decode = importlib.import_module(Q + "decode")
    decode_tp = importlib.import_module(Q + "decode_tp")
    limits = []

    def prefill(w, prompt, sampling, draft=None, *, state=None, limit=0, keep_at=None, **stops):
        limits.append(("one", limit))
        st = SimpleNamespace(pos=len(prompt))
        return (st, 5) if keep_at is None else (st, 5, (SimpleNamespace(pos=keep_at), None))

    def prefill_tp(w, prompt, sampling, rank, draft=None, *, state=None, limit=0, keep_at=None, **stops):
        limits.append((f"rank {rank}", limit))
        st = SimpleNamespace(pos=len(prompt))
        return (st, 5) if keep_at is None else (st, 5, (SimpleNamespace(pos=keep_at), None))

    done = SimpleNamespace(seconds=0.0, rounds=0, widths=[])
    monkeypatch.setattr(decode, "prefill", prefill)
    monkeypatch.setattr(decode, "draft_decode", lambda *a, **kw: done)
    monkeypatch.setattr(decode_tp, "prefill_tp", prefill_tp)
    monkeypatch.setattr(decode_tp, "decode_tp", lambda *a, **kw: done)
    monkeypatch.setattr(decode_tp, "_share", lambda values, rank, device: list(values))
    for draft in (True, False):
        _engine(1, 0).generate([1, 2, 3], 4, None, lambda tokens: None, draft=draft)
        _engine(2, 0).generate([1, 2, 3], 4, None, lambda tokens: None, draft=draft)
    shared = iter([[1, 4, 0, 1, *decode_tp.pack_sampling(None)], [1, 2, 3]])
    monkeypatch.setattr(decode_tp, "_share", lambda values, rank, device: next(shared))

    def follow_once(*a, **kw):
        raise _Followed

    monkeypatch.setattr(decode_tp, "decode_tp", follow_once)
    with pytest.raises(_Followed):
        _engine(2, 1).follow()
    assert limits == [("one", WINDOW), ("rank 0", WINDOW)] * 2 + [("rank 1", WINDOW)]
