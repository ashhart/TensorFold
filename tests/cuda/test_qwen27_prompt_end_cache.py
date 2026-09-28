"""The 27B's prompt-end cache entry, kept one token before the prompt's end, keeps prefill and resumes exact.

A chat prompt ends in ``<think>`` and a newline (Qwen token 198); the next request's history renders that turn with an
empty reasoning block, ``<think>`` and two newlines (token 271), so only ``prompt[:n - 1]`` is a prefix of it.
``prefill(keep_at=k)`` also returns the state after the prompt's first ``k`` tokens: the chunk that holds ``k`` runs
each GDN chain as two launches, rows before ``k`` and rows from it. Compared as raw bytes on a toy model with the
27B's layer pattern (three GDN layers, then attention):

- a cut chunk leaves the chunk's state and last row as one launch does, and its state at the cut is that of a chunk
  of the rows before the cut;
- ``prefill(keep_at=k)`` leaves the prompt's state, last logits and first token as they are without it, and the kept
  state is a fresh prefill of the first ``k`` tokens, on the prompt state's key/value buffers;
- a prompt that extends the kept prefix (the next chat turn, a longer prompt, the same prompt again) resumes from it
  and ends in a fresh prefill's state, last logits and first sampled token; the engine's and the concurrent
  decoder's replies are a fresh engine's and the serial reference's.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.sampling import sample_rows  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine, entry_end  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import State  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen3_5.cuda.prefill import CHUNK, prefill_chunk, prefill_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

V = 512
NL, NL2 = 198, 271          # Qwen's "\n" and "\n\n"
THINK, END_THINK = 300, 301  # stand-ins for <think> and </think> (the real ids are past the toy vocabulary)


def _model(layers: int = 4) -> Weights:
    """The 27B's layer pattern at toy width: one key and two value GDN heads, two query heads over one KV head,
    heads of 128."""

    gen = torch.Generator(device="cuda").manual_seed(29)
    dev = "cuda"

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device=dev, dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device=dev) * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    def norm(d):
        return (1.0 + 0.1 * torch.randn(d, generator=gen, device=dev)).bfloat16()

    c = Config(hidden=128, intermediate=128, layers=layers, heads=2, kv_heads=1, head_dim=128, vocab=V,
               k_heads=1, v_heads=2, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
               rope_theta=10000000.0, eos=(0,))
    pattern = []
    for i in range(c.interval):
        gdn = attn = None
        if c.is_linear(i):
            cd = 2 * c.k_heads * c.dk + c.v_heads * c.dv
            gdn = GDN(qlinear(cd, c.hidden), qlinear(c.v_heads * c.dv, c.hidden), qlinear(c.v_heads, c.hidden),
                      qlinear(c.v_heads, c.hidden), qlinear(c.hidden, c.v_heads * c.dv),
                      (torch.randn(cd, c.conv_kernel, generator=gen, device=dev) * 0.1).bfloat16(),
                      torch.randn(c.v_heads, generator=gen, device=dev) * 0.5,
                      torch.randn(c.v_heads, generator=gen, device=dev) * 0.5, norm(c.dv))
        else:
            kv = c.kv_heads * c.head_dim
            attn = Attention(qlinear(c.heads * c.head_dim * 2, c.hidden), qlinear(kv, c.hidden), qlinear(kv, c.hidden),
                             qlinear(c.hidden, c.heads * c.head_dim), norm(c.head_dim), norm(c.head_dim))
        pattern.append(Layer(c.is_linear(i), norm(c.hidden), norm(c.hidden), gdn, attn,
                             qlinear(c.intermediate, c.hidden), qlinear(c.intermediate, c.hidden),
                             qlinear(c.hidden, c.intermediate)))
    w = Weights(c, qlinear(V, c.hidden), [pattern[i % c.interval] for i in range(layers)], norm(c.hidden),
                qlinear(V, c.hidden), torch.ones(c.rope_dims // 2, device=dev))
    prepare(w)
    return w


@pytest.fixture(scope="module")
def w():
    return _model()


def _prompt(n: int, seed: int = 5) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=gen).tolist()


def _ids(tokens) -> torch.Tensor:
    return torch.tensor(list(tokens), dtype=torch.int32, device="cuda")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def _assert_same_state(a: State, b: State) -> None:
    assert a.pos == b.pos
    for i in range(len(a.rec)):
        if a.rec[i] is None:
            assert b.rec[i] is None and b.conv[i] is None
            (ka, va), (kb, vb) = a.kv[i], b.kv[i]
            assert _same_bits(ka[:a.pos], kb[:b.pos]) and _same_bits(va[:a.pos], vb[:b.pos]), f"layer {i} K/V"
        else:
            assert _same_bits(a.rec[i], b.rec[i]), f"layer {i} GDN state"
            assert _same_bits(a.conv[i], b.conv[i]), f"layer {i} conv rows"


def _shares_kv(kept: State, st: State) -> bool:
    """The kept state holds the prompt state's key/value buffers, not buffers of its own."""

    return all(kv is None or kv is st.kv[i] for i, kv in enumerate(kept.kv))


def _chunked(w, tokens, bounds, st=None) -> State:
    st = State(w) if st is None else st
    ids = _ids(tokens)
    for a, b in bounds:
        prefill_chunk(w, ids[a:b], st)
    return st


def _prefill(monkeypatch, w, prompt, sampling=None, **kwargs):
    """``decode.prefill``'s result, with the logits it samples from."""

    seen = []

    def spy(logits, positions, samp):
        seen.append((logits.clone(), list(positions)))
        return sample_rows(logits, positions, samp)

    monkeypatch.setattr(decode, "sample_rows", spy)
    try:
        out = prefill(w, prompt, sampling, **kwargs)
    finally:
        monkeypatch.setattr(decode, "sample_rows", sample_rows)
    ((logits, positions),) = seen
    assert positions == [len(prompt)]
    return out, logits


def _turns(base: int, seed: int) -> tuple[list[int], list[int]]:
    """A prompt ending in the generation prompt, and the next turn's prompt, whose history renders that turn with an
    empty reasoning block: they agree up to the first prompt's last token, 198 against 271."""

    head = _prompt(base, seed)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(7, seed + 1) + [THINK, NL]
    return first, second


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 31, 32, 33, 64, 128, 298, 299])
@pytest.mark.parametrize("cached", [0, 137])
def test_a_cut_chunk_commits_as_one_and_returns_the_state_at_the_cut(w, cut, cached):
    """The chain kernel steps one row at a time on fp32 state: launches over [0, cut) and [cut, W) give one launch's
    rows and final state, and the state between them is a chunk of the rows before the cut."""

    tokens = _prompt(cached + 300, seed=21)
    one, two, prefix = (_chunked(w, tokens, [(0, cached)] if cached else []) for _ in range(3))
    ids = _ids(tokens)
    h_one, _ = prefill_chunk(w, ids[cached:], one)
    h_two, _, part = prefill_chunk(w, ids[cached:], two, cut=cut)
    assert _same_bits(h_one, h_two)
    _assert_same_state(two, one)
    prefill_chunk(w, ids[cached:cached + cut], prefix)
    assert part.pos == cached + cut and _shares_kv(part, two)
    _assert_same_state(part, prefix)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 127, 128, 129, 300, CHUNK + 1, CHUNK + 3])
def test_keeping_a_point_leaves_the_prefill_unchanged_and_keeps_the_prefix_state(monkeypatch, w, n):
    prompt = _prompt(n)
    (ref, ref_pending), ref_logits = _prefill(monkeypatch, w, prompt)
    for point in sorted({0, 1, 2, 3, n - 1, n, n // 2, n // 2 + 1, CHUNK} & set(range(n + 1))):
        (st, pending, (kept, snap)), logits = _prefill(monkeypatch, w, prompt, keep_at=point)
        assert _same_bits(logits, ref_logits) and pending == ref_pending, point
        _assert_same_state(st, ref)
        assert kept.pos == point and snap is None and _shares_kv(kept, st)
        if point:
            (fresh, _), _ = _prefill(monkeypatch, w, prompt[:point])
            _assert_same_state(kept, fresh)


@pytest.mark.parametrize("size", [7, 64])
@pytest.mark.parametrize("cached", [0, 37])
def test_short_spans_keep_the_same_states(w, size, cached):
    prompt = _prompt(300, seed=9)
    ref = _chunked(w, prompt, [(0, cached)] if cached else [])
    h_ref = prefill_state(w, prompt, ref, size=size)
    for point in sorted({cached, cached + 1, 63, 64, 65, 150, 299, 300} & set(range(cached, 301))):
        st = _chunked(w, prompt, [(0, cached)] if cached else [])
        h, (kept, _) = prefill_state(w, prompt, st, size=size, keep_at=point)
        assert _same_bits(h, h_ref)
        _assert_same_state(st, ref)
        assert kept.pos == point and _shares_kv(kept, st)
        if point:
            _assert_same_state(kept, _chunked(w, prompt, [(0, point)]))


@pytest.mark.parametrize("sampling", [None, Sampling(1234)])
@pytest.mark.parametrize("base", [0, 1, 2, 125, 126, 127, 128, 200, 254, 383, CHUNK - 3, CHUNK - 2, CHUNK - 1])
def test_the_next_turn_resumed_one_token_early_equals_a_fresh_prefill(monkeypatch, w, sampling, base):
    """``...<think>\\n`` then ``...<think>\\n\\n</think>\\n\\n...``: the state kept at n - 1 resumes the next turn,
    which ends in the state (``pos`` included), last logits and first sampled token of a fresh prefill."""

    first, second = _turns(base, seed=40 + base)
    assert second[:len(first)] != first and second[:len(first) - 1] == first[:-1]
    (_, _, (kept, _)), _ = _prefill(monkeypatch, w, first, sampling, keep_at=entry_end(first))
    assert kept.pos == len(first) - 1
    (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, second, sampling)
    (st, pending), logits = _prefill(monkeypatch, w, second, sampling, state=kept)
    assert torch.isfinite(fresh_logits.float()).all()
    assert _same_bits(logits, fresh_logits)
    assert pending == fresh_pending == sample_rows(fresh_logits, [len(second)], sampling)[0]
    _assert_same_state(st, fresh)


@pytest.mark.parametrize("cached", [1, 3, 128, 130, 298])
def test_a_resumed_prefill_keeps_the_prefix_state_too(monkeypatch, w, cached):
    prompt = _prompt(300, seed=6)
    prefix = _prefill(monkeypatch, w, prompt[:cached])[0][0]
    (st, _, (kept, _)), _ = _prefill(monkeypatch, w, prompt, state=prefix, keep_at=299)
    (fresh_prefix, _), _ = _prefill(monkeypatch, w, prompt[:299])
    _assert_same_state(kept, fresh_prefix)
    following = prompt[:299] + [NL2] + _prompt(40, seed=7)
    (again, again_pending), again_logits = _prefill(monkeypatch, w, following, state=kept)
    (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, following)
    assert _same_bits(again_logits, fresh_logits) and again_pending == fresh_pending
    _assert_same_state(again, fresh)


@pytest.mark.parametrize("sampling", [None, Sampling(99)])
@pytest.mark.parametrize("n", [2, 129, 200])
def test_longer_and_repeated_prompts_resume_exactly(monkeypatch, w, sampling, n):
    """The reply prefilled again after the kept prefix, a longer prompt, and the same prompt again (one token to
    prefill) all resume from the state kept at n - 1 and end as fresh prefills do."""

    prompt = _prompt(n, seed=12)
    st, pending, (kept, _) = prefill(w, prompt, sampling, keep_at=entry_end(prompt))
    reply = serial_decode(w, st, pending, 12, sampling, stop_eos=False).tokens
    for following in (prompt + reply[:-1] + _prompt(3, seed=14), prompt + _prompt(5, seed=13), list(prompt)):
        (again, again_pending), again_logits = _prefill(monkeypatch, w, following, sampling, state=kept)
        (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, following, sampling)
        assert _same_bits(again_logits, fresh_logits) and again_pending == fresh_pending
        _assert_same_state(again, fresh)


@pytest.mark.parametrize("cached,n,point", [(0, 1025, 1024), (1000, 1025, 1024), (1000, 1030, 1024),
                                            (2000, 2049, 2048)])
def test_the_kept_state_holds_the_buffers_grown_after_the_point(monkeypatch, w, cached, n, point):
    """Key/value buffers grow to at least 1,024 rows and double; one stream sets no limit. A prefill resumed at 1,000
    or 2,000 tokens grows them inside the chunk that holds the point: the kept state then holds the grown buffers,
    whose first rows are the old ones' copies, and no buffer of its own."""

    prompt = _prompt(n, seed=15)
    # two separately built prefixes: states resumed from one prefix share its key/value buffers
    prefix_a, prefix_b = ((_prefill(monkeypatch, w, prompt[:cached])[0][0] for _ in range(2)) if cached
                          else (None, None))
    (ref, ref_pending), ref_logits = _prefill(monkeypatch, w, prompt, state=prefix_a)
    (st, pending, (kept, _)), logits = _prefill(monkeypatch, w, prompt, state=prefix_b, keep_at=point)
    assert _same_bits(logits, ref_logits) and pending == ref_pending
    _assert_same_state(st, ref)
    if cached:
        assert prefix_b.kv[3][0].shape[0] < n <= st.kv[3][0].shape[0]      # grown inside the chunk
    assert _shares_kv(kept, st)
    (fresh_prefix, _), _ = _prefill(monkeypatch, w, prompt[:point])
    _assert_same_state(kept, fresh_prefix)
    following = prompt[:point] + [NL2] + _prompt(5, seed=16)
    (again, again_pending), again_logits = _prefill(monkeypatch, w, following, state=kept)
    (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, following)
    assert _same_bits(again_logits, fresh_logits) and again_pending == fresh_pending
    _assert_same_state(again, fresh)


class TapDraft:
    """DFlash2's context bookkeeping with the taps kept by position: the last ``window`` rows, ``skip``, snapshots."""

    def __init__(self, window):
        self.window, self.rows, self.end = window, [], 0

    def skip(self, n):
        self.rows, self.end = [], self.end + n

    def add_taps(self, taps):
        assert taps.shape[0] > 0
        self.rows = (self.rows + [(self.end + r, taps[r].clone()) for r in range(taps.shape[0])])[-self.window:]
        self.end += taps.shape[0]

    def snapshot(self):
        return list(self.rows), self.end


def _same_taps(a, b) -> bool:
    (rows_a, end_a), (rows_b, end_b) = a, b
    return (end_a == end_b and [p for p, _ in rows_a] == [p for p, _ in rows_b]
            and all(_same_bits(x, y) for (_, x), (_, y) in zip(rows_a, rows_b)))


@pytest.mark.parametrize("window", [40, 1000])
@pytest.mark.parametrize("point", [0, 100, 150, 151, 260, 299, 300])
def test_the_drafter_ends_as_without_the_point_and_its_snapshot_is_the_prefixs(monkeypatch, window, point):
    """With a drafter the 64-layer target captures its five taps. The drafter ends holding the rows, and bits, it holds
    without ``keep_at``, and the kept snapshot holds those a prefill of the first ``point`` tokens leaves."""

    w = _model(layers=64)
    prompt = _prompt(300, seed=10)
    ref, new = TapDraft(window), TapDraft(window)
    (ref_st, ref_pending), ref_logits = _prefill(monkeypatch, w, prompt, draft=ref)
    (st, pending, (kept, snap)), logits = _prefill(monkeypatch, w, prompt, draft=new, keep_at=point)
    assert _same_bits(logits, ref_logits) and pending == ref_pending
    _assert_same_state(st, ref_st)
    assert _same_taps(new.snapshot(), ref.snapshot())
    if point:
        prefix = TapDraft(window)
        _prefill(monkeypatch, w, prompt[:point], draft=prefix)
        assert _same_taps(snap, prefix.snapshot())
    else:
        assert snap == ([], 0)
    assert kept.pos == point


def test_an_inplace_decode_commits_into_the_state_it_is_given(w):
    """The engine hands the prompt state to the decode (``inplace``): same tokens, and the state it passed moves on,
    so nothing keeps the prompt end's GDN states once the first round commits."""

    prompt = _prompt(50, seed=31)
    st, pending = prefill(w, prompt, None)
    ref = draft_decode(w, st, prompt, pending, 16, None, None, stop_eos=False)
    assert st.pos == len(prompt)
    mine = draft_decode(w, st, prompt, pending, 16, None, None, stop_eos=False, inplace=True)
    assert mine.tokens == ref.tokens and st.pos == len(prompt) + len(ref.tokens) - 1


def _engine(w) -> Qwen27Engine:
    engine = object.__new__(Qwen27Engine)
    engine.torch = torch
    engine.tp, engine.rank, engine.max_rows, engine.allow_copy = 1, 0, 12, True
    engine.w, engine.draft, engine.eos, engine.cache = w, None, tuple(w.config.eos), []
    engine.context_window, engine.concurrent, engine.multi, engine.scheduler = 1 << 14, False, None, None
    return engine


def _generate(engine, prompt, sampling, max_tokens=24, **kwargs):
    out = []

    def on_tokens(tokens):
        out.extend(tokens)
        return False

    stats = engine.generate(list(prompt), max_tokens, sampling, on_tokens, **kwargs)
    return out, stats


@pytest.mark.parametrize("sampling", [None, Sampling(4321)])
@pytest.mark.parametrize("base", [3, 126, 127, 250])
def test_the_engine_resumes_the_next_turn_exactly(monkeypatch, w, sampling, base):
    """End to end on one GPU: the second turn resumes from the entry at n - 1 (``cached``), its prefill ends in a
    fresh prefill's state, logits and first token, and its reply is a fresh engine's and the serial reference's. The
    entries end one token before their prompts' ends and equal fresh prefills of their tokens."""

    first, second = _turns(base, seed=60 + base)
    engine = _engine(w)
    _generate(engine, first, sampling)
    assert [ids for ids, _, _ in engine.cache] == [first[:-1]]
    captured, sampled = [], []
    real_prefill = decode.prefill

    def spy_prefill(*args, **kwargs):
        out = real_prefill(*args, **kwargs)
        captured.append((clone_state(out[0]), *out[1:]))     # the decode then commits into out[0]
        return out

    def spy_sample(logits, positions, samp):                # the prefill samples first, then the decode's rounds
        sampled.append((logits.clone(), list(positions)))
        return sample_rows(logits, positions, samp)

    monkeypatch.setattr(decode, "prefill", spy_prefill)
    monkeypatch.setattr(decode, "sample_rows", spy_sample)
    reply, stats = _generate(engine, second, sampling)
    monkeypatch.setattr(decode, "prefill", real_prefill)
    monkeypatch.setattr(decode, "sample_rows", sample_rows)
    assert stats["cached"] == len(first) - 1
    ((st, pending, _),) = captured
    logits, positions = sampled[0]
    assert positions == [len(second)]
    (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, second, sampling)
    assert _same_bits(logits, fresh_logits) and pending == fresh_pending == reply[0]
    _assert_same_state(st, fresh)
    assert reply == _generate(_engine(w), second, sampling)[0] == _generate(_engine(w), second, sampling,
                                                                           draft=False)[0]
    assert [ids for ids, _, _ in engine.cache] == [first[:-1], second[:-1]]
    for ids, entry, _ in engine.cache:
        _assert_same_state(entry, _prefill(monkeypatch, w, ids)[0][0])


@pytest.mark.parametrize("sampling", [None, Sampling(77)])
def test_the_concurrent_decoder_resumes_the_next_turn_exactly(w, sampling):
    """``--parallel N``: the prefix cache holds private states at n - 1; the next turn resumes from one and decodes
    the serial reference's tokens."""

    dec = MultiDecoder(w, None, keep=8, context=1 << 14)
    first, second = _turns(140, seed=70)
    out = {}
    for prompt in (first, second):
        got: list[int] = []
        s = Stream(list(prompt), 16, sampling, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        out[tuple(prompt)] = (got, s.cached)
    st, first_token = prefill(w, second, sampling)
    want = serial_decode(w, st, first_token, 16, sampling).tokens
    assert out[tuple(second)] == (want, len(first) - 1)
    assert [ids for ids, _, _ in dec.cache.entries] == [first[:-1], second[:-1]]
    for ids, entry, _ in dec.cache.entries:
        assert all(kv is None or kv[0].shape[0] == len(ids) for kv in entry.kv)          # private copies
        _assert_same_state(entry, prefill(w, ids, None)[0])
