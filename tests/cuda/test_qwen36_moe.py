"""Qwen3.6 MoE on CUDA: routed-expert layers keep each row's serial bits, and MTP-drafted decoding equals serial."""

import random

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.cuda.moe import Routed  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill as serial_prefill  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.mtp import Head  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.weights import MTP  # noqa: E402

V, D, E, WIDTH, TOP = 256, 256, 16, 64, 4


def _model(seed: int = 11, vocab: int = V):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    def words(*shape):
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=gen, device=dev, dtype=torch.int64).to(torch.int32)

    def affine(*shape):
        return ((torch.rand(*shape, generator=gen, device=dev) * 0.003 + 0.001).bfloat16(),
                (torch.rand(*shape, generator=gen, device=dev) * 0.003 - 0.0015).bfloat16())

    def qlinear(n, k):
        s, b = affine(n, k // 64)
        return tile(QLinear(words(n, k // 8), s, b))

    def routed():
        def table(n, k):
            s, b = affine(E + 1, n, k // 64)
            return words(E + 1, n, k // 8), s, b

        ex = grouped.make([table(WIDTH, D), table(WIDTH, D)], table(D, WIDTH), 64)
        router = (torch.randn((E + 1, D), generator=gen, device=dev) * 0.05).bfloat16()
        return Routed(router, ex, TOP)

    norm = torch.ones(D, device=dev, dtype=torch.bfloat16)
    hnorm = torch.ones(128, device=dev, dtype=torch.bfloat16)
    gdn = GDN(qlinear(384, D), qlinear(128, D), qlinear(1, D), qlinear(1, D), qlinear(D, 128),
              torch.randn(384, 4, generator=gen, device=dev).bfloat16() * 0.1, torch.zeros(1, device=dev),
              torch.zeros(1, device=dev), hnorm)

    def attn():
        return Attention(qlinear(2 * 2 * 128, D), qlinear(128, D), qlinear(128, D), qlinear(D, 2 * 128), hnorm, hnorm)

    layers = [Layer(True, norm, norm, gdn, None, None, None, None, routed()),
              Layer(False, norm, norm, None, attn(), None, None, None, routed())]
    config = Config(hidden=D, intermediate=0, layers=2, heads=2, kv_heads=1, head_dim=128, vocab=vocab, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,), experts=E, top_k=TOP, moe_width=WIDTH)
    s, b = affine(vocab, D // 64)
    embed = QLinear(words(vocab, D // 8), s, b)
    w = Weights(config, embed, layers, norm, qlinear(vocab, D), torch.ones(16, device=dev))
    m = MTP(norm_e=norm, norm_h=norm, fc_e=qlinear(D, D), fc_h=qlinear(D, D), input_norm=norm, post_norm=norm,
            attn=attn(), moe=routed(), norm=norm)
    return w, Head(w, m)


def _serial(w, prompt, sampling, count):
    st, first = serial_prefill(w, prompt, sampling)
    return draft_decode(w, st, prompt, first, count, sampling, None, allow_copy=False).tokens


PROMPTS = [[5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17], [3, 4, 5]]
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.8, 0, 1.0)]


@pytest.mark.parametrize("oracle", [False, True])
def test_mtp_decode_equals_serial(monkeypatch, oracle):
    """Chains from the MTP head (random, or mostly right by an oracle) never change a token."""

    w, head = _model()
    rng = random.Random(3)
    for prompt, sampling in zip(PROMPTS, SAMPLINGS):
        want = _serial(w, prompt, sampling, 24)
        if oracle:
            real = decode.draft

            def draft(logits, position, smp, ids=None, want=want, start=len(prompt)):
                token, prob = real(logits, position, smp, ids)
                i = position - start
                return (want[i] if 0 <= i < len(want) and rng.random() < 0.8 else token), 0.9

            monkeypatch.setattr(decode, "draft", draft)
        st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
        res = decode.mtp_decode(w, head, st, mc, carry, first, 24, sampling, depth=4, confidence=0.3)
        assert res.tokens == want, (prompt, res.tokens, want)
        assert min(res.widths) >= 2
        if oracle:
            assert res.accepted > 0 and res.rounds < 23
            monkeypatch.undo()


def test_prefill_chunks_give_the_same_state_and_head_cache(monkeypatch):
    from tensorfold.families.qwen3_5.cuda import prefill as prefill_mod

    w, head = _model()
    prompt = list(range(20, 43))
    st, mc, first, carry = decode.prefill(w, head, prompt, None)
    monkeypatch.setattr(prefill_mod, "CHUNK", 5)
    monkeypatch.setattr(decode, "chunks", lambda a, b: prefill_mod.chunks(a, b, 5))
    st2, mc2, first2, carry2 = decode.prefill(w, head, prompt, None)
    assert first == first2 and st.pos == st2.pos == len(prompt) and mc.pos == mc2.pos == len(prompt) - 1
    assert all(torch.equal(a, b) for a, b in zip(st.rec, st2.rec) if a is not None)
    assert torch.equal(carry.states, carry2.states) and carry.tokens == carry2.tokens


def test_a_prompt_resumed_at_a_kept_start_equals_fresh():
    """The state, head cache and held row kept at a stop resume another prompt with that prefix to fresh bits."""

    w, head = _model()
    shared = [11, 12, 13, 14, 15, 16]
    a, b = shared + [21, 22, 23], shared + [31, 32, 33, 34]
    kept = {}
    decode.prefill(w, head, a, None, stops=[6], keep=lambda p, st, mc, held: kept.setdefault(p, (st, mc, held)))
    st, mc, held = kept[6]
    assert st.pos == 6 and mc.pos == 5 and held.shape[0] == 1
    fresh_st, fresh_mc, fresh_first, fresh_carry = decode.prefill(w, head, b, None)
    st_b, mc_b, first_b, carry_b = decode.prefill(w, head, b, None, state=st, cache=mc, held=held)
    assert first_b == fresh_first and carry_b.tokens == fresh_carry.tokens
    assert torch.equal(carry_b.states, fresh_carry.states) and mc_b.pos == fresh_mc.pos == len(b) - 1
    res = decode.mtp_decode(w, head, st_b, mc_b, carry_b, first_b, 16, None, depth=3, confidence=0.3)
    assert res.tokens == _serial(w, b, None, 16)


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_graph_replays_equal_eager_rounds(sampling):
    """Rounds replayed as CUDA graphs in fixed buffers give the eager rounds' tokens, request after request."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    w, head = _model()
    runner = Graphs(w, head, 128)
    for prompt in (PROMPTS[1], PROMPTS[0], PROMPTS[1]):
        want = _serial(w, prompt, sampling, 40)
        st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
        eager = decode.mtp_decode(w, head, st, mc, carry, first, 40, sampling, depth=3, confidence=0.0)
        st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
        graphs = decode.mtp_decode(w, head, st, mc, carry, first, 40, sampling, depth=3, confidence=0.0,
                                   runner=runner)
        assert eager.tokens == want and graphs.tokens == want
        assert graphs.widths == eager.widths and graphs.accepted == eager.accepted
    assert runner.target and runner.mtp                  # captured once, replayed by the later requests


@pytest.mark.parametrize("expandable", [False, True])
@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_graphs_recapture_after_the_buffers_grow(expandable, sampling):
    """A request past the first buffers' rows grows them and recaptures (expandable segments leave the dropped
    graphs' pool registered); drafted tokens stay serial before, across and after the growth."""

    import os

    from tensorfold.families.qwen3_5_moe.cuda.graphs import BUCKET, Graphs

    w, head = _model(vocab=1 << 20)          # two chain widths' logits map the pool's expandable segment twice
    runner = Graphs(w, head, 4 * BUCKET)
    long = [3 + (i * 7) % 200 for i in range(BUCKET)]           # with its reply, past the first BUCKET rows
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{expandable}")
    try:
        for prompt, depth, rows in ((PROMPTS[1], 3, BUCKET), (PROMPTS[0], 2, BUCKET), (long, 3, 2 * BUCKET),
                                    (PROMPTS[2], 3, 2 * BUCKET)):
            want = _serial(w, prompt, sampling, 24)
            st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
            res = decode.mtp_decode(w, head, st, mc, carry, first, 24, sampling, depth=depth, confidence=0.0,
                                    runner=runner)
            assert res.tokens == want and runner.rows == rows
    finally:
        default = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{default}")


def test_a_long_prompt_absorbs_through_the_prefill_kernel_and_decodes_serially():
    """Prompt chunks past the tree kernel's 128 rows absorb into the head with the prefill kernel; tokens stay serial."""

    w, head = _model()
    prompt = [3 + (i * 7) % 200 for i in range(300)]
    want = _serial(w, prompt, None, 24)
    st, mc, first, carry = decode.prefill(w, head, prompt, None)
    assert mc.pos == len(prompt) - 1
    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    res = decode.mtp_decode(w, head, st, mc, carry, first, 24, None, depth=3, confidence=0.0,
                            runner=Graphs(w, head, 400))
    assert res.tokens == want


def test_ignore_eos_decodes_past_end_tokens_as_serial_does():
    """``stop_eos=False`` (ignore_eos) runs drafted rounds through an end token to the count, as serial rounds do."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    w, head = _model()
    prompt = PROMPTS[0]
    st, first = serial_prefill(w, prompt, None)
    free = draft_decode(w, st, prompt, first, 40, None, None, allow_copy=False, stop_eos=False).tokens
    w.config.eos = (free[5],)                               # an end token inside the reply
    runs = {}
    for stop_eos in (False, True):
        st, mc, first, carry = decode.prefill(w, head, prompt, None)
        runs[stop_eos] = decode.mtp_decode(w, head, st, mc, carry, first, 40, None, depth=3, confidence=0.0,
                                           stop_eos=stop_eos, runner=Graphs(w, head, 128)).tokens
    assert runs[False] == free and len(free) == 40
    assert runs[True] == free[:free.index(free[5]) + 1]


# --parallel: several streams' rounds together (multi.MultiDecoder)

from contextlib import contextmanager  # noqa: E402
import threading  # noqa: E402

from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import multi_tree_forward, tree_forward  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda import multi  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.multi import MultiDecoder  # noqa: E402

LONG = [3 + (i * 7) % 200 for i in range(300)]       # repeats itself: copied continuations, and a wide head chunk
MIXED = [PROMPTS[1], PROMPTS[0], list(range(20, 60)), PROMPTS[2], LONG]
SAMPLED = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.8, 0, 1.0), Sampling(5, 1.0, 20, 0.95), None]


@contextmanager
def _segments(expandable: bool):
    import os

    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{expandable}")
    try:
        yield
    finally:
        default = "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{default}")


def _solo(w, head, prompt, sampling, count, confidence=0.3):
    st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
    return decode.mtp_decode(w, head, st, mc, carry, first, count, sampling, depth=3, confidence=confidence,
                             prompt=prompt)


def _drain(dec):
    while dec.live():
        dec.finish(dec.round())


def test_head_rows_of_several_streams_equal_each_alone():
    """One head call over several streams gives each row ``forward``'s bits and writes each stream's own cache."""

    w, head = _model()
    gen = torch.Generator(device="cuda").manual_seed(4)
    preps = [decode.prefill(w, head, p, None) for p in (PROMPTS[1], PROMPTS[0], LONG)]
    sizes = [3, 1, 16]
    rows = [torch.randn((n, D), generator=gen, device="cuda").bfloat16() for n in sizes]
    tokens = [[7 + 3 * i for i in range(n)] for n in sizes]
    alone = [mc.view(len(LONG) + 32) for _, mc, _, _ in preps]         # copies with room: every call its own
    together = [mc.view(len(LONG) + 32) for _, mc, _, _ in preps]
    want = [head.forward(c, r, t, c.pos) for c, r, t in zip(alone, rows, tokens)]
    got = head.forward_streams(together, rows, tokens, [c.pos for c in together])
    a0 = 0
    for c1, c2, n, ref in zip(alone, together, sizes, want):
        assert torch.equal(got[a0:a0 + n], ref)
        assert torch.equal(c1.k[:c1.pos + n], c2.k[:c2.pos + n]) and torch.equal(c1.v[:c1.pos + n], c2.v[:c2.pos + n])
        a0 += n


def test_verify_rows_of_many_streams_equal_each_alone():
    """Sixteen streams' windows in one forward (past the experts' 1,024-pair plan and the router's widest tiles):
    each stream's logits and final normed rows are its own window's bits."""

    w, _ = _model()
    rng = random.Random(5)
    states, wins = [], []
    for s in range(16):
        prompt = [rng.randrange(1, V) for _ in range(rng.randint(3, 40))]
        states.append(serial_prefill(w, prompt, None)[0])
        wins.append([rng.randrange(1, V) for _ in range({3: 1, 9: 7}.get(s, 16))])
    logits, _, hidden, starts = multi_tree_forward(
        w, [(t, list(range(-1, len(t) - 1)), st) for t, st in zip(wins, states)], hidden=True)
    assert starts[-1] * (TOP + 1) > 1024
    for k, (t, st) in enumerate(zip(wins, states)):
        ref, _, rows = tree_forward(w, torch.tensor(t, dtype=torch.int32, device="cuda"), list(range(-1, len(t) - 1)),
                                    st, hidden=True)
        assert torch.equal(logits[starts[k]:starts[k + 1]], ref) and torch.equal(hidden[starts[k]:starts[k + 1]], rows)


@pytest.mark.parametrize("expandable", [False, True])
@pytest.mark.parametrize("confidence", [0.0, 0.3])
def test_streams_decoded_together_equal_solo_and_serial(expandable, confidence):
    """Mixed lengths, greedy and keyed sampling, drafted and serial streams: each emits its serial tokens, and each
    drafted one takes the rounds the solo engine takes (the head's rows keep their bits too)."""

    w, head = _model()
    with _segments(expandable):
        refs = [_serial(w, p, smp, 24) for p, smp in zip(MIXED, SAMPLED)]
        solo = [_solo(w, head, p, smp, 24, confidence) for p, smp in zip(MIXED, SAMPLED)]
        dec = MultiDecoder(w, head, depth=3, confidence=confidence)
        streams = []
        for i, (prompt, sampling) in enumerate(zip(MIXED, SAMPLED)):
            got: list[int] = []
            s = Stream(prompt, 24, sampling, draft=i != 3, emit=lambda new, got=got: got.extend(new))
            dec.admit(s)
            streams.append((s, got))
        _drain(dec)
    for i, (s, got) in enumerate(streams):
        assert got == refs[i] and s.out == got and solo[i].tokens == got, i
        if s.draft:
            assert (s.rounds, s.min_rows) == (solo[i].rounds, min(solo[i].widths)), (i, s.rounds, solo[i].rounds)
        else:
            assert s.min_rows == 1 and s.rounds == len(got) - 1
    assert not dec.streams and not dec.filling


def test_a_stream_that_ignores_end_tokens_decodes_past_them_beside_one_that_stops():
    """``stop_eos=False`` is per request: that stream decodes through an end token (together, then alone in the
    graphs) while one beside it stops there, each equal to its serial reply."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    w, head = _model()
    prompt = PROMPTS[0]
    st, first = serial_prefill(w, prompt, None)
    free = draft_decode(w, st, prompt, first, 40, None, None, allow_copy=False, stop_eos=False).tokens
    w.config.eos = (free[5],)                               # an end token inside the reply
    dec = MultiDecoder(w, head, depth=3, confidence=0.0, graphs=Graphs(w, head, 1024))
    ignoring, stopping = Stream(prompt, 40, None, stop_eos=False), Stream(prompt, 40, None)
    for s in (ignoring, stopping):
        dec.admit(s)
    _drain(dec)
    assert ignoring.out == free and stopping.out == free[:free.index(free[5]) + 1]
    alone = Stream(prompt, 40, None, stop_eos=False)
    dec.admit(alone)
    _drain(dec)
    assert alone.out == free


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_streams_join_and_leave_mid_round(monkeypatch, sampling):
    """Requests arrive while others decode (their prompts prefilled a few rows a round) and finish at different
    rounds; every stream still emits its serial tokens."""

    monkeypatch.setattr(multi, "STEP", 4)
    w, head = _model()
    prompts = [PROMPTS[1], list(range(20, 43)), PROMPTS[0], PROMPTS[2], list(range(50, 81)), LONG]
    counts = [40, 12, 30, 5, 20, 16]
    arrive = {0: [0], 2: [1, 2], 5: [3], 9: [4], 12: [5]}          # round -> requests admitted before it
    refs = [_serial(w, p, sampling, n) for p, n in zip(prompts, counts)]
    dec = MultiDecoder(w, head, depth=3, confidence=0.3)
    streams: dict[int, Stream] = {}
    joined = left = 0                                  # prompts prefilling beside decoding streams; streams leaving
    first_left = None                                  # others still live, and the round the first one left
    r = 0
    while r <= max(arrive) or dec.live():
        for i in arrive.get(r, []):
            streams[i] = Stream(prompts[i], counts[i], sampling, draft=i != 2)
            dec.admit(streams[i])
        joined += bool(dec.filling) and any(not s.done for s in dec.streams.values())
        done = dec.round()
        dec.finish(done)
        if done and dec.live():
            left += 1
            first_left = r if first_left is None else first_left
        r += 1
    assert joined and left >= 2 and first_left < max(arrive)       # requests joined after others had left
    for i, s in streams.items():
        assert s.out == refs[i], (i, s.out, refs[i])


@pytest.mark.parametrize("expandable", [False, True])
def test_a_stream_past_8192_rows_beside_others(expandable):
    """A stream whose caches hold more than 8,192 rows prefills in steps while short streams decode, then another
    long one reuses the freed memory; all emit their serial tokens (expandable segments on and off)."""

    w, head = _model()
    long_a = [3 + (i * 7) % 200 for i in range(8192)]
    long_b = [5 + (i * 11) % 190 for i in range(8300)]
    with _segments(expandable):
        refs = {tuple(p): _serial(w, p, smp, n) for p, smp, n in
                ((long_a, None, 24), (long_b, SAMPLED[1], 24), (PROMPTS[1], SAMPLED[1], 48), (PROMPTS[0], None, 64))}
        dec = MultiDecoder(w, head, depth=3, confidence=0.3)
        short = [Stream(PROMPTS[1], 48, SAMPLED[1]), Stream(PROMPTS[0], 64, None)]
        for s in short:
            dec.admit(s)
        dec.finish(dec.round())
        dec.finish(dec.round())
        a = Stream(long_a, 24, None)
        dec.admit(a)
        assert len(a.st.kv[1][0]) == len(long_a) + 24 > 8192
        _drain(dec)
        b = Stream(long_b, 24, SAMPLED[1])
        dec.admit(b)
        _drain(dec)
    assert a.out == refs[tuple(long_a)] and b.out == refs[tuple(long_b)]
    assert all(s.out == refs[tuple(s.prompt)] for s in short)


def test_copied_windows_of_16_rows_keep_serial_tokens(monkeypatch):
    """Copied continuations (here mostly right, sometimes wrong) fill 16-row windows for several streams at once."""

    w, head = _model()
    prompts, counts = MIXED[:4], [40, 30, 36, 20]
    refs = {tuple(p): _serial(w, p, smp, n) for p, smp, n in zip(prompts, SAMPLED, counts)}
    rng = random.Random(9)

    class Oracle:
        def propose(self, context, most):
            prompt = next(p for p in refs if list(context[:len(p)]) == list(p))
            truth = refs[prompt][len(context) - len(prompt):][:most]
            return [t if rng.random() < 0.9 else rng.randrange(1, V) for t in truth]

    monkeypatch.setattr(multi, "CopyIndex", Oracle)
    dec = MultiDecoder(w, head, depth=3, confidence=0.3)
    streams = [Stream(p, n, smp) for p, smp, n in zip(prompts, SAMPLED, counts)]
    for s in streams:
        dec.admit(s)
    _drain(dec)
    for s in streams:
        assert s.out == refs[tuple(s.prompt)], s.prompt
    assert sum(s.rounds for s in streams) < sum(counts) // 2           # long runs of copied rows were kept


def _points_after(k):
    return lambda ids: [k] if len(ids) > k + 1 else []


@pytest.mark.parametrize("step", [1024, 3])
def test_prompts_resume_a_kept_start_and_equal_fresh(monkeypatch, step):
    """A prompt resuming another's state kept at a message start, or at a finished prompt's end, decodes a fresh
    prefill's tokens, prefill steps interleaved or not; a serial request resumes nothing."""

    from tensorfold.cuda import markers

    monkeypatch.setattr(multi, "STEP", step)
    monkeypatch.setattr(multi, "MIN_GAP", 2)
    monkeypatch.setattr(markers, "MIN_GAP", 2)
    w, head = _model()
    shared = [11, 12, 13, 14, 15, 16]
    prompts = [shared + [21, 22, 23], shared + [31, 32], shared + [41, 42, 43, 44, 45], [7, 8, 9, 10, 11]]
    refs = [_serial(w, p, smp, 16) for p, smp in zip(prompts, SAMPLED)]
    dec = MultiDecoder(w, head, depth=3, confidence=0.3, keep=8, points=_points_after(len(shared)))
    first = Stream(prompts[0], 16, SAMPLED[0])
    dec.admit(first)
    _drain(dec)
    assert first.out == refs[0] and first.cached == 0 and any(e[0] == shared for e in dec.cache.entries)
    rest = [Stream(p, 16, smp) for p, smp in zip(prompts[1:], SAMPLED[1:])]
    for s in rest:
        dec.admit(s)
    _drain(dec)
    for s, ref in zip(rest, refs[1:]):
        assert s.out == ref and s.cached == (len(shared) if s.prompt[:len(shared)] == shared else 0), s.prompt
    longer = prompts[1] + rest[0].out[:-1] + [42, 43]                # the reply's committed tokens, then new ones
    want = _serial(w, longer, SAMPLED[1], 12)
    warm = Stream(longer, 12, SAMPLED[1])
    dec.admit(warm)
    _drain(dec)
    assert warm.out == want and warm.cached == len(prompts[1])
    serial = Stream(longer, 12, SAMPLED[1], draft=False)
    dec.admit(serial)
    _drain(dec)
    assert serial.out == want and serial.cached == 0 and serial.min_rows == 1


def test_scheduler_serves_concurrent_requests_exactly():
    """Requests from several threads, as the server sends them: each reply equals its serial one."""

    w, head = _model()
    refs = [_serial(w, p, smp, 20) for p, smp in zip(MIXED, SAMPLED)]
    sched = Scheduler(MultiDecoder(w, head, depth=3, confidence=0.3), max_streams=3)
    results: dict[int, tuple] = {}

    def go(i, draft):
        got: list[int] = []
        stats = sched.submit(MIXED[i], 20, SAMPLED[i], draft, lambda new: got.extend(new) or False)
        results[(i, draft)] = (got, stats)

    threads = [threading.Thread(target=go, args=(i, d)) for i in range(len(MIXED)) for d in (True, False)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert len(results) == 2 * len(MIXED)
    for (i, draft), (got, stats) in results.items():
        assert got == refs[i] and stats["drafts"] == draft, (i, draft)
        assert stats["min_rows"] == 1 or draft


def test_a_stopped_stream_and_a_failed_prompt_end_copy_end_only_their_own(monkeypatch):
    """A client that stops reading ends its stream after that round; a prompt-end copy that runs out of memory fails
    its request; the other requests get their serial tokens and the scheduler goes on."""

    from tensorfold.families.qwen3_5.cuda.multi import kept

    w, head = _model()
    doomed = [2, 9, 4, 4, 1, 8, 8]                    # no other prompt has its length: only its copy fails

    def failing(st):
        if st.pos == len(doomed):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated at the prompt-end copy)")
        return kept(st)

    monkeypatch.setattr(multi, "kept", failing)
    refs = [_serial(w, p, smp, 20) for p, smp in zip(MIXED, SAMPLED)]
    sched = Scheduler(MultiDecoder(w, head, depth=3, confidence=0.3), max_streams=4)
    results: dict = {}

    def go(key, prompt, sampling, stop_after=0):
        got: list[int] = []

        def emit(new):
            got.extend(new)
            return bool(stop_after) and len(got) >= stop_after

        try:
            results[key] = (got, sched.submit(prompt, 20, sampling, True, emit))
        except Exception as exc:                        # noqa: BLE001
            results[key] = (got, exc)

    jobs = [(i, MIXED[i], SAMPLED[i]) for i in range(4)] + [("doomed", doomed, None), ("gone", MIXED[4], None, 5)]
    threads = [threading.Thread(target=go, args=job, daemon=True) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    got, err = results["doomed"]
    assert got == [] and isinstance(err, torch.OutOfMemoryError), err
    got, stats = results["gone"]
    assert isinstance(stats, dict) and 5 <= len(got) < 20 and got == refs[4][:len(got)]
    for i in range(4):
        got, stats = results[i]
        assert isinstance(stats, dict) and got == refs[i], (i, stats)
    assert sched.thread.is_alive() and not sched.decoder.streams and not sched.decoder.filling
    assert all(entry[0] != doomed for entry in sched.decoder.cache.entries)


def test_context_bounds_each_stream_and_warm_leaves_nothing():
    w, head = _model()
    dec = MultiDecoder(w, head, depth=3, confidence=0.3, context=40)
    dec.warm(3)
    assert not dec.live() and not dec.cache.entries
    with pytest.raises(ValueError, match="no room in the 40-token context"):
        dec.admit(Stream(list(range(1, 37)), 5))
    s = Stream(PROMPTS[1], 100, SAMPLED[1])
    dec.admit(s)
    need = len(PROMPTS[1]) + s.count                   # admission sizes the stream's caches once, in full
    assert s.count == 40 - len(PROMPTS[1]) - 4 and s.snap.cache.k.shape[0] == need + 3
    assert all(kv is None or kv[0].shape[0] == need for kv in s.st.kv)
    _drain(dec)
    assert s.out == _serial(w, PROMPTS[1], SAMPLED[1], s.count)
    assert all(kv is None or kv[0].shape[0] == need for kv in s.st.kv)
    with pytest.raises(ValueError, match="1 to 15"):
        MultiDecoder(w, head, depth=16, confidence=0.3)


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_a_stream_alone_replays_the_one_stream_graphs(sampling):
    """A stream decoding alone takes the solo engine's rounds in its graphs; another joins (both decode eagerly, the
    first still in the graphs' buffers) and leaves; the first goes on in the graphs. All emit serial tokens."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    w, head = _model()
    runner = Graphs(w, head, 1024)
    dec = MultiDecoder(w, head, depth=3, confidence=0.3, graphs=runner)
    a = Stream(PROMPTS[1], 24, sampling)
    dec.admit(a)
    _drain(dec)
    solo = _solo(w, head, PROMPTS[1], sampling, 24)
    assert a.out == _serial(w, PROMPTS[1], sampling, 24) == solo.tokens and a.rounds == solo.rounds
    assert runner.target and runner.mtp and dec.resident is None
    first, second = Stream(list(range(20, 60)), 64, sampling), Stream(PROMPTS[0], 12, SAMPLED[1])
    dec.admit(first)
    for _ in range(4):
        dec.finish(dec.round())
    assert dec.resident is first and first.st is runner.st
    dec.admit(second)
    rounds = 0
    while not second.done:
        dec.finish(dec.round())
        rounds += 1
    assert rounds > 2 and not first.done and dec.resident is first    # decoded together, first still resident
    _drain(dec)
    assert first.out == _serial(w, first.prompt, sampling, 64)
    assert second.out == _serial(w, PROMPTS[0], SAMPLED[1], 12)


@pytest.mark.parametrize("expandable", [False, True])
@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_a_stream_alone_grows_the_graph_buffers_while_others_run(expandable, sampling):
    """A long stream alone grows the graphs' buffers past 8,192 rows (recapturing into a new pool, which expandable
    segments need); a short one joins, decodes beside it and leaves; the long one goes on in the new graphs."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import BUCKET, Graphs

    w, head = _model(vocab=1 << 20)          # two chain widths' logits map the pool's expandable segment twice
    runner = Graphs(w, head, 4 * BUCKET)
    long = [3 + (i * 7) % 200 for i in range(BUCKET)]           # with its reply, past the first BUCKET rows
    with _segments(expandable):
        refs = {tuple(p): _serial(w, p, sampling, n) for p, n in ((PROMPTS[1], 24), (long, 40), (PROMPTS[0], 12),
                                                                   (PROMPTS[2], 24))}
        dec = MultiDecoder(w, head, depth=3, confidence=0.0, graphs=runner)
        short = Stream(PROMPTS[1], 24, sampling)
        dec.admit(short)
        _drain(dec)
        assert runner.rows == BUCKET
        grown = Stream(long, 40, sampling)
        dec.admit(grown)
        while grown.rounds < 2:
            dec.finish(dec.round())
        assert runner.rows == 2 * BUCKET and dec.resident is grown
        joined = Stream(PROMPTS[0], 12, sampling)
        dec.admit(joined)
        while not joined.done:
            dec.finish(dec.round())
        assert not grown.done
        _drain(dec)
        last = Stream(PROMPTS[2], 24, sampling)
        dec.admit(last)
        _drain(dec)
    assert runner.rows == 2 * BUCKET
    for s in (short, grown, joined, last):
        assert s.out == refs[tuple(s.prompt)], s.prompt


def test_constrained_streams_together_equal_solo_and_serial():
    """A grammar per stream cuts the drafts it rejects and masks each row by its path: beside plain streams, and alone
    in the one-stream graphs, a constrained stream emits its serial constrained tokens, which the grammar takes."""

    from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs

    pytest.importorskip("xgrammar")
    from toy_grammar import toy

    grammars, compiled = toy(V)
    w, head = _model()
    shaped = {0, 2, 3}

    def fresh(i):
        return {"constraint": grammars.constraint(compiled)} if i in shaped else {}

    refs, solo = [], []
    for i, (p, smp) in enumerate(zip(MIXED, SAMPLED)):
        c = fresh(i)
        st, first = serial_prefill(w, p, smp, **c)
        refs.append(draft_decode(w, st, p, first, 24, smp, None, allow_copy=False, **c).tokens)
        c = fresh(i)
        st, mc, first, carry = decode.prefill(w, head, p, smp, **c)
        solo.append(decode.mtp_decode(w, head, st, mc, carry, first, 24, smp, depth=3, confidence=0.3, prompt=p,
                                      **c).tokens)
    assert solo == refs and refs[0] != _serial(w, MIXED[0], SAMPLED[0], 24)
    for i in shaped:
        m = grammars.xgr.GrammarMatcher(compiled)
        assert all(m.accept_token(t) for t in refs[i]), i
    dec = MultiDecoder(w, head, depth=3, confidence=0.3, graphs=Graphs(w, head, 1024))
    streams = [Stream(p, 24, smp, draft=i != 3, **fresh(i)) for i, (p, smp) in enumerate(zip(MIXED, SAMPLED))]
    for s in streams:
        dec.admit(s)
    _drain(dec)
    assert [s.out for s in streams] == refs
    alone = Stream(MIXED[0], 24, SAMPLED[0], **fresh(0))
    dec.admit(alone)
    _drain(dec)
    assert alone.out == refs[0] and dec.graphs.target
