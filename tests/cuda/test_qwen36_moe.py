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


def _model(seed: int = 11):
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
    config = Config(hidden=D, intermediate=0, layers=2, heads=2, kv_heads=1, head_dim=128, vocab=V, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,), experts=E, top_k=TOP, moe_width=WIDTH)
    s, b = affine(V, D // 64)
    embed = QLinear(words(V, D // 8), s, b)
    w = Weights(config, embed, layers, norm, qlinear(V, D), torch.ones(16, device=dev))
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
