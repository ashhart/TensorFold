"""An int8 KV cache (``--kv-dtype int8``) on CUDA, against the bf16 cache and against ExLlamaV3's scheme.

The write path stores ExLlamaV3's Q8 bits (the group of 32 rotated by H32, its absmax as an fp16 scale, the
midpoint grid); the read path dequantizes them in the attention kernel, the query and the merged output
carrying the rotation instead of the keys and values. These tests check the cache the kernels write against
an independent reference quantizer, the attention kernel against the bf16 cache within a stated tolerance,
and the engine's own contract (a window's rows == serial steps, chunks == one shot, drafted == serial) with
the int8 cache in place.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache, dequant_ref, h32_ref, quantize_ref  # noqa: E402

DEV = "cuda"
PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]


def _engines(w, **kw):
    """The same weights through a bf16 and an int8 engine (and a fresh pair per call: the caches are state)."""

    return (Engine(w, capacity=1024, max_rows=8, prefill_rows=16, **kw),
            Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype="int8", **kw))


def _write(w, rows: int, seed: int, quantized: bool):
    """Run the attention write path over ``rows`` random projections; return the q, the indexer keys and the
    cache it wrote."""

    c = w.cfg
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    p = (torch.randn((rows, c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim
                      + (c.index_heads + 1) * c.index_dim), device=DEV,
                     generator=torch.Generator(device=DEV).manual_seed(seed)) * 0.5).to(torch.bfloat16)
    pos = torch.zeros((1,), dtype=torch.int32, device=DEV)
    cache = KVCache(1024, c.kv_heads, c.head_dim, DEV, "int8" if quantized else "bf16")
    q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device=DEV)
    iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device=DEV)
    ikc = torch.zeros((1024, c.index_dim), dtype=torch.bfloat16, device=DEV)
    glue.attn_prep(p, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, cache.k, cache.v, iq, ikc, c.eps,
                   q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, index_heads=c.index_heads,
                   index_dim=c.index_dim, ks=cache.ks, vs=cache.vs, kvq=quantized)
    return q, iq, ikc, cache


def _serial_logits(w, e, tokens) -> torch.Tensor:
    """One row a step through the int8 (or bf16) engine: [len(tokens), V] fp32 on the host."""

    out = []
    for t in tokens:
        out.append(forward(w, e.st, e.buf, [t])[0].float().cpu())
        commit(w, e.st, e.buf, 1, 1)
    return torch.stack(out)


# -- the cache -------------------------------------------------------------------------------------------
def test_the_kernels_write_the_reference_quantizers_bits():
    """Every int8 code and every fp16 scale the write path stores is the reference quantizer's, bit for bit,
    for the keys and values the bf16 path writes; the indexer keys stay bf16."""

    w = _model()
    rows = 6
    q_b, _, ikc_b, bf = _write(w, rows, 11, False)
    q_i, _, ikc_i, qi = _write(w, rows, 11, True)
    kc, ks, vc, vs = quantize_ref(bf.k[:rows], bf.v[:rows])
    assert torch.equal(qi.k[:rows], kc)
    assert torch.equal(qi.ks[:rows], ks)
    assert torch.equal(qi.v[:rows], vc)
    assert torch.equal(qi.vs[:rows], vs)
    assert torch.equal(ikc_i, ikc_b)                       # the indexer keys are not quantized
    # the query the write path stores rides the cache's rotation: one H32 over each group of 32
    rot = h32_ref(q_b.float().reshape(-1, 32)).to(torch.bfloat16).reshape(rows, -1)
    assert torch.equal(q_i.reshape(rows, -1), rot)
    assert qi.ks.dtype == torch.float16 and qi.ks.shape[-1] == bf.k.shape[-1] // 32
    assert bool((qi.ks[:rows] > 0).all())


def test_the_quantized_cache_round_trips_inside_the_grid():
    """Dequantizing the stored codes gives the values back within the group's grid: at most half a step
    (s / 256) a value in the rotated domain, and the measured error is well inside that."""

    g = torch.Generator(device=DEV).manual_seed(3)
    k = (torch.randn((64, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    v = (torch.randn((64, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    kc, ks, vc, vs = quantize_ref(k, v)
    back_k = h32_ref(dequant_ref(kc, ks).float()).reshape(k.shape).to(torch.bfloat16)
    back_v = h32_ref(dequant_ref(vc, vs).float()).reshape(v.shape).to(torch.bfloat16)
    bound = float((ks.float() / 256.0).max()) * 5.66         # half a step, through the inverse rotation
    dk = (back_k.float() - k.float()).abs().reshape(-1, 8, 32).amax(dim=-1)
    dv = (back_v.float() - v.float()).abs().reshape(-1, 8, 32).amax(dim=-1)
    rms = float(k.float().pow(2).mean().sqrt())
    assert float(dk.max()) <= bound and float(dv.max()) <= bound, (float(dk.max()), bound)
    assert float(dk.mean()) < 0.02 * rms and float(dv.mean()) < 0.02 * rms
    assert torch.equal(ks, ks.float().half())


def test_the_scales_match_exllamav3s_own_quantizer():
    """The scheme is ExLlamaV3's, and the one number a reader can check against their engine agrees: the
    group's absmax, hence the fp16 scale, is bit for bit the same for the same fp16 groups. (The packed
    bytes inside a group are not compared: the prebuilt ``exllamav3_ext`` in this environment does not
    reproduce the packing its own source describes -- a one-hot group quantizes to a 0.1768 scale and all
    ~zero codes there.)"""

    ext = pytest.importorskip("exllamav3.ext.exllamav3_ext")
    g = torch.Generator(device=DEV).manual_seed(5)
    groups = 8192
    x = (torch.randn((groups, 32), generator=g, device=DEV) * 0.5).to(torch.float16)
    out = torch.zeros((groups, 8), dtype=torch.int32, device=DEV)
    scales = torch.zeros((groups,), dtype=torch.float16, device=DEV)
    ext.quant_cache_cont(x, out, scales, 0.0)
    pairs = x.float().reshape(groups, 1, 32).repeat(1, 2, 1)
    mine_s = quantize_ref(pairs, pairs)[1].reshape(groups, 2)[:, 0]
    agree = float((mine_s == scales).float().mean())
    assert agree > 0.9999, agree
    # their dequantizer reconstructs the same input with an error of the same size ours has
    back = torch.zeros((groups, 32), dtype=torch.float16, device=DEV)
    ext.dequant_cache_cont(out, scales, back, 0.0)
    theirs = float((back.float() - x.float()).abs().max())
    assert theirs < 0.05, theirs


def test_the_int8_cache_is_1_88x_smaller():
    """Bytes a token for the model's own shape: 12 attention layers, 2 KV heads, head dim 256, K and V."""

    layers, hk, d = 12, 2, 256
    bf = KVCache(4096, hk, d, DEV, "bf16").nbytes * layers // 4096
    qi = KVCache(4096, hk, d, DEV, "int8").nbytes * layers // 4096
    assert (bf, qi) == (24576, 13056)                      # 24 KiB vs 12.75 KiB a token
    assert 1.87 < bf / qi < 1.89


def test_an_unknown_cache_dtype_is_refused():
    with pytest.raises(ValueError):
        KVCache(16, 2, 256, DEV, "fp8")
    with pytest.raises(ValueError):
        Engine(_model(), capacity=64, max_rows=8, prefill_rows=16, kv_dtype="int4")


# -- attention -------------------------------------------------------------------------------------------
def test_attention_reads_the_quantized_cache_within_a_stated_tolerance():
    """The same attention kernel over the bf16 cache and over the int8 cache: the output differs by the
    quantization error only (under 2% of the row's largest value), and a row alone has its window's bits."""

    g = torch.Generator(device=DEV).manual_seed(21)
    rows, heads, hk, d, cap = 4, 24, 2, 256, 96
    q = (torch.randn((rows, heads, d), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    k = (torch.randn((cap, hk, d), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    v = (torch.randn((cap, hk, d), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, heads, d, cap, DEV, budget=2048, ratio=4)
    ref = attn_mod.attention(q, k, v, pos0, scratch, rows, d ** -0.5).clone()
    kc, ks, vc, vs = quantize_ref(k, v)
    qr = h32_ref(q.float().reshape(-1, 32)).to(torch.bfloat16).view(rows, heads, d)
    got = attn_mod.attention(qr, kc, vc, pos0, scratch, rows, d ** -0.5, ks=ks, vs=vs, kvq=True).clone()
    rel = float((got.float() - ref.float()).abs().max()) / float(ref.float().abs().max())
    assert rel < 0.02, rel
    for r in range(rows):
        one = attn_mod.attention(qr[r:r + 1], kc, vc, pos0 + r, scratch, 1, d ** -0.5, ks=ks, vs=vs,
                                 kvq=True).clone()
        assert torch.equal(one[0], got[r]), r


# -- the engine's contract -------------------------------------------------------------------------------
def test_windows_match_serial_steps_with_int8():
    w = _model()
    _, e = _engines(w)
    forward(w, e.st, e.buf, PROMPT)
    commit(w, e.st, e.buf, len(PROMPT), len(PROMPT))
    nxt = [401, 33, 2048, 5, 77, 1500, 9, 10]
    serial = e.st.clone()
    logits = []
    for t in nxt:
        logits.append(forward(w, serial, e.buf, [t])[0].clone())
        commit(w, serial, e.buf, 1, 1)
    for R in (2, 3, 6):
        for keep in sorted({1, max(1, R // 2), R}):
            st = e.st.clone()
            lg = forward(w, st, e.buf, nxt[:R])
            for r in range(R):
                assert torch.equal(lg[r], logits[r]), (R, r)
            commit(w, st, e.buf, R, keep)
            assert torch.equal(forward(w, st, e.buf, [nxt[keep]])[0], logits[keep]), (R, keep)


def test_chunked_prefill_is_one_shot_with_int8():
    """A prompt committed in chunks of 16, 5 and 1 rows ends on the same logits and the same sequence."""

    w = _model()
    _, e = _engines(w)
    prompt = PROMPT * 4
    wanted = None
    for chunk in (16, 5, 1):
        e.reset()
        lg = None
        for i in range(0, len(prompt), chunk):
            part = prompt[i:i + chunk]
            lg = forward(w, e.st, e.buf, part)
            commit(w, e.st, e.buf, len(part), len(part))
        want = lg[-1].cpu().clone()
        if wanted is None:
            wanted = want
        assert e.st.pos == len(prompt)
        assert torch.equal(want, wanted), chunk


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_drafted_equals_serial_with_int8(sampling):
    """MTP-drafted decoding emits serial decoding's tokens with the int8 cache in place, eager and through
    CUDA graphs."""

    w = _model()
    _, eager = _engines(w)
    _, graphs = _engines(w, graphs=True)
    for e in (eager, graphs):
        first = prefill(e, PROMPT, sampling)
        ref = serial_decode(e, first, 24, sampling).tokens
        assert len(ref) == 24
        for depth in (1, 3, 5):
            prefill(e, PROMPT, sampling)
            got = mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.3)
            assert got.tokens == ref, (depth, e.graphs is not None)


def test_int8_logits_stay_close_to_bf16_end_to_end():
    """The whole small model (real head sizes, two layers) teacher-forced over the same tokens, bf16 cache
    against int8 cache: the added NLL of the quantized cache and the top-1 agreement."""

    w = _model()
    bf, qi = _engines(w)
    tokens = PROMPT + [401, 33, 2048, 5, 77, 1500, 9, 10]
    la, lb = _serial_logits(w, bf, tokens), _serial_logits(w, qi, tokens)
    want = torch.tensor(tokens[1:], device="cpu")
    nll_a = float(-torch.log_softmax(la[:-1], -1).gather(-1, want[:, None]).mean())
    nll_b = float(-torch.log_softmax(lb[:-1], -1).gather(-1, want[:, None]).mean())
    top1 = float((la[:-1].argmax(-1) == lb[:-1].argmax(-1)).float().mean())
    assert nll_b - nll_a < 0.05, (nll_a, nll_b)
    assert top1 > 0.95, top1
