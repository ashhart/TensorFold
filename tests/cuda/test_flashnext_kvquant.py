"""KV cache dtypes on CUDA (``--kv-dtype bf16|int8|int4``), against bf16 and against ExLlamaV3.

The write path stores ExLlamaV3's cache-quant bits (a group of 32 rotated by H32, its absmax as an fp16
scale, the midpoint grid). 8-bit stores ``q - 128`` as int8. 4-bit stores two unsigned codes per byte,
low nibble first. The read path dequantizes in the attention kernel; the query and the merged output
carry the rotation, so the stored keys and values stay rotated. These tests check both widths against
one reference quantizer, the rotation trick against an explicit un-rotate, and the engine contract
(windows, chunks, drafted tokens, the MTP cache) with the quantized cache in place.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model, _state  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kvcache import (  # noqa: E402
    KVCache, R32, dequant_ref, h32_ref, pack_nibbles, quantize_ref, unpack_nibbles,
)

DEV = "cuda"
PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
WIDTHS = [("int8", 8), ("int4", 4)]
# half a code step, in the rotated domain, before the inverse H32 (gain sqrt(32))
HALF_STEP = {8: 256.0, 4: 16.0}
# attention output against bf16's, relative to its largest value
ATTN_REL = {8: 0.02, 4: 0.12}
NLL_DELTA = {8: 0.05, 4: 0.40}
TOP1 = {8: 0.95, 4: 0.80}


def _engines(w, dtype: str = "int8", **kw):
    """A bf16 engine and one at ``dtype``, both with windows of up to 16 rows (the prompt and chunks go through them)."""

    return (Engine(w, capacity=1024, max_rows=16, prefill_rows=16, **kw),
            Engine(w, capacity=1024, max_rows=16, prefill_rows=16, kv_dtype=dtype, **kw))


def _write(w, rows: int, seed: int, dtype: str, bits: int):
    c = w.cfg
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    p = (torch.randn((rows, c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim
                      + (c.index_heads + 1) * c.index_dim), device=DEV,
                     generator=torch.Generator(device=DEV).manual_seed(seed)) * 0.5).to(torch.bfloat16)
    pos = torch.zeros((1,), dtype=torch.int32, device=DEV)
    cache = KVCache(1024, c.kv_heads, c.head_dim, DEV, dtype)
    q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device=DEV)
    iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device=DEV)
    ikc = torch.zeros((1024, c.index_dim), dtype=torch.bfloat16, device=DEV)
    glue.attn_prep(p, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, cache.k, cache.v, iq, ikc, c.eps,
                   q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, index_heads=c.index_heads,
                   index_dim=c.index_dim, ks=cache.ks, vs=cache.vs, bits=0 if dtype == "bf16" else bits)
    return q, iq, ikc, cache


def _serial_logits(w, e, tokens) -> torch.Tensor:
    out = []
    for t in tokens:
        out.append(forward(w, e.st, e.buf, [t])[0].float().cpu())
        commit(w, e.st, e.buf, 1, 1)
    return torch.stack(out)


def _sylvester(n: int) -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (n ** 0.5)


def _exl3_h32(x: torch.Tensor) -> torch.Tensor:
    """ExLlamaV3's had_4_inreg, then had_8_subgroup, then * 1/sqrt(32).

    had_4 is the in-register butterfly on the four values a lane holds (index bits 0 and 1).
    had_8 is the subgroup butterfly on the lane index (bits 2, 3, 4): local value sign-flipped
    when ``lane & i``, then added to the peer lane's value. Their source applies the 1/sqrt(32)
    after both butterflies, which is where ``h32_ref`` puts it too.
    """

    lead = x.shape[:-1]
    v = x.reshape(-1, 8, 4).clone()
    a, b, c, d = v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, 3]
    s0, d0, s1, d1 = a + b, a - b, c + d, c - d
    v = torch.stack([s0 + s1, d0 + d1, s0 - s1, d0 - d1], dim=-1)
    for i in (1, 2, 4):
        peer = torch.empty_like(v)
        for lane in range(8):
            peer[:, lane] = v[:, lane ^ i]
        sign = torch.ones(8, dtype=v.dtype)
        for lane in range(8):
            if lane & i:
                sign[lane] = -1.0
        v = v * sign[None, :, None] + peer
    return (v.reshape(*lead, 32) * R32)


def _exl3_ext():
    """The installed ExLlamaV3 extension, if this machine has it; skips otherwise."""

    return pytest.importorskip("exllamav3_ext")


# -- the transform ---------------------------------------------------------------------------------------
def test_h32_matches_exllamav3s_butterfly_and_is_an_involution():
    """Our H32, ExLlamaV3's had_4-then-had_8 order, and the Sylvester Hadamard are the same transform.

    The two butterfly orders commute. Applied twice, H32 returns the input (it is its own inverse),
    which is what the query-rotation trick relies on.
    """

    g = torch.Generator().manual_seed(1)
    x = torch.randn((256, 32), generator=g, dtype=torch.float64)
    ours = h32_ref(x.float()).double().reshape(-1, 32)
    theirs = _exl3_h32(x).reshape(-1, 32)
    dense = x @ _sylvester(32).T
    assert float((ours - dense).abs().max()) < 1e-5
    assert float((theirs - dense).abs().max()) < 1e-5
    assert float((ours - theirs).abs().max()) < 1e-5
    back = h32_ref(ours.float()).double().reshape(-1, 32)
    assert float((back - x).abs().max()) < 1e-5


def test_unpack_pack_is_the_identity_at_4_bits():
    q = torch.randint(0, 16, (128, 64), device=DEV)
    assert torch.equal(unpack_nibbles(pack_nibbles(q)), q)


# -- the cache -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_the_kernels_write_the_reference_quantizers_bits(dtype, bits):
    """Every code and every fp16 scale the write path stores is the reference quantizer's, bit for bit.
    The indexer keys stay bf16. The query rides one H32 over each group of 32."""

    w = _model()
    rows = 6
    q_b, _, ikc_b, bf = _write(w, rows, 11, "bf16", 0)
    q_i, _, ikc_i, qi = _write(w, rows, 11, dtype, bits)
    kc, ks, vc, vs = quantize_ref(bf.k[:rows], bf.v[:rows], bits=bits)
    assert torch.equal(qi.k[:rows], kc)
    assert torch.equal(qi.ks[:rows], ks)
    assert torch.equal(qi.v[:rows], vc)
    assert torch.equal(qi.vs[:rows], vs)
    assert torch.equal(ikc_i, ikc_b)
    rot = h32_ref(q_b.float().reshape(-1, 32)).to(torch.bfloat16).reshape(rows, -1)
    assert torch.equal(q_i.reshape(rows, -1), rot)
    assert qi.ks.dtype == torch.float16 and qi.bits == bits
    assert bool((qi.ks[:rows] > 0).all())


@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_the_quantized_cache_round_trips_inside_the_grid(dtype, bits):
    """Dequantizing the stored codes lands inside the group's grid: at most half a step, through the
    inverse rotation. ``dtype`` is unused; the reference is what the kernels are checked against."""

    del dtype
    g = torch.Generator(device=DEV).manual_seed(3)
    k = (torch.randn((64, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    v = (torch.randn((64, 2, 256), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    kc, ks, vc, vs = quantize_ref(k, v, bits=bits)
    back_k = h32_ref(dequant_ref(kc, ks, bits=bits).float()).reshape(k.shape).to(torch.bfloat16)
    back_v = h32_ref(dequant_ref(vc, vs, bits=bits).float()).reshape(v.shape).to(torch.bfloat16)
    bound = float((ks.float() / HALF_STEP[bits]).max()) * 5.66
    dk = (back_k.float() - k.float()).abs().reshape(-1, 8, 32).amax(dim=-1)
    dv = (back_v.float() - v.float()).abs().reshape(-1, 8, 32).amax(dim=-1)
    assert float(dk.max()) <= bound and float(dv.max()) <= bound, (bits, float(dk.max()), bound)


def test_scales_and_dequant_match_exllamav3s_own_quantizer():
    """Scales are bit for bit ExLlamaV3's, at 8 and at 4. Their dequantizer and ours agree within
    rounding, not bit for bit.

    Their kernel stores the same fp16 absmax we do (H32, then absmax, ``compand_a == 0``). On the way
    out they fold another ``1/sqrt(32)`` into the scale and apply the unnormalized butterfly; we apply
    the normalized H32 to ``(q - (m - 0.5)) * s / m``. Those are the same linear map. The residual
    (measured here, under 0.01) is the fp16 store in their kernel against our bf16 store, not a
    different grid. Their packed words are a different layout (uint32 bit planes, value j in bits
    ``[j*bits, (j+1)*bits)``) from our int8 bytes and our low-nibble-first pairs, so the words are
    not compared.
    """

    ext = _exl3_ext()
    g = torch.Generator(device=DEV).manual_seed(7)
    for bits, words in ((8, 8), (4, 4)):
        n = 2048
        src = (torch.randn((n, 32), generator=g, device=DEV) * 0.7).to(torch.float16)
        out = torch.zeros((n, words), dtype=torch.int32, device=DEV)
        scales = torch.zeros((n,), dtype=torch.float16, device=DEV)
        ext.quant_cache_cont(src, out, scales, 0.0)
        back = torch.zeros_like(src)
        ext.dequant_cache_cont(out, scales, back, 0.0)
        x = src.reshape(n, 1, 32)
        kc, ks, _, _ = quantize_ref(x, x, bits=bits)
        assert torch.equal(ks.reshape(n).half(), scales), bits
        mine = dequant_ref(kc, ks, bits=bits).float().reshape(n, 32)
        unrot = h32_ref(mine).reshape(n, 32)
        gap = float((unrot - back.float()).abs().max())
        assert gap < 0.01, (bits, gap)


def test_nbytes_includes_the_fp16_scales():
    """K and V only, 12 layers, 2 KV heads, head dim 256. Scales are in the count."""

    layers, hk, d, cap = 12, 2, 256, 4096
    bf = KVCache(cap, hk, d, DEV, "bf16").nbytes * layers // cap
    i8 = KVCache(cap, hk, d, DEV, "int8").nbytes * layers // cap
    i4 = KVCache(cap, hk, d, DEV, "int4").nbytes * layers // cap
    assert (bf, i8, i4) == (24576, 13056, 6912)
    cache = KVCache(cap, hk, d, DEV, "int4")
    assert cache.nbytes == cache.k.nbytes + cache.v.nbytes + cache.ks.nbytes + cache.vs.nbytes
    assert cache.ks.numel() == cap * hk * (d // 32)


def test_an_unknown_cache_dtype_is_refused():
    with pytest.raises(ValueError):
        KVCache(16, 2, 256, DEV, "fp8")
    with pytest.raises(ValueError):
        Engine(_model(), capacity=64, max_rows=8, prefill_rows=16, kv_dtype="int2")


# -- attention -------------------------------------------------------------------------------------------
def _attn_pair(q, k, v, pos0, rows, bits, scratch):
    """Kernel with the rotation trick, and the same kernel on explicitly un-rotated dequantized values."""

    d = q.shape[-1]
    scale = d ** -0.5
    kc, ks, vc, vs = quantize_ref(k, v, bits=bits)
    qr = h32_ref(q.float().reshape(-1, 32)).to(torch.bfloat16).view_as(q)
    got = attn_mod.attention(qr, kc, vc, pos0, scratch, rows, scale, ks=ks, vs=vs, bits=bits).clone()
    uk = h32_ref(dequant_ref(kc, ks, bits=bits).float()).reshape(k.shape).to(torch.bfloat16)
    uv = h32_ref(dequant_ref(vc, vs, bits=bits).float()).reshape(v.shape).to(torch.bfloat16)
    ref = attn_mod.attention(q, uk, uv, pos0, scratch, rows, scale).clone()
    return got, ref


@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_attention_reads_the_quantized_cache_within_a_stated_tolerance(dtype, bits):
    del dtype
    g = torch.Generator(device=DEV).manual_seed(21)
    rows, heads, hk, d, cap = 4, 24, 2, 256, 96
    q = (torch.randn((rows, heads, d), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    k = (torch.randn((cap, hk, d), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    v = (torch.randn((cap, hk, d), generator=g, device=DEV) * 0.6).to(torch.bfloat16)
    pos0 = torch.tensor([cap - rows], dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, heads, d, cap, DEV, budget=2048, ratio=4)
    ref = attn_mod.attention(q, k, v, pos0, scratch, rows, d ** -0.5).clone()
    kc, ks, vc, vs = quantize_ref(k, v, bits=bits)
    qr = h32_ref(q.float().reshape(-1, 32)).to(torch.bfloat16).view(rows, heads, d)
    got = attn_mod.attention(qr, kc, vc, pos0, scratch, rows, d ** -0.5, ks=ks, vs=vs, bits=bits).clone()
    rel = float((got.float() - ref.float()).abs().max()) / float(ref.float().abs().max())
    assert rel < ATTN_REL[bits], (bits, rel)
    for r in range(rows):
        one = attn_mod.attention(qr[r:r + 1], kc, vc, pos0 + r, scratch, 1, d ** -0.5, ks=ks, vs=vs,
                                 bits=bits).clone()
        assert torch.equal(one[0], got[r]), (bits, r)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("g", [4, 8, 12])
def test_the_rotation_trick_matches_an_explicit_unrotate(bits, g):
    """G is not 16. The merge kernel tiles 16 query heads and masks ``gg < G``, and H32 runs along the
    head dim, so a G under 16 does not mix the padding lanes into the stored rows.

    One case is a short window. The other starts at position 504 and runs 16 rows, so the online
    softmax rescale crosses CHUNK = 512.
    """

    hk, d = 2, 256
    heads = hk * g
    gen = torch.Generator(device=DEV).manual_seed(30 + g + bits)
    for rows, pos, cap in ((4, 20, 64), (16, 504, 640)):
        q = (torch.randn((rows, heads, d), generator=gen, device=DEV) * 0.4).to(torch.bfloat16)
        k = (torch.randn((cap, hk, d), generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
        v = (torch.randn((cap, hk, d), generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
        pos0 = torch.tensor([pos], dtype=torch.int32, device=DEV)
        scratch = attn_mod.AttnScratch(rows, heads, d, cap, DEV, budget=2048, ratio=4)
        got, ref = _attn_pair(q, k, v, pos0, rows, bits, scratch)
        gap = float((got.float() - ref.float()).abs().max())
        assert gap < 0.02, (bits, g, pos, gap)


def test_a_group_wider_than_the_tile_is_refused():
    g = torch.Generator(device=DEV).manual_seed(1)
    rows, heads, hk, d, cap = 2, 48, 2, 256, 8
    q = torch.randn((rows, heads, d), generator=g, device=DEV).to(torch.bfloat16)
    k = torch.randn((cap, hk, d), generator=g, device=DEV).to(torch.bfloat16)
    v = torch.randn((cap, hk, d), generator=g, device=DEV).to(torch.bfloat16)
    pos0 = torch.zeros((1,), dtype=torch.int32, device=DEV)
    scratch = attn_mod.AttnScratch(rows, heads, d, cap, DEV, budget=2048, ratio=4)
    with pytest.raises(ValueError, match="16 query heads"):
        attn_mod.attention(q, k, v, pos0, scratch, rows, d ** -0.5)


# -- the engine's contract -------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_windows_match_serial_steps(dtype, bits):
    del bits
    w = _model()
    _, e = _engines(w, dtype)
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
                assert torch.equal(lg[r], logits[r]), (dtype, R, r)
            commit(w, st, e.buf, R, keep)
            assert torch.equal(forward(w, st, e.buf, [nxt[keep]])[0], logits[keep]), (dtype, R, keep)


@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_chunked_prefill_is_one_shot(dtype, bits):
    del bits
    w = _model()
    _, e = _engines(w, dtype)
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
        assert torch.equal(want, wanted), (dtype, chunk)


@pytest.mark.parametrize("dtype,bits", WIDTHS)
@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_drafted_equals_serial(dtype, bits, sampling):
    del bits
    w = _model()
    _, eager = _engines(w, dtype)
    _, graphs = _engines(w, dtype, graphs=True)
    for e in (eager, graphs):
        first = prefill(e, PROMPT, sampling)
        ref = serial_decode(e, first, 24, sampling).tokens
        assert len(ref) == 24
        for depth in (1, 3, 5):
            prefill(e, PROMPT, sampling)
            got = mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.3)
            assert got.tokens == ref, (dtype, depth, e.graphs is not None)


@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_the_mtp_cache_uses_the_same_dtype_and_the_same_write_path(dtype, bits):
    """Draft steps and verification share ``State.mtp_kc``. It is built with the engine's kv_dtype, and
    ``attn_block`` writes it through the same quantizing path as the main cache. If drafts ran on a
    bf16 cache while verification ran on int4, this comparison would fail even when drafted tokens
    still matched serial (both sides would share the wrong cache)."""

    w = _model()
    bf, qi = _engines(w, dtype)
    prefill(bf, PROMPT, None)
    prefill(qi, PROMPT, None)
    assert qi.st.mtp_kc.dtype == dtype and qi.st.mtp_kc.bits == bits
    assert qi.st.kc[0].dtype == dtype and qi.st.kc[0].bits == bits
    n = qi.st.mtp_len
    assert n == bf.st.mtp_len and n > 0
    # prefill wrote scales; a bf16 cache would still be holding zeros in ks
    assert bool((qi.st.mtp_kc.ks[:n] > 0).all())
    # the MTP cache object takes the same quantizing write as the main cache, bit for bit
    c = w.cfg
    a = [layer for layer in w.layers if layer.attn is not None][0].attn
    rows = 4
    proj = (torch.randn((rows, c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim
                         + (c.index_heads + 1) * c.index_dim), device=DEV,
                        generator=torch.Generator(device=DEV).manual_seed(19)) * 0.5).to(torch.bfloat16)
    pos = torch.zeros((1,), dtype=torch.int32, device=DEV)
    bf_cache = KVCache(1024, c.kv_heads, c.head_dim, DEV, "bf16")
    q = torch.empty((rows, c.heads, c.head_dim), dtype=torch.bfloat16, device=DEV)
    iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=torch.bfloat16, device=DEV)
    ikc = torch.zeros((1024, c.index_dim), dtype=torch.bfloat16, device=DEV)
    glue.attn_prep(proj, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, bf_cache.k, bf_cache.v,
                   iq, ikc, c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim,
                   index_heads=c.index_heads, index_dim=c.index_dim, bits=0)
    glue.attn_prep(proj, pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, q, qi.st.mtp_kc.k,
                   qi.st.mtp_kc.v, iq, ikc, c.eps, q_heads=c.heads, kv_heads=c.kv_heads,
                   head_dim=c.head_dim, index_heads=c.index_heads, index_dim=c.index_dim,
                   ks=qi.st.mtp_kc.ks, vs=qi.st.mtp_kc.vs, bits=bits)
    kc, ks, vc, vs = quantize_ref(bf_cache.k[:rows], bf_cache.v[:rows], bits=bits)
    assert torch.equal(qi.st.mtp_kc.k[:rows], kc)
    assert torch.equal(qi.st.mtp_kc.ks[:rows], ks)
    assert torch.equal(qi.st.mtp_kc.v[:rows], vc)
    assert torch.equal(qi.st.mtp_kc.vs[:rows], vs)


@pytest.mark.parametrize("dtype,bits", WIDTHS)
def test_logits_stay_close_to_bf16_end_to_end(dtype, bits):
    w = _model()
    bf, qi = _engines(w, dtype)
    tokens = PROMPT + [401, 33, 2048, 5, 77, 1500, 9, 10]
    la, lb = _serial_logits(w, bf, tokens), _serial_logits(w, qi, tokens)
    want = torch.tensor(tokens[1:], device="cpu")
    nll_a = float(-torch.log_softmax(la[:-1], -1).gather(-1, want[:, None]).mean())
    nll_b = float(-torch.log_softmax(lb[:-1], -1).gather(-1, want[:, None]).mean())
    top1 = float((la[:-1].argmax(-1) == lb[:-1].argmax(-1)).float().mean())
    assert nll_b - nll_a < NLL_DELTA[bits], (bits, nll_a, nll_b)
    assert top1 > TOP1[bits], (bits, top1)


@pytest.mark.parametrize("dtype", ["int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=5, top_k=20, top_p=0.95)])
def test_sparse_rows_and_prompt_blocks_read_the_quantized_cache(dtype, sampling):
    """Past the indexer budget rows attend over their selected blocks (QSA) of the quantized cache, and prompt chunks
    attend in 256-row blocks: chunkings and a resume leave the same codes, and drafted tokens equal serial ones."""

    w = _model()
    prompt = [(37 * i + 11) % V for i in range(2600)]            # rows past 2,051 keys are sparse
    ref_e = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, graphs=True, kv_dtype=dtype)
    assert ref_e.pbuf.attn.qsa and ref_e.buf.attn.qsa
    first = prefill(ref_e, prompt, sampling)
    want = _state(ref_e)
    ref = serial_decode(ref_e, first, 24, sampling).tokens
    for rows, graphs in ((512, True), (1000, False)):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=rows, graphs=graphs, kv_dtype=dtype)
        assert prefill(e, prompt, sampling) == first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
        for depth in (3, 6):
            prefill(e, prompt, sampling)
            assert mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.3).tokens == ref, (rows, depth)
    e = Engine(w, capacity=4096, max_rows=8, prefill_rows=512, graphs=True, kv_dtype=dtype)
    prefill(e, prompt[:2300], sampling)
    kept = {"state": e.st.snapshot(), "tail": e.last_streams.clone()}
    serial_decode(e, 5, 9, sampling)                              # a reply decodes past the kept prompt
    assert prefill(e, prompt, sampling, resume=kept) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    assert mtp_decode(e, first, 24, sampling, depth=6, confidence=0.3).tokens == ref
