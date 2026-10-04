"""DeepSeek-V4.1's native KV numerics: NVFP4 compressed entries, MXFP4 indexer keys and queries, FP8 (UE8M0) window.

``dsv41_fp4_ref`` ports DeepSeek's quantizers (DeepSeek-V4.1-Flash ``inference/kernel.py``: ``fp4_act_quant``,
``act_quant``; Copyright (c) 2026 DeepSeek, MIT License) to PyTorch. Here it is checked against an independent NumPy
oracle (nearest value by distance, ties to the even code, scales from frexp), and on a GPU the engine's Triton store /
decode kernels are checked against it byte for byte.
"""

import dsv41_fp4_ref as Q
import numpy as np
import pytest
import torch

MAGS = np.array(Q.MAGS, dtype=np.float32)
_E4M3 = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()   # 0 .. 448, ascending


def _nearest_even(table: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Index of the nearest table value to each a (table ascending, a within it), ties to the even index."""

    hi = np.clip(np.searchsorted(table, a), 1, len(table) - 1)
    lo = hi - 1
    dl, dh = a - table[lo], table[hi] - a
    return np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))


def _e2m1(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * MAGS[_nearest_even(MAGS, np.abs(y))]


def _e4m3(v: np.ndarray) -> np.ndarray:
    return _E4M3[_nearest_even(_E4M3, np.minimum(np.abs(v), 448.0))] * np.sign(v)


def _log2_ceil(t: np.ndarray) -> np.ndarray:
    m, e = np.frexp(t.astype(np.float64))
    return np.where(m == 0.5, e - 1, e)


def _groups(x: torch.Tensor, g: int) -> np.ndarray:
    a = x.float().numpy()
    return a.reshape(-1, a.shape[-1] // g, g)


def oracle_nvfp4(x: torch.Tensor) -> np.ndarray:
    g = _groups(x, 16)
    amax = np.maximum(np.abs(g).max(-1), np.float32(6 * 2 ** -9))
    s = _e4m3(amax / np.float32(6)).astype(np.float32)[..., None]
    return (_e2m1(np.clip(g / s, -6, 6)) * s).reshape(x.shape)


def oracle_mx(x: torch.Tensor, fmax: float, floor: float, fp8: bool) -> np.ndarray:
    g = _groups(x, 32)
    amax = np.maximum(np.abs(g).max(-1), np.float32(floor))
    s = np.ldexp(np.float32(1), _log2_ceil(amax * np.float32(1 / fmax))).astype(np.float32)[..., None]
    y = np.clip(g / s, -fmax, fmax)
    return ((_e4m3(y) if fp8 else _e2m1(y)) * s).reshape(x.shape)


def _bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)


def test_e2m1_ties_go_to_the_even_code():
    y = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.2499, 0.7501, 5.0001, -0.25, -0.0, -0.1, -5.5, 6.0])
    want = [0, 2, 2, 4, 4, 6, 6, 0, 2, 7, 0, 0, 0, 15, 7]
    assert Q.e2m1_codes(y).tolist() == want
    assert torch.equal(Q.e2m1_values(Q.e2m1_codes(y)), torch.from_numpy(_e2m1(y.numpy())).float())


def test_nibble_order_and_scale_bytes():
    x = torch.zeros((1, 16), dtype=torch.bfloat16)
    x[0, 0], x[0, 1], x[0, 2] = 6.0, -0.5, 3.0                # amax 6: scale e4m3(1.0) = 0x38
    packed, sc, deq = Q.nvfp4(x)
    assert packed[0, :2].tolist() == [7 | (9 << 4), 5] and sc.tolist() == [[0x38]]
    assert torch.equal(deq, x)
    k = torch.zeros((1, 32), dtype=torch.bfloat16)
    k[0, 3] = 3.0                                            # 3 / 6 = 0.5: scale 2^-1, byte 126, 3 -> 6
    packed, sc, deq = Q.mxfp4(k)
    assert sc.tolist() == [[126]] and packed[0, 1].item() == 7 << 4 and torch.equal(deq, k)


def test_floors_zero_groups_and_signed_zero():
    z = torch.zeros((1, 512), dtype=torch.bfloat16)
    z[0, 5] = -0.0
    packed, sc, deq = Q.nvfp4(z)
    assert (sc == 1).all() and (packed == 0).all() and torch.equal(deq, z)       # 2^-9: the least e4m3 subnormal
    packed, sc, deq = Q.mxfp4(z[:, :128])
    assert (sc == 1).all() and (packed == 0).all()                                # 6 * 2^-126 / 6 -> 2^-126
    assert torch.equal(Q.mxfp8(z), z)


def test_nvfp4_clamps_when_the_scale_rounds_down():
    x = torch.zeros((1, 16), dtype=torch.bfloat16)
    x[0, 0] = 6.375                                          # 6.375 / 6 = 1.0625 -> e4m3 1.0: 6.375 / 1 clamps to 6
    _, sc, deq = Q.nvfp4(x)
    assert sc.item() == 0x38 and deq[0, 0].item() == 6.0
    assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x)).float())


def test_mx_scales_next_to_powers_of_two():
    rows = []
    for k in range(-20, 12):
        for f in (1 - 2 ** -8, 1.0, 1 + 2 ** -7):            # bf16 neighbours of 6 * 2^k
            rows.append(6 * 2.0 ** k * f)
    x = torch.zeros((len(rows), 32), dtype=torch.bfloat16)
    x[:, 7] = torch.tensor(rows)
    _, sc, deq = Q.mxfp4(x)
    amax = x.float().abs().amax(-1).numpy()
    want = _log2_ceil(np.maximum(amax, np.float32(6 * 2 ** -126)) * np.float32(1 / 6)) + 127
    assert sc[:, 0].numpy().tolist() == want.tolist()
    assert torch.equal(deq.float(), torch.from_numpy(oracle_mx(x, 6, 6 * 2 ** -126, False)).float())
    assert torch.equal(Q.mxfp8(x).float(), torch.from_numpy(oracle_mx(x, 448, 1e-4, True)).float())


@pytest.mark.parametrize("scale", [1e-3, 0.3, 2.0, 40.0])
def test_random_rows_match_the_oracle(scale):
    g = torch.Generator().manual_seed(int(scale * 1000))
    x = (torch.randn((2048, 512), generator=g) * scale * torch.rand((2048, 1), generator=g)).to(torch.bfloat16)
    x[:7, :16] = 0.0
    packed, sc, deq = Q.nvfp4(x)
    assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x)).float())
    assert torch.equal(Q.dequant(packed, sc, 16, True), deq)
    k = x[:, :128].contiguous()
    packed, sc, deq = Q.mxfp4(k)
    assert torch.equal(deq.float(), torch.from_numpy(oracle_mx(k, 6, 6 * 2 ** -126, False)).float())
    assert torch.equal(Q.dequant(packed, sc, 32, False), deq)
    assert torch.equal(Q.mxfp8(x).float(), torch.from_numpy(oracle_mx(x, 448, 1e-4, True)).float())


def test_ties_reached_through_the_scale():
    """x / s landing exactly on an E2M1 midpoint (a scale of 1, 0.5 or 2^-9) rounds to the even code."""

    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    for s in (1.0, 0.5, 2.0 ** -9):
        x = torch.zeros((1, 16))
        x[0, :8], x[0, 8:] = mids * s, -mids * s
        packed, _, deq = Q.nvfp4(x.to(torch.bfloat16))
        codes = Q.unpack(packed)[0].tolist()
        assert codes == [0, 2, 2, 4, 4, 6, 6, 7, 0, 10, 10, 12, 12, 14, 14, 15]
        assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x.to(torch.bfloat16))).float())


# -- the engine's kernels against the port (GPU) -----------------------------------------------------------------
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _kernels():
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    return K


def _tables(K, n=1 << 20):
    freqs = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    return K.rope_tables(freqs, n)


def _rows(n, dim, seed, ties=True):
    """bf16 rows of mixed magnitudes, with zero groups and, at position 0 (identity RoPE), values landing on E2M1
    midpoints through scales 1, 0.5 and 2^-9."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, dim), generator=g, device="cuda") * torch.exp2(torch.randint(-12, 5, (n, 1), generator=g,
                                                                                     device="cuda").float())
    x[:3, :32] = 0.0
    if ties:
        mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
        for k, sc in enumerate((1.0, 0.5, 2.0 ** -9)):
            x[3 + k, :16] = torch.cat([mids, -mids]) * sc
    return x.to(torch.bfloat16)


@gpu
@pytest.mark.parametrize("dim,group,scale", [(512, 16, "e4m3"), (128, 32, "ue8m0")])
def test_store_is_byte_exact(dim, group, scale):
    """Fp4Rows.store (RoPE + quantize in one kernel) == the port of the reference on K.rope's bf16 output, for 1M
    rows, written at scattered slots (duplicate slots: the last writer of a row is unspecified, so none here)."""

    K = _kernels()
    cos, sin = _tables(K)
    E = 1 << 17
    rows = K.Fp4Rows(E, dim, group=group, scale=scale, device="cuda")
    for b in range(8):
        x = _rows(E, dim, b)
        pos = torch.randint(0, 1 << 20, (E,), device="cuda")
        pos[:8] = 0
        slot = torch.randperm(E, device="cuda")
        rows.store(slot, x, pos, cos, sin)
        ref = K.rope(x, pos, cos, sin)
        packed, sc, deq = Q.nvfp4(ref) if scale == "e4m3" else Q.mxfp4(ref, group)
        assert torch.equal(rows.q[slot], packed) and torch.equal(rows.s[slot], sc)
        assert torch.equal(rows.dequant()[slot], deq)


@gpu
@pytest.mark.parametrize("fmt", ["fp8", "mxfp4"])
def test_fake_quant_matches_the_port(fmt):
    K = _kernels()
    cos, sin = _tables(K)
    for b in range(4):
        shape = (4096, 512) if fmt == "fp8" else (64, 64, 128)
        x = _rows(shape[0] * (shape[1] if len(shape) == 3 else 1), shape[-1], 100 + b).view(shape)
        pos = torch.randint(0, 1 << 20, (shape[0],), device="cuda")
        pos[:2] = 0
        got = K.rope_q(x, pos, cos, sin, fmt)
        ref = K.rope(x, pos, cos, sin)
        want = Q.mxfp8(ref) if fmt == "fp8" else Q.mxfp4(ref)[2]
        assert torch.equal(got, want)


@gpu
def test_store_under_a_graph_and_row_moves():
    K = _kernels()
    from tensorfold.families.deepseek_v41.cuda.pool import move_rows

    cos, sin = _tables(K, 8192)
    rows = K.Fp4Rows(4096, 512, device="cuda")
    x = torch.zeros((4, 512), dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros((4,), dtype=torch.long, device="cuda")
    slot = torch.zeros((4,), dtype=torch.long, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        rows.store(slot, x, pos, cos, sin)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rows.store(slot, x, pos, cos, sin)
    for k in range(3):
        x.copy_(_rows(4, 512, 7 + k, ties=False))
        pos.copy_(torch.tensor([5, 77, 4000, 1], device="cuda"))
        slot.copy_(torch.tensor([10, 11, 12 + k, 400], device="cuda"))
        g.replay()
        want = Q.nvfp4(K.rope(x, pos, cos, sin))
        assert torch.equal(rows.q[slot], want[0]) and torch.equal(rows.s[slot], want[1])
    full = _rows(4096, 512, 3)
    rows.store(torch.arange(4096, device="cuda"), full, torch.arange(4096, device="cuda"), cos, sin)
    ref = rows.clone()
    move_rows(rows, 100, 140, 300)                           # overlapping, up and down
    assert torch.equal(rows.q[140:440], ref.q[100:400]) and torch.equal(rows.s[140:440], ref.s[100:400])
    move_rows(rows, 140, 100, 300)
    assert torch.equal(rows.q[100:400], ref.q[100:400])
    idx = torch.tensor([5, 9, 4095], device="cuda")
    taken = rows.take(idx)
    rows[0:8].zero_()
    rows.put_rows(idx, taken)
    assert torch.equal(rows.q[idx], ref.q[idx]) and torch.equal(rows.s[9], ref.s[9]) and not rows.q[0].any()
    assert rows.nbytes() == 4096 * K.Fp4Rows.row_bytes(512, 16) == 4096 * 288
    assert [p for p in rows.planes] == ["q", "s"] and K.Fp8Rows(1, 512, plain=64).planes == ("q", "r", "s")


@gpu
@pytest.mark.parametrize("R,streams", [(1, 1), (6, 1), (5, 2), (40, 1)])
def test_attention_over_packed_rows_equals_its_dequant(R, streams):
    """mqa over Fp4Rows == mqa over their bf16 dequantization, bit for bit: decode chunks (one stream or rows of two
    with window bases), prompt rows (_mqa_full), -1 indices, and NaN scale bytes in rows nobody selects."""

    K = _kernels()
    cos, sin = _tables(K, 1 << 16)
    E, D, H, W = 3000, 512, 32, 128
    rows = K.Fp4Rows(E, D, device="cuda")
    rows.store(torch.arange(E, device="cuda"), _rows(E, D, 11), torch.arange(E, device="cuda") * 2, cos, sin)
    rows.s[E - 100:] = 0x7F                                   # e4m3 NaN: never selected below
    g = torch.Generator(device="cuda").manual_seed(R)
    q = (torch.randn((R, H, D), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    pos = torch.arange(5000, 5000 + R, device="cuda")
    idx = torch.stack([torch.randperm(E - 100, generator=g, device="cuda")[:512] for _ in range(R)]).int()
    idx[0, 100:140] = -1
    sink = torch.randn((H,), generator=g, device="cuda")
    buf = K.AttnBuffers(max(R, 32), H, D, 512 + W, "cuda")
    if streams == 1:
        swa = torch.randn((4096, D), generator=g, device="cuda").to(torch.bfloat16)
        kw = {}
    else:                                                     # two streams' 256-row rings side by side
        swa = torch.randn((512, D), generator=g, device="cuda").to(torch.bfloat16)
        kw = {"sbase": (torch.arange(R, device="cuda") % 2) * 256, "ring": 256}
    a = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, **kw)
    deq = rows.dequant()
    b = K.mqa(q, deq, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, **kw)
    assert torch.isfinite(a).all() and torch.equal(a, b)
    c = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, **kw)              # fp32 out, no inverse RoPE
    assert torch.equal(c, K.mqa(q, deq, idx, swa, pos, sink, W, buf, D ** -0.5, **kw))


def _ik(K, E, seed):
    cos, sin = _tables(K, 1 << 16)
    keys = K.Fp4Rows(E, 128, group=32, scale="ue8m0", device="cuda")
    keys.store(torch.arange(E, device="cuda"), _rows(E, 128, seed), torch.arange(E, device="cuda"), cos, sin)
    return keys


@gpu
def test_indexer_scores_over_packed_keys_equal_their_dequant():
    """_index_scores over MXFP4 keys == over their bf16 dequantization: whole table, decode rows with key bases and a
    narrower width, the segment scratch (dequant_rows) equal to dequant(), FAST_TOPK's selection alike."""

    K = _kernels()
    from tensorfold.families.deepseek_v41.cuda import topk as TK

    E = 9000
    keys = _ik(K, E, 21)
    keys.s[E - 50:] = 255                                     # past every row's visible keys: never read
    deq = keys.dequant()
    assert torch.equal(keys.dequant_rows(torch.empty((E, 128), dtype=torch.bfloat16, device="cuda"), 0, E)[:E - 50],
                       deq[:E - 50])
    assert torch.equal(keys.dequant_rows(torch.empty((777, 128), dtype=torch.bfloat16, device="cuda"), 4000, 777),
                       deq[4000:4777])
    g = torch.Generator(device="cuda").manual_seed(3)
    R = 7
    iq = (torch.randn((R, 64, 128), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    wts = torch.randn((R, 64), generator=g, device="cuda")
    pos = torch.tensor([10, 900, 4000, 8000, 3, 5000, 6100], device="cuda")
    a = K.index_scores(iq, wts, keys[:E - 50], pos, 1)
    assert torch.equal(a, K.index_scores(iq, wts, deq[:E - 50], pos, 1))
    kbase = torch.tensor([0, 0, 1000, 1000, 2000, 2000, 2000], device="cuda")
    a = K.index_scores(iq, wts, keys, pos // 2, 2, kbase=kbase, n_keys=3000)
    assert torch.equal(a, K.index_scores(iq, wts, deq, pos // 2, 2, kbase=kbase, n_keys=3000))
    assert torch.equal(TK.top_entries(a, pos // 2, 2, 512), TK.top_entries(
        K.index_scores(iq, wts, deq, pos // 2, 2, kbase=kbase, n_keys=3000), pos // 2, 2, 512))


@gpu
@pytest.mark.parametrize("ties", [False, True])
def test_blocked_select_over_packed_keys(ties):
    """index_select_blocked over MXFP4 keys == over bf16 dequantized keys == top_entries / candidate_blocks of the
    full scores (with and without candidate blocks); with equal keys (exact ties) and TIE_KEYS the lower index wins
    on every path, as topk.py's decode selection."""

    K = _kernels()
    from tensorfold.families.deepseek_v41.cuda import topk as TK

    E, R = 6000, 700
    keys = _ik(K, E, 31)
    if ties:                                                 # blocks of identical keys: equal non-zero scores
        keys.q[1::3] = keys.q[0::3][:keys.q[1::3].shape[0]]
        keys.s[1::3] = keys.s[0::3][:keys.s[1::3].shape[0]]
    deq = keys.dequant()
    g = torch.Generator(device="cuda").manual_seed(5)
    iq = (torch.randn((R, 64, 128), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    wts = torch.randn((R, 64), generator=g, device="cuda")
    pos = torch.randint(0, E, (R,), generator=g, device="cuda")
    old = K.SELECT_SEG, K.TIE_KEYS
    try:
        K.SELECT_SEG, K.TIE_KEYS = 2048, ties
        full = K.index_scores(iq, wts, deq, pos, 1)
        want_c = K.candidate_blocks(full, pos, 1, 8, 64)
        got, cand = K.index_select_blocked(iq, wts, keys, pos, 1, 512, block=8, candidates=64)
        ref, rcand = K.index_select_blocked(iq, wts, deq, pos, 1, 512, block=8, candidates=64)
        assert torch.equal(got, ref) and torch.equal(cand, rcand)
        assert torch.equal(got, K.top_entries(full, 512))
        assert torch.equal(cand.sort(1).values, want_c.sort(1).values)
        masked, _ = K.index_select_blocked(iq, wts, keys, pos, 1, 512, block=8, blocks=want_c)
        assert torch.equal(masked, K.top_entries(K.mask_to_blocks(full, want_c, 8), 512))
        if ties:
            assert torch.equal(got, TK.top_entries(full, pos, 1, 512))
            flags = TK.candidate_flags(full, pos, 1, 8, 64)
            mine = torch.zeros_like(flags).scatter_(1, cand.clamp(min=0), 1) * (cand >= 0).any(1, keepdim=True)
            assert torch.equal(flags, mine.to(flags.dtype))
        else:                                                # tie-keyed without ties: the same sets
            K.TIE_KEYS = True
            fast, fcand = K.index_select_blocked(iq, wts, keys, pos, 1, 512, block=8, candidates=64)
            assert torch.equal(fast, got) and torch.equal(fcand, cand)
            assert torch.equal(fast, K.top_entries(full, 512))
    finally:
        K.SELECT_SEG, K.TIE_KEYS = old


@gpu
def test_unsorted_tie_topk_equals_the_int64_keys():
    """topk_lo's fp32 path (ids None) picks the int64 tie-key path's set, sorted in its order: few distinct values,
    -inf rows."""

    K = _kernels()
    g = torch.Generator(device="cuda").manual_seed(9)
    old = K.TIE_KEYS
    try:
        K.TIE_KEYS = True
        for n, k, levels in ((512, 512, 7), (300, 512, 3), (64, 33, 1000), (128, 512, 2)):
            x = torch.randint(0, levels, (n, 4000), generator=g, device="cuda").float() * 0.25 - 0.5
            x[:5, 100:] = float("-inf")                       # fewer visible than k: -inf ties too
            x[7] = 1.5
            v, i = K.topk_lo(x, k, sorted=False)
            b = x.view(torch.int32).long()
            b = torch.where(b < 0, b ^ 0x7FFFFFFF, b)
            ref = torch.topk((b << 32) | (0x7FFFFFFF - torch.arange(x.shape[1], device="cuda")), k, dim=1).values
            want = (0x7FFFFFFF - (ref & 0xFFFFFFFF)).sort(1).values
            order = i.sort(1)
            assert torch.equal(order.values, want)
            assert torch.equal(torch.gather(v, 1, order.indices), torch.gather(x, 1, want))
            vs, is_ = K.topk_lo(x, k)                         # sorted: value descending, ties lower index first
            assert torch.equal(is_, 0x7FFFFFFF - (ref & 0xFFFFFFFF)) and torch.equal(vs, torch.gather(x, 1, is_))
    finally:
        K.TIE_KEYS = old


@gpu
@pytest.mark.parametrize("mode", ["bf16", "fp8", "fp4"])
def test_one_source_of_cache_bytes(mode, monkeypatch):
    """serial.entry_bytes == the row classes' bytes; the engine's per-token figures and carveout plan follow it."""

    K = _kernels()
    from tensorfold.families.deepseek_v41.cuda import engine as EN
    from tensorfold.families.deepseek_v41.cuda import serial as S

    monkeypatch.setattr(S, "KV_MODE", mode)
    comp, ik = S.entry_bytes()
    n = 1000
    rows = {"fp4": (K.Fp4Rows(n, 512, device="cpu"), K.Fp4Rows(n, 128, group=32, scale="ue8m0", device="cpu")),
            "fp8": (K.Fp8Rows(n, 512, plain=64, device="cpu"), K.Fp8Rows(n, 128, group=128, device="cpu"))}.get(mode)
    want = (288, 68) if mode == "fp4" else (604, 134) if mode == "fp8" else (1024, 256)
    assert (comp, ik) == want
    if rows is not None:
        assert (rows[0].nbytes(), rows[1].nbytes()) == (n * comp, n * ik)
    assert EN._cache_bytes() == int(2.5 * (comp + ik)) and EN._comp_bytes() == int(2.5 * comp)
    assert EN.carved_bytes(4096, 1 << 40) == sum((4096 // r + 1) * comp for r in (2, 2, 2, 1))
