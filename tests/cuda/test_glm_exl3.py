"""GLM-5.3-Flash's EXL3 reference decoder (``cuda/exl3.py``) on synthetic tensors: the vectorized decoder against a
bit-by-bit one written from the format description, the layer's weight against its rotated forward, and the
two-rank splits against the whole layer. No GPU work.

The decoder was also checked against ExLlamaV3 0.0.43's own ``reconstruct`` and ``get_weight_tensor`` on 12 expert
matrices of Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw, bit for bit (see the family's docs)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.families.glm5_next.cuda import exl3


def _slow_values(tile_words: np.ndarray, bits: int = 4) -> np.ndarray:
    """One tile, bit by bit: int16 words -> 256 fp16 values in stream order."""

    stream = []
    for j in range(0, len(tile_words), 2):
        word = (int(tile_words[j]) & 0xFFFF) | ((int(tile_words[j + 1]) & 0xFFFF) << 16)
        stream += [(word >> (31 - b)) & 1 for b in range(32)]
    n = len(stream)
    out = []
    for p in range(256):
        end = (p + 1) * bits
        s = 0
        for b in range(end - 16, end):
            s = (s << 1) | stream[b % n]
        x = (s * 0xCBAC1FED) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = np.array([x & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
        hi = np.array([x >> 16], dtype=np.uint16).view(np.float16)[0]
        out.append(np.float16(np.float64(lo) + np.float64(hi)))
    return np.array(out, dtype=np.float16)


def _trellis(kt: int, nt: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, 64), dtype=np.int64).astype(np.int16))


def test_codebook_and_tile_order():
    values = exl3.mcg_values()
    assert values.dtype == np.float16 and values.shape == (65536,) and np.isfinite(values).all()
    assert 0.5 < float(np.std(values.astype(np.float64))) < 2.0          # a unit-scale Gaussian-like codebook
    rows, cols = exl3.tile_positions()
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(r, c) for r in range(16) for c in range(16)]


def test_vectorized_decoder_matches_bit_by_bit():
    t = _trellis(2, 3, seed=1)
    wq = exl3.unpack(t).numpy()
    rows, cols = exl3.tile_positions()
    for k in range(2):
        for n in range(3):
            slow = _slow_values(t[k, n].numpy())
            tile = wq[16 * k:16 * k + 16, 16 * n:16 * n + 16]
            assert np.array_equal(tile[rows, cols].view(np.int16), slow.view(np.int16)), (k, n)


def test_weight_and_rotated_forward_agree():
    K, N = 256, 384
    t = _trellis(K // 16, N // 16, seed=2)
    rng = np.random.default_rng(3)
    suh = torch.from_numpy((rng.standard_normal(K) * 0.02).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(N) * 0.5).astype(np.float16))
    x = torch.from_numpy(rng.standard_normal((5, K)))
    w = exl3.dequantize(t, suh, svh)
    y = exl3.forward(x, t, suh, svh)
    assert torch.allclose(x.double() @ w, y, rtol=1e-12, atol=1e-12)
    h = torch.from_numpy(exl3.hadamard(128)) / np.sqrt(128)
    assert torch.allclose(h @ h, torch.eye(128, dtype=torch.float64), atol=1e-12)


def test_two_rank_splits_add_up_to_the_layer():
    """By outputs (gate/up): each rank its columns of tiles and of svh, all of suh. By inputs (down): its rows of
    tiles and of suh, all of svh; the two partial outputs add up to the layer's output."""

    K, N = 512, 256
    t = _trellis(K // 16, N // 16, seed=4)
    rng = np.random.default_rng(5)
    suh = torch.from_numpy((rng.standard_normal(K) * 0.02).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(N) * 0.5).astype(np.float16))
    x = torch.from_numpy(rng.standard_normal((3, K)))
    y = exl3.forward(x, t, suh, svh)
    half_n, half_k = N // 2, K // 2
    by_out = [exl3.forward(x, t[:, r * half_n // 16:(r + 1) * half_n // 16], suh, svh[r * half_n:(r + 1) * half_n])
              for r in (0, 1)]
    assert torch.allclose(torch.cat(by_out, dim=1), y, rtol=1e-12, atol=1e-12)
    by_in = [exl3.forward(x[:, r * half_k:(r + 1) * half_k], t[r * half_k // 16:(r + 1) * half_k // 16],
                          suh[r * half_k:(r + 1) * half_k], svh) for r in (0, 1)]
    assert torch.allclose(by_in[0] + by_in[1], y, rtol=1e-10, atol=1e-10)


# -- the CUDA kernels (GPU only) ----------------------------------------------------------------------------------
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@cuda
def test_bf16_matmul_rows_are_independent_and_exact():
    """The BF16 matmul (EXL3 checkpoints' non-expert weights): a row alone equals the same row in any window, and the
    fp32 sums match a float64 reference to fp32 rounding."""

    from tensorfold.families.glm5_next.cuda import qmm

    gen = torch.Generator().manual_seed(1)
    for n, k in ((384, 512), (1024, 4096), (4096, 1024)):
        w = qmm.make_b16((torch.randn((n, k), generator=gen) * 0.05).cuda())
        x = (torch.randn((20, k), generator=gen)).to(torch.bfloat16).cuda()
        whole = qmm.matmul(x, w, f32=True)
        for r in (0, 7, 19):
            alone = qmm.matmul(x[r:r + 1].contiguous(), w, f32=True)
            assert torch.equal(alone[0], whole[r])
        ref = x.double() @ w.weight.double().t()
        assert (whole.double() - ref).abs().max().item() <= 1e-5 * ref.abs().max().item() * 16


@cuda
def test_bf16_matmul_of_a_prompt_chunk_keeps_each_rows_bits():
    """A prompt chunk (hundreds of rows, past the 128-row bucket) through the BF16 matmul, split-K or not, strided or
    not: each row equals its one-row call. 0.3.5.1 refused any EXL3 prompt past 128 tokens here."""

    from tensorfold.families.glm5_next.cuda import qmm

    gen = torch.Generator().manual_seed(53)
    for n, k in ((512, 1024), (4096, 1024)):
        w = qmm.make_b16((torch.randn((n, k), generator=gen) * 0.05).cuda())
        wide = torch.randn((300, k + 64), generator=gen).to(torch.bfloat16).cuda()
        for x in (wide[:, :k].contiguous(), wide[:, 64:]):
            for f32 in (True, False):
                whole = qmm.matmul(x, w, f32=f32)
                for r in (0, 127, 128, 255, 299):
                    alone = qmm.matmul(x[r:r + 1], w, f32=f32)
                    assert torch.equal(alone[0], whole[r]), (n, k, f32, r)


@cuda
def test_exl3_experts_match_the_reference_and_rows_are_independent():
    """A layer's routed experts (rotations, trellis GEMVs, SwiGLU) against the float64 reference with the kernels'
    roundings, on synthetic experts; each row alone gives the bits it gets inside a window."""

    from tensorfold.cuda import experts as grouped
    from tensorfold.families.glm5_next.cuda import exl3_mm

    D, NI, E, SLOTS, LIMIT = 512, 128, 6, 3, 10.0
    rng = np.random.default_rng(7)

    def trellis(k, n):
        return torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 64)).astype(np.int16))

    def signs(n, sc):
        return torch.from_numpy((rng.standard_normal(n) * sc).astype(np.float16))

    def expert():
        return (trellis(D, NI), trellis(D, NI), trellis(NI, D), signs(D, 0.02), signs(D, 0.02), signs(NI, 0.5),
                signs(NI, 0.5), signs(NI, 0.05), signs(D, 0.2))

    def wds(ts):
        return torch.stack([exl3_mm.words(t) for t in ts]).cuda()

    def hs(ts):
        return torch.stack(list(ts)).cuda()

    ex = [expert() for _ in range(E)]
    cols = list(zip(*ex))
    experts = exl3_mm.Exl3Experts(wds(cols[0]), wds(cols[1]), wds(cols[2]), hs(cols[3]), hs(cols[4]), hs(cols[5]),
                                  hs(cols[6]), hs(cols[7]), hs(cols[8]), E, NI, D)
    R = 3
    x = torch.randn((R, D), generator=torch.Generator().manual_seed(2)).to(torch.bfloat16).cuda()
    pick = torch.tensor([[0, 3, E], [3, 5, E], [1, 0, E]], dtype=torch.int32)

    def run(rows, picks):
        n = len(rows)
        full_pick = torch.full((8, SLOTS), E, dtype=torch.int32)
        full_pick[:n] = picks
        plan = grouped.Plan(8, SLOTS, E + 1, "cuda")
        grouped.route(full_pick[:n].contiguous().cuda(), plan)
        scratch = exl3_mm.Scratch(8, SLOTS, D, NI, "cuda")
        y = torch.zeros((8 * SLOTS, D), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x[rows].contiguous(), full_pick.cuda(), plan, experts, scratch, y, n, LIMIT)
        return y.view(8, SLOTS, D)[:n].cpu()

    def bf16(v):
        return v.float().to(torch.bfloat16).double()

    def reference(row, e):
        tg, tu, td, sg, su, vg, vu, sd, vd = ex[e]
        xr = x[row].cpu().double()
        xg = exl3.rotate(xr * sg.double(), -1).half().double()
        xu = exl3.rotate(xr * su.double(), -1).half().double()
        g = bf16(exl3.rotate(xg @ exl3.unpack(tg).double(), -1) * vg.double()).clamp(max=LIMIT)
        u = bf16(exl3.rotate(xu @ exl3.unpack(tu).double(), -1) * vu.double()).clamp(-LIMIT, LIMIT)
        act = bf16(bf16(g / (1 + torch.exp(-g))) * u)
        xd = exl3.rotate(act * sd.double(), -1).half().double()
        return exl3.rotate(xd @ exl3.unpack(td).double(), -1) * vd.double()

    y = run(list(range(R)), pick)
    for r in range(R):
        for s in range(SLOTS - 1):
            ref = reference(r, int(pick[r, s]))
            assert ((y[r, s].double() - ref).norm() / ref.norm()).item() < 2e-3, (r, s)
        alone = run([r], pick[r:r + 1])
        assert torch.equal(alone[0, :SLOTS - 1], y[r, :SLOTS - 1]), r
