"""NVFP4: the format's reference decoder and the row-invariant kernels, on CPU (layout math) and on CUDA
(the Triton matmul, row invariance kernel-first, the way ``adding-a-cuda-family.md`` asks).

The checkpoint's quantized tensors are the routed experts: per expert and projection, packed E2M1 nibbles
(uint8 [N, K/2]), fp8e4m3 block scales ([N, K/16], groups of 16) and a second per-tensor scale (fp32 scalar).
The dequantized weight is code * (scale_2 * fp32(block scale)), groups of 16 — the fp32 row scale
the FP4 table stores, the E2M1 codes riding as exact bf16 operands.
"""

from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")
HAS_CUDA = importlib.util.find_spec("triton") is not None and torch.cuda.is_available()

from tensorfold.families.qwen4_exp.cuda import nvfp4  # noqa: E402


def _block_scale(n: int, k: int, g: torch.Generator):
    """E4m3 block scales in the checkpoint's range (its weights are ~1e-4)."""

    return (torch.rand(n, k // nvfp4.GS, generator=g) * 0.9 + 0.1).to(torch.float8_e4m3fn)


def _packed(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] codes -> [N, K/2] uint8 words: low nibble = even index, high nibble = odd."""

    return ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8)


def fake_layer(n: int, k: int, seed: int = 0):
    """A random layer in the checkpoint's storage. Returns (words, scales, scale_2, exact fp32 weight)."""

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g)
    q = nvfp4.e2m1_table()[codes]
    scale = _block_scale(n, k, g)
    scale2 = torch.tensor(2.0 ** -7, dtype=torch.float32)
    w = q * nvfp4.row_scales(scale, scale2).repeat_interleave(nvfp4.GS, dim=1)
    return _packed(codes), scale, scale2, w


def test_dequantize_is_modelopts_formula():
    """code * e4m3 * scale_2, with no other factor: the shared reader's numpy dequantizer and torch's fp8 cast agree."""

    import numpy as np

    from tensorfold.cuda.nvfp4 import format as fmt

    words, scale, _, _ = fake_layer(64, 128, seed=3)
    scale2 = 0.0123
    want = fmt.dequant("nvfp4", words.numpy(), scale.view(torch.uint8).numpy(), scale2)
    got = nvfp4.dequantize(words, scale, torch.tensor(scale2, dtype=torch.float32))
    assert np.allclose(got.numpy(), want, rtol=1e-6, atol=0)
    by_torch = nvfp4.unpack_codes(words) * (scale.float() * torch.tensor(scale2)).repeat_interleave(16, 1)
    assert torch.equal(got, by_torch)


def test_e2m1_table_matches_the_bf16_patterns_the_kernel_uses():
    table = nvfp4.e2m1_table()
    bits = torch.tensor(nvfp4.BF16_BITS, dtype=torch.uint16)
    mags = bits.view(torch.bfloat16).to(torch.float32)
    assert torch.equal(table[:8], mags)
    assert torch.equal(table[8:], -mags)


def test_e4m3_bits_widens_every_code_exactly():
    """The fp8e4m3 byte -> bf16 pattern, checked byte-for-byte over all 256 codes against torch's
    fp8 -> fp32 cast (torch's fp8 -> bf16 cast is broken, the fp32 cast is the format's truth)."""

    b = torch.arange(256, dtype=torch.uint8)
    ref = b.view(torch.float8_e4m3fn).float()
    ours = nvfp4._bits_to_f32(nvfp4.e4m3_bits(b))
    ok = (ours == ref) | (ref.isnan() & ours.isnan())
    assert bool(ok.all()), [hex(int(x)) for x in b[~ok]]


def test_nibble_packing_is_low_even_high_odd():
    codes = torch.tensor([[0x1, 0x2, 0xA, 0xF]], dtype=torch.int64)
    words = _packed(codes)
    assert torch.equal(words, torch.tensor([[0x21, 0xFA]], dtype=torch.uint8))
    assert torch.equal(nvfp4.unpack_codes(words), nvfp4.e2m1_table()[codes])


def test_dequantize_is_the_exact_product():
    words, scale, s2, w = fake_layer(64, 128, seed=1)
    assert torch.equal(nvfp4.dequantize(words, scale, s2), w)


def test_fp4_layout_round_trips_to_the_reference():
    words, scale, s2, _ = fake_layer(128, 256, seed=2)
    fp = nvfp4.make_fp4(words, scale, s2)
    assert torch.equal(nvfp4.dequantize_fp4(fp), nvfp4.dequantize(words, scale, s2))


def test_fp4_from_rows_stacks_the_projections_exactly():
    """The gate/up stack's form: code grids joined as bit patterns, each projection's row scales its own."""

    a = fake_layer(128, 256, seed=3)
    b = fake_layer(192, 256, seed=4)
    bits = torch.cat([nvfp4.e2m1_bits(a[0]), nvfp4.e2m1_bits(b[0])])
    rows = torch.cat([nvfp4.row_scales(a[1], a[2]), nvfp4.row_scales(b[1], b[2])])
    fp = nvfp4.fp4_from_rows(bits, rows)
    want = torch.cat([a[3], b[3]])
    assert torch.equal(nvfp4.dequantize_fp4(fp), want)


def test_fp4_from_bf16_keeps_the_rows_and_scales_are_one():
    rows = (torch.randn(64, 256) * 0.5).to(torch.bfloat16)     # BF16 rows ride as scale-1 FP4 tables
    fp = nvfp4.fp4_from_bf16(rows)
    assert torch.equal(nvfp4.dequantize_fp4(fp), rows.to(torch.float32))
    assert torch.equal(fp.scale, torch.ones(256 // nvfp4.GS, 64))


def test_split_k_depends_only_on_shape():
    sk = nvfp4.split_k(640, 2560)
    assert sk & (sk - 1) == 0 and sk >= 1
    assert (2560 // nvfp4.GS) % sk == 0 and (2560 // nvfp4.GS) // sk >= 4
    big = nvfp4.split_k(640, 2560, target=10_000_000)
    assert big >= sk and big <= 32          # bounded by the loop's caps, never by the row count


@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU with Triton")
@pytest.mark.parametrize("n,k", [(640, 2560), (2560, 640), (128, 128), (64, 64)])
def test_matmul_matches_the_reference(n, k):
    words, scale, s2, w = fake_layer(n, k, seed=5)
    fp = nvfp4.make_fp4(words.cuda(), scale.cuda(), s2)
    g = torch.Generator(device="cuda").manual_seed(6)
    x = (torch.randn(17, k, generator=g, device="cuda", dtype=torch.float32) * 0.3).to(torch.bfloat16)
    got = nvfp4.matmul(x, fp)
    want = (x.float() @ w.cuda().t()).to(torch.bfloat16)
    rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
    assert rel < 3e-2, f"fp4 matmul off the fp32 reference: rel {rel:.2e}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU with Triton")
@pytest.mark.parametrize("n,k", [(640, 2560), (2560, 640)])
def test_matmul_rows_are_row_invariant(n, k):
    """Every row of a window equals the same row computed alone, for window widths 1..16 and 17/64/128."""

    words, scale, s2, _ = fake_layer(n, k, seed=7)
    fp = nvfp4.make_fp4(words.cuda(), scale.cuda(), s2)
    g = torch.Generator(device="cuda").manual_seed(8)
    rows = (torch.randn(128, k, generator=g, device="cuda", dtype=torch.float32) * 0.3).to(torch.bfloat16)
    for m in (1, 2, 3, 16, 17, 64, 128):
        window = nvfp4.matmul(rows[:m], fp)
        for i in range(min(m, 16)):
            alone = nvfp4.matmul(rows[i:i + 1], fp)
            assert torch.equal(window[i], alone[0]), f"row {i} of a {m}-row window differs from its one-row bits"


@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU with Triton")
def test_matmul_splitk_sum_order_is_the_reduces_one():
    """The split-K reduce sums the slices in slice order (the same rank-order rule for sums)."""

    import triton

    n, k = 2560, 640                                   # the tuned shape with split > 1
    words, scale, s2, _ = fake_layer(n, k, seed=9)
    fp = nvfp4.make_fp4(words.cuda(), scale.cuda(), s2)
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    sk = nvfp4.split_for(n, k)
    assert sk > 1, "this test pins the reduce's order: it needs a split shape"
    part = torch.empty((sk, 4, n), dtype=torch.float32, device="cuda")
    out = torch.empty((4, n), dtype=torch.bfloat16, device="cuda")
    nvfp4._fp4mm[(1, n // nvfp4.BN, sk)](x, fp.weight, fp.scale, fp.scale2, out, part, 4, x.stride(0),
                                         N=n, K=k, SK=sk, BM=16, SBN=nvfp4.BN, BLOCK_N=nvfp4.BN,
                                         GPI=nvfp4.gpi_for((k // nvfp4.GS) // sk, 2), F32=False,
                                         PACKED=fp.packed, num_warps=4, num_stages=3)
    got = torch.empty_like(out)
    nvfp4._reduce[(triton.cdiv(4 * n, 1024),)](part, got, 4 * n, SK=sk, BLOCK=1024, F32=False, num_warps=4)
    serial = part[0]
    for s in range(1, sk):
        serial = serial + part[s]
    assert torch.equal(got, serial.to(torch.bfloat16)), "the bf16 face rounds each sum, as the loop does"
    # The fp32 face is the same sums with no rounding: the two faces a layer takes are the reason the
    # reduce runs in one launch at all, and this pins that it kept the loop's own arithmetic.
    f32 = torch.empty((4, n), dtype=torch.float32, device="cuda")
    nvfp4._reduce[(triton.cdiv(4 * n, 1024),)](part, f32, 4 * n, SK=sk, BLOCK=1024, F32=True, num_warps=4)
    assert torch.equal(f32, serial), "the fp32 face adds in the same order, with no bf16 rounding"


def test_a_packed_table_holds_the_checkpoints_own_bytes():
    """The point of the packed form: the device holds the file's own bytes — a nibble a value, a byte a
    scale — so a checkpoint that fits on disk fits in memory. The widened grid was four times the codes."""

    words, scale, s2, _ = fake_layer(128, 256, seed=11)
    fp = nvfp4.make_fp4(words, scale, s2)
    assert fp.packed
    assert fp.nbytes() == words.numel() + scale.numel() + fp.scale2.numel() * 4     # codes + fp8 + factors
    assert fp.weight.dtype == torch.uint8 and fp.weight.numel() == words.numel()
    assert fp.scale.dtype == torch.uint8 and fp.scale.numel() == scale.numel()


def test_a_packed_stack_keeps_each_experts_bytes_and_decodes_them():
    """A stacked table carries the same guarantee per expert, and expert e's slab still decodes to that
    expert's weights (the slicing the serving path and the grouped kernels both rely on)."""

    e, n, k = 3, 128, 256
    words = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8)
    scale = torch.randint(90, 115, (e, n, k // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    factors = torch.tensor([0.01, 0.02, 0.03])
    fp = nvfp4.stacked_fp4(words, scale, factors)
    assert fp.nbytes() == words.numel() + scale.numel() + e * n * 4
    assert fp.weight.shape == (e, n // nvfp4.BN, k // 64, 32, nvfp4.BN)
    slab = nvfp4.FP4(fp.weight[1], fp.scale[1], n, k, scale2=fp.scale2[1], packed=True)
    assert torch.equal(nvfp4.dequantize_fp4(slab), nvfp4.dequantize(words[1], scale[1], factors[1]))


def test_the_nvfp4_experts_are_capturable():
    """The NVFP4 MoE step keeps its plan on the device (the grouped kernel), so decode graphs capture it."""

    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe

    assert nvfp4_moe.MoE4.capturable is True
