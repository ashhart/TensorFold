"""The EXL3 linear (``cuda/exl3/linear.py``, ``linear.cu``, ``decode.cuh``) on a GPU, on synthetic layers: the device
decoder against ``format.unpack`` bit for bit, the layer against a float64 reference built from the decoded weight,
and row invariance — every row's bits alone, in 2, 3, 16, 17, 64 and 128-row windows and under every split plan."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

COMBOS = [("3inst", b) for b in (1, 2, 3, 4, 5, 6, 7, 8)] + [("mcg", b) for b in (1, 2, 3, 4, 6, 8)] \
    + [("mul1", b) for b in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)]
ROWS = (1, 2, 3, 16, 17, 64, 128)


def _tensors(codebook: str, bits: float, kt: int = 16, nt: int = 8, seed: int = 0):
    """A synthetic group: K and N are multiples of 128 (ExLlamaV3 pads), the trellis words random."""
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    k, n = 16 * kt, 16 * nt
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    return trellis, suh, svh


def _reference(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, bits: float, codebook: str,
               x: torch.Tensor) -> torch.Tensor:
    """float64 forward with the same fp16 rounding of the rotated input as the kernel's ``rot_in``."""

    xs = x.double().cpu().numpy() * suh.double().numpy()
    xh = fmt.rotate(xs, -1).astype(np.float16).astype(np.float64)
    wq = fmt.unpack(trellis, bits, codebook).astype(np.float64)
    return torch.from_numpy(fmt.rotate(xh @ wq, -1) * svh.double().numpy())


@pytest.mark.parametrize("codebook,bits", COMBOS)
def test_decode_is_bit_exact(codebook: str, bits: float):
    trellis, _, _ = _tensors(codebook, bits)
    w = linear.unpack_cuda(trellis, codebook)
    ref = torch.from_numpy(fmt.unpack(trellis, bits, codebook)).view(torch.int16)
    assert torch.equal(w.cpu().view(torch.int16), ref)


@pytest.mark.parametrize("codebook,bits", [("mul1", 2), ("mul1", 3), ("mul1", 6), ("3inst", 4), ("mcg", 8)])
@pytest.mark.parametrize("split", [(1, 4), (2, 4), (8, 8), (2, 8)])
def test_row_invariance(codebook: str, bits: float, split: tuple[int, int]):
    """Rows 1..128 identical in every window and under every split of K (the contract's 1, 2, 3, 16, 17, 64, 128)."""

    trellis, suh, svh = _tensors(codebook, bits, kt=64)          # K = 1024: every split divides
    layer = linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook)
    kt = trellis.shape[0]
    if kt % (split[0] * split[1]):
        pytest.skip(f"K/16 = {kt} does not divide over split {split}")
    layer.split = split
    x = torch.randn((128, layer.k), device="cuda").half()
    whole = layer(x)
    for m in ROWS:
        window = layer(x[:m])
        assert torch.equal(window, whole[:m]), f"rows 1..{m}"
        for r in sorted({0, m // 2, m - 1}):
            assert torch.equal(layer(x[r:r + 1])[0], whole[r]), f"row {r} alone vs {m}-row window"


@pytest.mark.parametrize("codebook,bits", [("mul1", 2), ("mul1", 4), ("mul1", 6), ("3inst", 8), ("mcg", 4)])
def test_matches_the_float64_reference(codebook: str, bits: float):
    trellis, suh, svh = _tensors(codebook, bits)
    layer = linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook)
    x = torch.randn((17, layer.k), device="cuda").half()
    got = layer(x, out_dtype=torch.float32)
    want = _reference(trellis, suh, svh, bits, codebook, x)
    assert ((got.double().cpu() - want).norm() / want.norm()).item() < 2e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_input_and_output_dtypes(dtype):
    trellis, suh, svh = _tensors("mul1", 4)
    layer = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    x = torch.randn((8, layer.k), device="cuda").to(dtype)
    half = layer(x.half(), out_dtype=torch.float32)
    got = layer(x, out_dtype=torch.float32)
    assert torch.allclose(got, half, rtol=2e-3, atol=2e-3)
    assert layer(x).dtype == dtype and layer(x, out_dtype=torch.bfloat16).dtype == torch.bfloat16


def test_bias_and_the_scale_words_of_older_checkpoints():
    trellis, suh, svh = _tensors("mcg", 4)
    bias = torch.from_numpy((np.random.default_rng(2).standard_normal(svh.numel()) * 0.05).astype(np.float16))
    plain = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mcg")
    with_bias = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mcg", bias=bias)
    x = torch.randn((3, plain.k), device="cuda").half()
    assert torch.allclose((with_bias(x, out_dtype=torch.float32) - plain(x, out_dtype=torch.float32)).cpu(),
                          bias.float(), rtol=2e-2, atol=2e-2)
    def pack(scales: torch.Tensor) -> torch.Tensor:            # bit b of word w set means element 16w + b is -1
        neg = (np.asarray(scales) < 0).reshape(-1, 16).astype(np.uint16)
        words = (neg * (1 << np.arange(16, dtype=np.uint16))).sum(1).astype(np.uint16)
        return torch.from_numpy(words.view(np.int16).copy())

    packed_su, packed_sv = pack(suh), pack(svh)
    signs = linear.Exl3Linear.from_tensors(trellis, packed_su, packed_sv, "mcg")
    assert torch.equal(signs.suh.cpu(), torch.from_numpy(fmt.unpack_signs(packed_su)))
    assert torch.equal(signs.svh.cpu(), torch.from_numpy(fmt.unpack_signs(packed_sv)))


def test_stored_layout_gives_the_same_bits():
    trellis, suh, svh = _tensors("mul1", 3)
    strips = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1", layout="strips")
    stored = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1", layout="stored")
    assert stored.words.shape != strips.words.shape or not torch.equal(stored.words, strips.words)
    x = torch.randn((5, strips.k), device="cuda").half()
    assert torch.equal(strips(x), stored(x))
    assert torch.equal(strips.unpack().cpu().view(torch.int16), stored.unpack().cpu().view(torch.int16))


def test_the_plan_depends_on_the_shape_only():
    assert linear.plan(4096, 12288) == linear.plan(4096, 12288)
    assert linear.plan(4096, 12288) == (1, 8)                            # 96 blocks: one per program, eight warps
    assert linear.plan(8192, 4096)[0] > 1                                # 32 blocks: split K to fill the SMs
    assert linear.plan(4096, 152576) == (1, 8)                           # plenty of programs already
    assert linear.plan(4096, 2048)[0] >= linear.plan(8192, 4096)[0]      # narrower still: at least as many splits
    for k, n in ((128, 128), (256, 256), (4096, 12288), (8192, 4096), (4096, 152576), (4096, 2048)):
        sk, wk = linear.plan(k, n)
        per_warp = (k // 16) // (sk * wk)
        assert (k // 16) % (sk * wk) == 0 and per_warp >= 1
        assert per_warp >= 8 or (k // 16) < 8 * 8              # only a tiny layer may go below eight tiles
