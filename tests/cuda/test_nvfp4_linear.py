"""NVFP4 and FP8 linears: the decode matmul equals the dequantized weight's product (exact weights, fp32 sums) with rows
independent of the row count; the prompt GEMM tracks it through e4m3 staging and keeps a row's bits in any chunk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels.qmm import quantize_rows
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4.linear import Concat, Fp4Linear, Fp8BlockLinear, Fp8Linear, Mx8Linear


def _fp4(n, k, seed):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scale = rng.integers(0x20, 0x50, size=(n, k // 16), dtype=np.uint8)           # e4m3 0.03-4
    return packed, scale, 0.0123


def _fp8(n, k, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30                                                 # finite, below 2^7
    return w, 0.0371


def _check_rows(lin, x):
    full = lin(x)
    for rows in (1, 2, 3, 5, 12, 16):
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    return full


@pytest.mark.parametrize("n,k", [(128, 256), (320, 1024), (1000, 5120)])
def test_fp4_decode_is_the_dequantized_product_and_rows_are_independent(n, k):
    packed, scale, g = _fp4(n, k, n)
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g)
    x = (torch.randn((16, k), generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16).cuda()
    full = _check_rows(lin, x)
    ref = x.double() @ torch.from_numpy(fmt.dequant("nvfp4", packed, scale, g)).double().cuda().t()
    err = ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-2, err


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120)])
def test_fp8_decode_is_the_dequantized_product_and_rows_are_independent(n, k):
    w, s = _fp8(n, k, n)
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s)
    x = (torch.randn((16, k), generator=torch.Generator().manual_seed(2)) * 0.5).to(torch.bfloat16).cuda()
    full = _check_rows(lin, x)
    ref = x.double() @ torch.from_numpy(fmt.dequant("fp8", w, np.array([s], np.float32))).double().cuda().t()
    err = ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-2, err


def test_prompt_gemms_track_decode_and_keep_rows_in_any_chunk():
    n, k, m = 320, 1024, 300
    packed, scale, g = _fp4(n, k, 7)
    fp4 = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g)
    w, s = _fp8(n, k, 8)
    fp8 = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s)
    x = (torch.randn((m, k), generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16).cuda()
    for lin, bound in ((fp4, 0.06), (fp8, 0.04)):
        want = lin(x).float()
        got = lin.prefill(quantize_rows(x))
        assert float((got.float() - want).norm() / want.norm()) < bound
        parts = [lin.prefill(quantize_rows(x[a:b].contiguous())) for a, b in ((0, 1), (1, 130), (130, 300))]
        assert torch.equal(torch.cat(parts), got)


def _mx8(n, k, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    return w, rng.integers(118, 132, size=(n, k // 32), dtype=np.uint8)            # e8m0 2^-9 .. 2^4


@pytest.mark.parametrize("n,k", [(128, 256), (320, 2560)])
def test_mxfp8_decode_is_exact_and_prompts_track_it_in_any_chunk(n, k):
    w, s = _mx8(n, k, n)
    lin = Mx8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), torch.from_numpy(s).cuda())
    x = (torch.randn((16, k), generator=torch.Generator().manual_seed(5)) * 0.5).to(torch.bfloat16).cuda()
    full = _check_rows(lin, x)
    ref = x.double() @ torch.from_numpy(fmt.dequant("mxfp8", w, s)).double().cuda().t()
    assert ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item() < 1e-2
    xp = (torch.randn((300, k), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16).cuda()
    want, got = lin(xp).float(), lin.prefill(xp)
    assert float((got.float() - want).norm() / want.norm()) < 0.04
    parts = [lin.prefill(xp[a:b].contiguous()) for a, b in ((0, 1), (1, 130), (130, 300))]
    assert torch.equal(torch.cat(parts), got)


def test_mxfp8_stack_keeps_each_projection():
    a, b = (Mx8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), torch.from_numpy(s).cuda())
            for w, s in (_mx8(96, 256, 1), _mx8(48, 256, 2)))
    st = Mx8Linear.stack([a, b])
    x = (torch.randn((5, 256), generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).cuda()
    assert st.n == 144 and torch.allclose(st(x).float(), torch.cat([a(x), b(x)], 1).float(), rtol=1e-2, atol=1e-2)


def _fp8b(n, k, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    s = (rng.random((-(-n // 128), k // 128)) * 1e-2 + 1e-4).astype(np.float32)   # arbitrary fp32 block scales
    return w, s


@pytest.mark.parametrize("n,k", [(128, 256), (200, 512), (320, 2560)])
def test_block_fp8_decode_is_exact_and_prompts_track_it_in_any_chunk(n, k):
    w, s = _fp8b(n, k, n)
    lin = Fp8BlockLinear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn),
                                         torch.from_numpy(s).cuda())
    x = (torch.randn((16, k), generator=torch.Generator().manual_seed(8)) * 0.5).to(torch.bfloat16).cuda()
    full = _check_rows(lin, x)
    ref = x.double() @ torch.from_numpy(fmt.dequant("fp8block", w, s)).double().cuda().t()
    assert ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item() < 1e-2
    xp = (torch.randn((300, k), generator=torch.Generator().manual_seed(9)) * 0.5).to(torch.bfloat16).cuda()
    want, got = lin(xp).float(), lin.prefill(xp)
    assert float((got.float() - want).norm() / want.norm()) < 0.04
    parts = [lin.prefill(xp[a:b].contiguous()) for a, b in ((0, 1), (1, 130), (130, 300))]
    assert torch.equal(torch.cat(parts), got)


def test_block_fp8_and_bf16_concat_keeps_each_projection():
    from tensorfold.families.qwen4_exp.cuda.bf16 import b16_from_rows, matmul

    w, s = _fp8b(200, 512, 3)
    a = Fp8BlockLinear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn),
                                       torch.from_numpy(s).cuda())
    b = b16_from_rows((torch.randn((48, 512), generator=torch.Generator().manual_seed(4)) * 0.05)
                      .to(torch.bfloat16).cuda())
    st = Concat([a, b])
    for m in (1, 5, 300):
        x = (torch.randn((m, 512), generator=torch.Generator().manual_seed(m)) * 0.5).to(torch.bfloat16).cuda()
        out = torch.empty((m, st.n), dtype=torch.bfloat16, device="cuda")
        st(x, out)
        assert torch.equal(out, torch.cat([a(x), matmul(x, b)], 1))
        assert torch.equal(st.prefill(x), torch.cat([a.prefill(x), matmul(x, b)], 1))
