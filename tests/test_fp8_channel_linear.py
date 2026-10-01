"""Channel-scaled FP8 projections (compressed-tensors ``channel`` FP8) on the CUDA kernels against a dequantized reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.torch, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def _case(n=200, k=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.5).to(torch.float8_e4m3fn)
    cs = torch.rand(n, 1, generator=g) * 0.02 + 0.001
    return w.cuda(), cs.cuda()


def test_channel_fp8_decode_and_prompt_match_reference():
    from tensorfold.cuda.nvfp4.linear import Fp8ChannelLinear

    w, cs = _case()
    lin = Fp8ChannelLinear.from_checkpoint(w, cs)
    ref_w = (w.float() * cs).to(torch.float32)
    for rows in (1, 3, 16, 64):
        x = torch.randn(rows, w.shape[1], device="cuda").to(torch.bfloat16)
        want = x.float() @ ref_w.t()
        for got in (lin(x), lin.prefill(x)):
            assert got.shape == (rows, w.shape[0]) and got.dtype == torch.bfloat16
            err = (got.float() - want).abs().max() / want.abs().max()
            assert err < 2e-2


def test_channel_fp8_rows_are_independent_of_the_call():
    from tensorfold.cuda.nvfp4.linear import Fp8ChannelLinear

    w, cs = _case(seed=1)
    lin = Fp8ChannelLinear.from_checkpoint(w, cs)
    x = torch.randn(12, w.shape[1], device="cuda").to(torch.bfloat16)
    many = lin(x)
    for r in range(12):
        assert torch.equal(lin(x[r:r + 1])[0], many[r])


def test_channel_fp8_tiles_and_out():
    from tensorfold.cuda.nvfp4.linear import Fp8ChannelLinear

    w, cs = _case(n=256, seed=2)
    lin = Fp8ChannelLinear.from_checkpoint(w, cs)
    x = torch.randn(2, w.shape[1], device="cuda").to(torch.bfloat16)
    full = lin(x)
    part = lin.tiles(1, 3)
    assert part.n == 128 and torch.equal(part(x), full[:, 64:192])
    out = torch.empty_like(full)
    assert lin(x, out) is out and torch.equal(out, full)
    with pytest.raises(ValueError):
        Fp8ChannelLinear.from_checkpoint(w, cs[:-1])
