"""The EXL3 linear's walks and load paths (``linear.cu``'s ``loads``: the transposed mma walk, 16-byte weight loads,
PDL launches) give the bits of the original walk (loads 0) for every row window 1..16 (and longer, in 16-row passes)
and split plan, and every row's bits alone (row invariance), on GLM-5.3's 5-bit mul1 shapes and a few other widths."""

from __future__ import annotations

import pytest
import torch

from tensorfold.cuda.exl3 import linear

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

PATHS = (1, 3, 32, 33, 35)       # linear.LOADS values other than the original walk (0)


def _layer(k: int, n: int, bits: float, codebook: str = "mul1", seed: int = 0) -> linear.Exl3Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    tr = torch.randint(-2**15, 2**15, (k // 16, n // 16, int(16 * bits)), dtype=torch.int16, device="cuda",
                       generator=g)
    suh = (torch.randint(0, 2, (k,), device="cuda", generator=g) * 2 - 1).half()
    svh = (torch.randn(n, device="cuda", generator=g) * 0.05).half()
    return linear.Exl3Linear.from_tensors(tr, suh, svh, codebook)


def _run(lin: linear.Exl3Linear, x: torch.Tensor, loads: int, split: tuple[int, int]) -> torch.Tensor:
    lin.loads, lin.split = loads, split
    y = lin(x, out_dtype=torch.float32)
    torch.cuda.synchronize()
    return y


@pytest.mark.parametrize("k,n,split", [(6144, 2048, (8, 4)), (6144, 2048, (1, 8)), (2048, 4096, (4, 4)),
                                       (6144, 640, (8, 8)), (4096, 6144, (2, 4)), (512, 6144, (1, 8)),
                                       (6144, 512, (16, 4)), (3072, 6144, (1, 8)), (1024, 256, (2, 2))])
def test_glm_shapes_same_bits(k: int, n: int, split: tuple[int, int]):
    lin = _layer(k, n, 5)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    alone = torch.cat([_run(lin, x[r:r + 1], 0, split) for r in range(16)])
    for loads in (0,) + PATHS:
        for rows in range(1, 17):
            y = _run(lin, x[:rows], loads, split)
            assert torch.equal(y.view(torch.int32), alone[:rows].view(torch.int32)), (loads, rows)


@pytest.mark.parametrize("codebook,bits", [("mul1", 3), ("mul1", 3.5), ("mul1", 4), ("mul1", 6), ("3inst", 5),
                                           ("mcg", 6), ("mul1", 8), ("mul1", 1), ("mul1", 2)])
def test_widths_same_bits(codebook: str, bits: float):
    lin = _layer(1024, 512, bits, codebook, seed=1)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.float16)
    for split in ((1, 4), (4, 2), (2, 8)):
        ref = _run(lin, x, 0, split)
        for loads in PATHS:
            assert torch.equal(_run(lin, x, loads, split).view(torch.int32), ref.view(torch.int32)), (split, loads)


def test_long_windows_wide_range():
    """Rows of very different magnitudes (fp32 accumulation rounding at work), bf16 out, windows past one pass."""

    lin = _layer(2048, 1024, 5, seed=2)
    scale = torch.exp(torch.randn(128, 1, device="cuda") * 3) * torch.exp(torch.randn(1, 2048, device="cuda"))
    x = (torch.randn(128, 2048, device="cuda") * scale).clamp(-3e4, 3e4).to(torch.bfloat16)
    for split in ((4, 4), (1, 8)):
        lin.split = split
        lin.loads = 0
        alone = torch.cat([lin(x[r:r + 1]) for r in range(128)])
        for loads in PATHS:
            lin.loads = loads
            for rows in (5, 8, 9, 16, 17, 40, 128):
                y = lin(x[:rows])
                assert torch.equal(y.view(torch.int16), alone[:rows].view(torch.int16)), (split, loads, rows)


@pytest.mark.parametrize("loads", (0,) + PATHS)
def test_group_same_bits(loads: int):
    """q_a + kv_a, gate + up (+ a third) in one launch: each output the bits of the layer's own call."""

    layers = [_layer(6144, 2048, 5, seed=3), _layer(6144, 640, 5, seed=4), _layer(6144, 512, 5, seed=5)]
    for lin, sk in zip(layers, (8, 16, 2)):
        lin.split, lin.loads = (sk, 4), loads
    x = torch.randn(16, 6144, device="cuda", dtype=torch.bfloat16)
    for n in (2, 3):
        for rows in (1, 3, 8, 9, 16):
            alone = [lin(x[:rows]) for lin in layers[:n]]
            outs = [torch.empty_like(a) for a in alone]
            assert linear.groupable(layers[:n])
            linear.group(layers[:n], x[:rows], outs)
            for a, o in zip(alone, outs):
                assert torch.equal(a.view(torch.int16), o.view(torch.int16)), (n, rows)


def test_glm_group_tuner_and_lins():
    """GLM-5.3's start-up group tuner gives a group one warps-a-block (or leaves it), and the decode path's ``lins``
    then gives every output the bits of the layer's own call."""

    from tensorfold.families.glm_moe_dsa.cuda import fused

    gs = [[_layer(6144, 2048, 5, seed=10 + i), _layer(6144, 640, 5, seed=20 + i)] for i in range(3)]
    for g in gs:
        g[0].split, g[1].split = (8, 4), (8, 8)          # tuned alone: different warps a block
    chosen = fused.tune_groups(gs)
    own, best = chosen[((6144, 2048), (6144, 640))]
    for g in gs:
        assert [lin.split for lin in g] == (best if best is not None else own)
        if best is not None:
            assert linear.groupable(g)
    x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
    for rows in (1, 3, 8):
        for g in gs:
            alone = [lin(x[:rows]) for lin in g]
            outs = [torch.empty_like(a) for a in alone]
            fused.lins(g, None, x[:rows], outs)
            for a, o in zip(alone, outs):
                assert torch.equal(a.view(torch.int16), o.view(torch.int16))
