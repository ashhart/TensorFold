"""EXL3 routed experts in prompt windows (experts_prompt.cuh) give every pair the grouping kernel's bits.

Every codebook, mixed and 4-bit-only widths, skewed routing (experts spanning several items), skipped picks, with and
without the weighted combine, windows on both sides of the switch row count; decode windows keep the grouping kernel."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _trellis(k, n, k2, gen):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, gen):
    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().cuda()


def _layer(E, D, I, kfun, cb, seed):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = kfun(e)
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb)


def _skewed_picks(E, R, k, gen):
    """Top-k of a strongly biased router (a few experts take most rows: many member tiles), the last slot the shared
    expert (id E - 1 here), and every 7th row's slot 0 a skipped pick (id E)."""

    bias = torch.linspace(4.0, -4.0, E - 1)
    logits = torch.randn((R, E - 1), generator=gen) + bias
    sel = logits.topk(k, dim=1).indices.to(torch.int32)
    sel = torch.cat([sel, torch.full((R, 1), E - 1, dtype=torch.int32)], 1)
    sel[::7, 0] = E
    w = torch.rand((R, k + 1), generator=gen) * 0.2 + 0.05
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


CASES = [
    ("mul1-4bit", 2, lambda e: (8, 8, 8)),                                                  # EXL3 4.05's experts
    ("mul1-mixed", 2, lambda e: ((6, 6, 6) if e % 4 else (12, 12, 12))),                     # 3.05-like + 6-bit shared
    ("mcg", 1, lambda e: ((2, 4, 6, 8, 10, 12, 14, 16)[e % 8],) * 2 + ((4, 6, 8)[e % 3],)),
    ("3inst", 0, lambda e: ((4, 8, 16)[e % 3],) * 3),
]


@pytest.mark.parametrize("name,cb,kfun", CASES, ids=[c[0] for c in CASES])
def test_prompt_kernel_gives_the_grouping_kernels_bits(name, cb, kfun, monkeypatch):
    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, ROWS = 33, 512, 256, 6, 300
    ex = _layer(E, D, I, kfun, cb, seed=5 + cb)
    g = torch.Generator().manual_seed(23 + cb)
    x = (torch.randn((ROWS, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    sel, w = _skewed_picks(E, ROWS, TOPK, g)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)

    def run(R, mode, wts):
        monkeypatch.setattr(experts, "PROMPT", mode)
        scratch.y.fill_(float("nan"))                  # no row may pass on what another run left
        scratch.z.fill_(float("nan"))
        got = experts.routed(x[:R], sel[:R].contiguous(), w[:R].contiguous() if wts else None, ex, scratch, None, R)
        return got.clone()

    for R in (65, 129, 300):
        for wts in (False, True):
            ref = run(R, "group", wts)                 # the grouping kernel, one program a (expert, member tile)
            got = run(R, "prompt", wts)
            if not wts:                                # skipped slots stay NaN in both; every routed slot is set
                assert torch.isfinite(ref.view(R, TOPK + 1, D)[sel[:R] < E]).all()
            assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), (name, R, wts)


def test_prompt_rows_do_not_depend_on_the_window(monkeypatch):
    """A row's output in a prompt window equals its output alone (one row, the grouping kernel), for 4.05's widths."""

    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, ROWS = 33, 512, 256, 6, 200
    ex = _layer(E, D, I, lambda e: (12, 12, 12) if e == E - 1 else (8, 8, 8), 2, seed=3)
    g = torch.Generator().manual_seed(4)
    x = (torch.randn((ROWS, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    sel, _ = _skewed_picks(E, ROWS, TOPK, g)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)
    monkeypatch.setattr(experts, "PROMPT", "prompt")
    full = experts.routed(x, sel, None, ex, scratch, None, ROWS).clone().view(ROWS, TOPK + 1, D)
    for r in (0, 1, 77, 199):
        one = experts.routed(x[r:r + 1], sel[r:r + 1].contiguous(), None, ex, scratch, None, 1).view(TOPK + 1, D)
        keep = sel[r] < E
        assert torch.equal(full[r][keep].view(torch.int32), one[keep].view(torch.int32)), r


def test_decode_windows_keep_the_grouping_kernel(monkeypatch):
    """At or below PROMPT_ROWS rows (decode, graphs) routed() takes the grouping path: no plan is made."""

    from tensorfold.cuda.exl3 import experts

    monkeypatch.setattr(experts, "PROMPT", "prompt")
    ex = _layer(9, 512, 256, lambda e: (8, 8, 8), 2, seed=1)
    scratch = experts.Scratch(ex, 64, 3)
    x = torch.randn((8, 512)).to(torch.bfloat16).cuda()
    sel = torch.randint(0, 9, (8, 3), dtype=torch.int32).cuda()
    experts.routed(x, sel, None, ex, scratch, None, 8)
    assert scratch._plan is None


def test_prompt_bf16_y_is_the_rounded_fp32_y(monkeypatch):
    """y_out (bf16, the prompt path's per-slot Y) equals the grouping kernel's fp32 Y copied into bf16."""

    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, ROWS = 33, 512, 256, 6, 160
    ex = _layer(E, D, I, lambda e: (12, 12, 12) if e == E - 1 else (8, 8, 8), 2, seed=8)
    g = torch.Generator().manual_seed(9)
    x = (torch.randn((ROWS, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    logits = torch.randn((ROWS, E - 1), generator=g) + torch.linspace(4.0, -4.0, E - 1)
    sel = logits.topk(TOPK, dim=1).indices.to(torch.int32)        # every slot routed, no expert twice in a row
    sel = torch.cat([sel, torch.full((ROWS, 1), E - 1, dtype=torch.int32)], 1).cuda().contiguous()
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)
    monkeypatch.setattr(experts, "PROMPT", "group")
    ref32 = experts.routed(x, sel, None, ex, scratch, None, ROWS).clone()
    ref = ref32.to(torch.bfloat16)
    monkeypatch.setattr(experts, "PROMPT", "prompt")
    scratch.y.fill_(float("nan"))
    got32 = experts.routed(x, sel, None, ex, scratch, None, ROWS).clone()
    bad32 = (got32.view(torch.int32) != ref32.view(torch.int32)).sum().item()
    out = torch.full((ROWS * (TOPK + 1), D), float("nan"), dtype=torch.bfloat16, device="cuda")
    got = experts.routed(x, sel, None, ex, scratch, None, ROWS, y_out=out)
    assert got.data_ptr() == out.data_ptr()
    bad = (out.view(torch.int16) != ref.view(torch.int16))
    rows = bad.any(dim=1).nonzero().flatten().tolist()
    assert bad32 == 0 and not rows, (bad32, int(bad.sum()), rows[:20], len(rows))


@pytest.mark.parametrize("n,k", [(324, 10240), (10240, 320), (96, 2560), (2560, 2560), (512, 2560), (1, 2560)])
def test_f16_prompt_tiles_give_the_sliced_bits(n, k, monkeypatch):
    """The EXL3 pack's fp16 matmul at prompt row counts (larger tiles; slices summed in one program) equals the 16-row
    tiles with fp32 slices and the in-order reduce, row for row."""

    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    g = torch.Generator().manual_seed(n + k)
    w = (torch.randn((n, k), generator=g) * 0.02).half()
    sc = exl3_mm.Scratch(11)
    lin = exl3_mm.f16(sc, [w], "cuda")
    sc.part = torch.empty((max(1, lin.sk) * exl3_mm.ROWS * n,), dtype=torch.float32, device="cuda")
    for m in (65, 300, 2048):
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        for dt in (torch.float32, torch.bfloat16):
            monkeypatch.setattr(exl3_mm, "FUSED_MIN", 0)
            ref = lin(x, torch.empty((m, n), dtype=dt, device="cuda")).clone()
            monkeypatch.setattr(exl3_mm, "FUSED_MIN", 64)
            got = lin(x, torch.full((m, n), float("nan"), dtype=dt, device="cuda"))
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (n, k, m, dt)
