"""Grouped block-FP8 experts: the dense fp32 reference's values, and a pair's bits whatever else the call holds."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def fp8(e: int, n: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    w = (torch.randn(e, n, k, generator=g) * 2).clamp(-400, 400).to(torch.float8_e4m3fn)
    s = torch.rand(e, n // 128, k // 128, generator=g) * 0.01 + 0.001
    return w.cuda(), s.cuda()


def setup(rows: int, e: int = 6, width: int = 256, dims: int = 512, slots: int = 3, seed: int = 0):
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.fp8 import experts as fp8x

    g = torch.Generator().manual_seed(seed)
    gate, up, down = fp8(e, width, dims, g), fp8(e, width, dims, g), fp8(e, dims, width, g)
    ex = fp8x.make(gate, up, down)
    x = (torch.randn(rows, dims, generator=g) * 0.5).to(torch.bfloat16).cuda()
    picks = torch.stack([torch.randperm(e, generator=g)[:slots] for _ in range(rows)]).to(torch.int32).cuda()
    return grouped, fp8x, ex, (gate, up, down), x, picks


def run(grouped, fp8x, ex, x, picks, prefill: bool = False):
    rows, slots = picks.shape
    plan = grouped.Plan(rows, slots, ex.count, x.device, prefill=prefill)
    grouped.route(picks.contiguous(), plan, 16)
    act = torch.empty((rows * slots, ex.width), dtype=torch.bfloat16, device=x.device)
    fp8x.gate_up(x, ex, plan, act, rows)
    y = torch.empty((rows * slots, ex.dims), dtype=torch.float32, device=x.device)
    fp8x.down(act, ex, plan, y, rows)
    return act.view(rows, slots, -1), y.view(rows, slots, -1)


def test_fp8_experts_match_the_dense_reference():
    grouped, fp8x, ex, (gate, up, down), x, picks = setup(37)
    act, y = run(grouped, fp8x, ex, x, picks)
    dg, du, dd = (fp8x.dense(*m) for m in (gate, up, down))
    for r in range(x.shape[0]):
        for s, e in enumerate(picks[r].tolist()):
            gv = (x[r].float() @ dg[e].T).bfloat16().float()
            uv = (x[r].float() @ du[e].T).bfloat16().float()
            ref_act = ((gv * torch.sigmoid(gv)).bfloat16().float() * uv).bfloat16()
            torch.testing.assert_close(act[r, s].float(), ref_act.float(), rtol=2e-2, atol=2e-2)
            ref_y = act[r, s].float() @ dd[e].T                         # the kernel's own activation, so down alone
            torch.testing.assert_close(y[r, s], ref_y, rtol=1e-3, atol=1e-3)


def test_a_pairs_bits_never_depend_on_the_other_rows():
    grouped, fp8x, ex, _, x, picks = setup(40, seed=1)
    act, y = run(grouped, fp8x, ex, x, picks)
    for r in (0, 7, 39):
        a1, y1 = run(grouped, fp8x, ex, x[r:r + 1].contiguous(), picks[r:r + 1])
        assert torch.equal(act[r], a1[0]) and torch.equal(y[r], y1[0])
    ap, yp = run(grouped, fp8x, ex, x, picks, prefill=True)            # a prompt plan's grouping, same bits
    assert torch.equal(act, ap) and torch.equal(y, yp)


def test_unaligned_blocks_are_refused():
    from tensorfold.cuda.fp8 import experts as fp8x

    w = torch.zeros(2, 96, 256, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(ValueError, match="multiples of 128"):
        fp8x.pack(w)
