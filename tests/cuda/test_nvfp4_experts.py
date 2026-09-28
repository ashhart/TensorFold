"""Grouped NVFP4 experts: each pair's output is its expert's product (exact weights, fp32 sums), the same bits alone or
among any other pairs, in decode and prompt plans; the shared expert's pairs are left for the caller."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import format as fmt

E, D, NI, TOP = 6, 256, 128, 3


def _proj(n: int, k: int, g: torch.Generator):
    words = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0x28, 0x40, (E, n, k // 16), generator=g, dtype=torch.uint8)
    glob = torch.rand(E, generator=g) * 0.02 + 0.005
    return words, scales, glob


def _dense(p, e: int) -> torch.Tensor:
    w, s, g = p
    return torch.from_numpy(fmt.dequant("nvfp4", w[e].numpy(), s[e].numpy(), float(g[e]))).double()


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator().manual_seed(11)
    gate, up, down = _proj(NI, D, g), _proj(NI, D, g), _proj(D, NI, g)
    ex = nvx.make(*[tuple(t.cuda() for t in p) for p in (gate, up, down)])
    return ex, gate, up, down


def _plan(picks: torch.Tensor, prefill: bool = False) -> grouped.Plan:
    plan = grouped.Plan(picks.shape[0], picks.shape[1], E + 1, "cuda", prefill=prefill)
    grouped.route(picks.contiguous(), plan)
    return plan


def _picks(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    routed = torch.stack([torch.randperm(E, generator=g)[:TOP] for _ in range(rows)])
    return torch.cat([routed, torch.full((rows, 1), E)], dim=1).to(torch.int32).cuda()


def _gate_up(ex, x, picks, prefill=False):
    plan = _plan(picks, prefill)
    out = torch.zeros((x.shape[0] * (TOP + 1), NI), dtype=torch.bfloat16, device="cuda")
    nvx.gate_up(x, ex, plan, out, x.shape[0], skip=E)
    return out.view(x.shape[0], TOP + 1, NI)


def _down(ex, act, picks, prefill=False, dtype=torch.float32):
    plan = _plan(picks, prefill)
    out = torch.zeros((act.shape[0], D), dtype=dtype, device="cuda")
    nvx.down(act, ex, plan, out, picks.shape[0], skip=E)
    return out.view(picks.shape[0], TOP + 1, D)


def test_blocks_hold_the_checkpoint_weights_exactly(layer):
    ex, gate, up, down = layer
    for e in (0, 5):
        for which, p in (("gate", gate), ("up", up), ("down", down)):
            assert torch.equal(nvx.dense(ex, e, which).cpu(), _dense(p, e).float()), (e, which)


def test_gate_up_is_each_pairs_swiglu_and_pairs_are_independent(layer):
    ex, gate, up, _ = layer
    x = (torch.randn(5, D, generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16).cuda()
    picks = _picks(5, 2)
    act = _gate_up(ex, x, picks)
    assert torch.equal(act[:, TOP], torch.zeros_like(act[:, TOP]))                   # the shared slot untouched
    for r in range(5):
        for s in range(TOP):
            e = int(picks[r, s])
            gv = (x[r].double().cpu() @ _dense(gate, e).t()).float().to(torch.bfloat16).float()
            uv = (x[r].double().cpu() @ _dense(up, e).t()).float().to(torch.bfloat16).float()
            want = ((gv / (1 + torch.exp(-gv))).to(torch.bfloat16).float() * uv)
            got = act[r, s].float().cpu()
            assert float((got - want).abs().max() / want.abs().max()) < 2e-2, (r, s)
    alone = _gate_up(ex, x[2:3].contiguous(), picks[2:3])
    assert torch.equal(alone[0, :TOP], act[2, :TOP])
    assert torch.equal(_gate_up(ex, x, picks, prefill=True)[:, :TOP], act[:, :TOP])


def test_down_is_each_pairs_product_in_fp32_or_bf16(layer):
    ex, _, _, down = layer
    picks = _picks(7, 3)
    act = (torch.randn(7 * (TOP + 1), NI, generator=torch.Generator().manual_seed(4)) * 0.5).to(torch.bfloat16).cuda()
    y = _down(ex, act, picks)
    rows = act.view(7, TOP + 1, NI)
    for r in (0, 3, 6):
        for s in range(TOP):
            want = rows[r, s].double().cpu() @ _dense(down, int(picks[r, s])).t()
            err = float(((y[r, s].double().cpu() - want).abs().max() / want.abs().max()))
            assert err < 1e-4, (r, s, err)
    one = _down(ex, act.view(7, TOP + 1, NI)[3:4].reshape(-1, NI).contiguous(), picks[3:4])
    assert torch.equal(one[0, :TOP], y[3, :TOP])
    yb = _down(ex, act, picks, prefill=True, dtype=torch.bfloat16)
    assert torch.equal(yb[:, :TOP], y[:, :TOP].to(torch.bfloat16))


def test_quantize_round_trips_within_nvfp4_noise():
    w = (torch.randn(3, 64, 128) * 0.02).to(torch.bfloat16).cuda()
    words, scales, glob = nvx.quantize(w)
    back = np.stack([fmt.dequant("nvfp4", words[e].cpu().numpy(), scales[e].cpu().numpy(), float(glob[e]))
                     for e in range(3)])
    rel = np.linalg.norm(back - w.float().cpu().numpy()) / np.linalg.norm(w.float().cpu().numpy())
    assert rel < 0.12, rel
