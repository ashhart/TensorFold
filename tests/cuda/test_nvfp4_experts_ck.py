"""Grouped NVFP4 experts in the checkpoint's math: rows quantized once under the layer's input scale, each pair the
fp64 product of its quantized row and its expert's stored weights, SiLU(gate) * up handed to down as NVFP4 rows under
the expert's own down input scale, and a pair's bits the same alone or among any other pairs, in either plan."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("the block-scaled FP4 mma needs an SM 12.x GPU", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import format as fmt

E, D, NI, TOP = 6, 256, 128, 3
ACT = 0.0123                                              # gate and up: one input scale for every expert


def _proj(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(E, n, k // 2), dtype=np.uint8)
    scale = rng.integers(0x20, 0x50, size=(E, n, k // 16), dtype=np.uint8)           # e4m3 0.03-4
    glob = (rng.random(E) * 0.02 + 0.005).astype(np.float32)
    return packed, scale, glob


@pytest.fixture(scope="module")
def layer():
    gate, up, down = _proj(NI, D, 1), _proj(NI, D, 2), _proj(D, NI, 3)
    act_d = (np.random.default_rng(4).random(E) * 0.01 + 0.006).astype(np.float32)   # down: one an expert, unsaturated
    acts = (np.full(E, ACT, np.float32), np.full(E, ACT, np.float32), act_d)
    ex = nvx.make_ck(*[tuple(torch.from_numpy(t).cuda() for t in p) for p in (gate, up, down)],
                     tuple(torch.from_numpy(a) for a in acts))
    return ex, gate, up, down, act_d


def _dense(p, e: int) -> torch.Tensor:
    w, s, g = p
    return torch.from_numpy(fmt.dequant("nvfp4", w[e], s[e], float(g[e]))).double().cuda()


def _rows_dq(codes: torch.Tensor, scales: torch.Tensor, act: float) -> torch.Tensor:
    """NVFP4 rows (codes [M, K/2], scales [K/64, mpad, 4]) -> their values in fp64 (x = code x e4m3 x act)."""

    m, k = codes.shape[0], codes.shape[1] * 2
    e2m1 = torch.tensor(fmt.E2M1.tolist(), dtype=torch.float64, device=codes.device)
    vals = torch.stack([e2m1[(codes & 0xF).long()], e2m1[(codes >> 4).long()]], -1).view(m, k)
    sf = scales[:, :m].permute(1, 0, 2).reshape(m, k // 16).view(torch.float8_e4m3fn).double()
    return vals * sf.repeat_interleave(16, 1) * float(np.float32(act))


def _ref_codes(h: torch.Tensor, act: float) -> torch.Tensor:
    """fp32 rows -> NVFP4 code bytes and e4m3 scale bytes [M, K/16] as the epilogue quantizes them (in torch)."""

    g = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).to(h.device)
    m, k = h.shape
    blocks = h.float().view(m, k // 16, 16)
    sf = (g * (blocks.abs().amax(-1) * torch.tensor(1.0 / 6.0, dtype=torch.float32))).clamp(max=448.0)
    sf = sf.to(torch.float8_e4m3fn).float()
    v = blocks * torch.where(sf != 0, g / sf, torch.zeros_like(sf))[..., None]
    a = v.abs()
    edges = [(0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)]
    code = torch.zeros_like(a, dtype=torch.int32)
    for c, (edge, inclusive) in enumerate(edges):
        code = torch.where(a > edge if inclusive else a >= edge, torch.full_like(code, c + 1), code)
    code = torch.where((v < 0) & (code > 0), code | 8, code).view(m, k)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8), sf.to(torch.float8_e4m3fn).view(torch.uint8)


def _picks(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    routed = torch.stack([torch.randperm(E, generator=g)[:TOP] for _ in range(rows)])
    return torch.cat([routed, torch.full((rows, 1), E)], dim=1).to(torch.int32).cuda()


def _plan(picks: torch.Tensor, tile: int = 0) -> grouped.Plan:
    plan = grouped.Plan(picks.shape[0], picks.shape[1], E + 1, "cuda", prefill=tile > 0)
    grouped.route(picks.contiguous(), plan, tile or grouped.TILE)
    return plan


def _run(ex, x, picks, tile=0, dtype=torch.float32):
    plan = _plan(picks, tile)
    rows = x.shape[0]
    codes, scales = nvx.gate_up_ck(x, ex, plan, rows, skip=E)
    y = torch.zeros((rows * (TOP + 1), D), dtype=dtype, device="cuda")
    nvx.down_ck((codes, scales), ex, plan, y, rows, skip=E)
    return codes.view(rows, TOP + 1, NI // 2), scales, y.view(rows, TOP + 1, D)


def _x(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, D), generator=g) * 0.7
    x[:, ::97] *= 12.0                                                            # outliers past the calibrated range
    return x.to(torch.bfloat16).cuda()


def test_gate_up_is_each_pairs_swiglu_quantized_under_its_experts_down_scale(layer):
    ex, gate, up, _, act_d = layer
    x, picks = _x(37, 5), _picks(37, 6)
    codes, scales, _ = _run(ex, x, picks)
    xq = checkpoint.quant4(x, ACT)
    xv = _rows_dq(xq.codes, xq.scales, 1.0)                                      # code x e4m3: the kernel's chain
    flips = sflips = total = 0
    for r in range(37):
        for s in range(TOP):
            e, p = int(picks[r, s]), r * (TOP + 1) + s
            gv = (xv[r] @ _dense(gate, e).t()) * float(np.float32(ACT))
            uv = (xv[r] @ _dense(up, e).t()) * float(np.float32(ACT))
            want, want_sf = _ref_codes((gv * torch.sigmoid(gv) * uv).float()[None], float(act_d[e]))
            got_sf = scales[:, p].reshape(-1)
            flips += int((codes[r, s] != want[0]).sum())
            sflips += int((got_sf != want_sf[0]).sum())
            total += NI // 2
    # bytes differ only where the fp32 sums (mma order vs fp64 here) sit on an e2m1 or e4m3 edge
    assert flips / total < 0.01 and sflips / total < 0.01, (flips / total, sflips / total)


def test_down_is_each_pairs_product_and_bf16_is_fp32_rounded(layer):
    ex, _, _, down, act_d = layer
    x, picks = _x(29, 7), _picks(29, 8)
    codes, scales, y = _run(ex, x, picks)
    assert not y[:, TOP].any()                                                   # the shared slot untouched
    for r in (0, 11, 28):
        for s in range(TOP):
            e, p = int(picks[r, s]), r * (TOP + 1) + s
            hv = _rows_dq(codes[r, s][None], scales[:, p:p + 1], float(act_d[e]))
            want = (hv @ _dense(down, e).t())[0]
            err = ((y[r, s].double() - want).abs() / (want.abs() + want.abs().mean())).max().item()
            assert err < 1e-5, (r, s, err)
    _, _, yb = _run(ex, x, picks, tile=16, dtype=torch.bfloat16)
    assert torch.equal(yb[:, :TOP], y[:, :TOP].to(torch.bfloat16))


@pytest.mark.parametrize("tile", [0, 16, 64])
def test_pairs_keep_their_bits_alone_and_in_any_plan(layer, tile):
    ex = layer[0]
    x, picks = _x(70, 9), _picks(70, 10)
    codes, _, y = _run(ex, x, picks)
    c2, _, y2 = _run(ex, x, picks, tile=tile)
    assert torch.equal(c2[:, :TOP], codes[:, :TOP]) and torch.equal(y2[:, :TOP], y[:, :TOP])
    for r in (0, 33, 69):
        c1, _, y1 = _run(ex, x[r:r + 1].contiguous(), picks[r:r + 1])
        assert torch.equal(c1[0, :TOP], codes[r, :TOP]) and torch.equal(y1[0, :TOP], y[r, :TOP])


def test_make_ck_takes_the_largest_gate_up_input_scale_and_each_down_scale():
    gate, up, down = _proj(64, 64, 11), _proj(64, 64, 12), _proj(64, 64, 13)
    a_g = np.full(E, 0.01, np.float32)
    a_u = a_g.copy()
    a_u[2] = 0.02
    a_d = np.linspace(0.001, 0.006, E).astype(np.float32)
    ex = nvx.make_ck(*[tuple(torch.from_numpy(t).cuda() for t in p) for p in (gate, up, down)],
                     (torch.from_numpy(a_g), torch.from_numpy(a_u), torch.from_numpy(a_d)))
    assert ex.act == float(np.float32(0.02))
    want = torch.from_numpy(np.float32(0.02) * gate[2])
    assert torch.equal(ex.alpha[0].cpu(), want)
    assert torch.equal(ex.alpha[2].cpu(), torch.from_numpy(a_d * down[2]))
    assert torch.equal(ex.down_inv.cpu(), torch.from_numpy(np.float32(1.0) / a_d))
