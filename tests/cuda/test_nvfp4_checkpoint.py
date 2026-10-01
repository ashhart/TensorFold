"""Checkpoint-math kernels: quantizers byte-exact to an fp32 reference, matmuls the fp64 product, rows independent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("the block-scaled FP4 mma needs an SM 12.x GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear

E2M1 = torch.tensor(fmt.E2M1[:8].tolist(), dtype=torch.float32)


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 code (ties to the even code), saturating: the quantizer's rule in torch."""

    a = v.abs()
    edges = [(0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)]
    code = torch.zeros_like(a, dtype=torch.int32)
    for c, (edge, inclusive) in enumerate(edges):
        code = torch.where(a > edge if inclusive else a >= edge, torch.full_like(code, c + 1), code)
    return torch.where((v < 0) & (code > 0), code | 8, code)


def _ref_quant4(x: torch.Tensor, act: float):
    g = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)
    g = g.to(x.device)
    m, k = x.shape
    blocks = x.float().view(m, k // 16, 16)
    amax = blocks.abs().amax(-1)
    sf8 = (g * (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = sf8.float()
    mul = torch.where(sf != 0, g / sf, torch.zeros_like(sf))
    codes = _e2m1_codes(blocks * mul[..., None]).view(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, sf8.view(torch.uint8), sf


def _ref_quant8(x: torch.Tensor, act: float) -> torch.Tensor:
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).to(x.device)
    q = (x.float() * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    m, k = x.shape
    m16 = torch.arange(16, device=x.device)
    src = (m16 // 4) * 2 + (m16 % 4 % 2) + (m16 % 4 // 2) * 8          # byte 4q + j holds input 2q + j%2 + 8(j/2)
    return q.view(m, k // 16, 16)[:, :, src].reshape(m, k)


def _fp4_weight(n, k, seed):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scale = rng.integers(0x20, 0x50, size=(n, k // 16), dtype=np.uint8)           # e4m3 0.03-4
    return packed, scale, 0.0123


def _rows(m, k, seed, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=g) * spread
    x[:, :: 97] *= 12.0                                                           # outliers past the calibrated range
    return x.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("m,k", [(1, 64), (5, 1024), (130, 5120)])
def test_nvfp4_rows_are_the_checkpoint_format_byte_for_byte(m, k):
    x, act = _rows(m, k, m), 0.0123
    codes, scales = checkpoint.quant4(x, act)
    want_codes, want_sf8, _ = _ref_quant4(x, act)
    assert torch.equal(codes, want_codes)
    got_sf = scales[:, :m].permute(1, 0, 2).reshape(m, k // 16)
    assert torch.equal(got_sf, want_sf8)


@pytest.mark.parametrize("m,k", [(1, 64), (7, 2048)])
def test_fp8_rows_are_e4m3_under_the_static_scale_in_fragment_order(m, k):
    x, act = _rows(m, k, 50 + m), 0.0071
    assert torch.equal(checkpoint.quant8(x, act), _ref_quant8(x, act))


def _tiled(rows: torch.Tensor, step: int, swz) -> torch.Tensor:
    """Row-major bytes (M, K') -> the prompt GEMM's tiles [mpad / TB][K/64][TB][step], 16-byte chunks swizzled."""

    tb, (m, kb) = checkpoint.TB, rows.shape
    mpad = -(-m // tb) * tb
    full = torch.zeros((mpad, kb), dtype=torch.uint8, device=rows.device)
    full[:m] = rows
    t = full.view(mpad // tb, tb, kb // step, step // 16, 16)
    out = torch.empty_like(t)
    for r in range(tb):
        for c in range(step // 16):
            out[:, r, :, swz(r, c)] = t[:, r, :, c]
    return out.permute(0, 2, 1, 3, 4).reshape(-1)


@pytest.mark.parametrize("m,k", [(1, 64), (130, 5120)])
def test_tiled_rows_are_the_row_quantizers_bytes_in_the_prompt_gemms_tiles(m, k):
    x = _rows(m, k, 70 + m)
    codes, scales = checkpoint.quant4(x, 0.0123)
    tc, ts = checkpoint.quant4(x, 0.0123, checkpoint.TB)
    assert torch.equal(tc.view(-1), _tiled(codes, 32, lambda r, c: c ^ ((r >> 2) & 1)))
    ts = ts.permute(1, 0, 2, 3).reshape(k // 64, -1, 4)                       # [K/64, mpad, 4], as the rows'
    assert torch.equal(ts[:, :m], scales[:, :m]) and not ts[:, m:].any()
    t8 = checkpoint.quant8(x, 0.0071, checkpoint.TB)
    assert torch.equal(t8.view(-1), _tiled(checkpoint.quant8(x, 0.0071), 64, lambda r, c: c ^ ((r >> 1) & 3)))


def _fp4_ref(x, act, packed, scale, g):
    """fp64 product of the quantized rows and the stored weights, scaled as the kernel scales."""

    codes, _, sf = _ref_quant4(x, act)
    m, k = x.shape
    lo, hi = (codes & 0xF).long(), (codes >> 4).long()
    vals = torch.stack([fmt_e2m1(lo), fmt_e2m1(hi)], -1).view(m, k).double()
    xq = vals * sf.double().repeat_interleave(16, 1)
    w = torch.from_numpy(fmt.dequant("nvfp4", packed, scale, 1.0)).double().cuda()
    return (xq @ w.t()) * (np.float32(act) * np.float32(g))


def fmt_e2m1(code: torch.Tensor) -> torch.Tensor:
    return torch.tensor(fmt.E2M1.tolist(), dtype=torch.float64, device=code.device)[code]


ROWS = (1, 2, 3, 5, 12, 16, 17, 31, 33, 64, 65, 100, 128)


@pytest.mark.parametrize("n,k", [(128, 256), (1000, 5120), (320, 17408)])
def test_fp4_lane_is_the_quantized_product_and_rows_are_independent(n, k):
    packed, scale, g = _fp4_weight(n, k, n)
    act = 0.0123
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=act)
    x = _rows(128, k, 3, 0.7)
    full = lin(x)
    for rows in ROWS:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    one = torch.cat([lin(x[r:r + 1].contiguous()) for r in range(0, 128, 9)])
    assert torch.equal(one, full[0:128:9])
    ref = _fp4_ref(x, act, packed, scale, g)
    exact = checkpoint.matmul(checkpoint.A4, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err                                                      # exact products, fp32 sums
    err = ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-2, err                                                      # and bf16 out


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120)])
def test_fp8_lane_is_the_quantized_product_and_rows_are_independent(n, k):
    rng = np.random.default_rng(n)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    s, act = 0.0371, 0.0071
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s, act=act)
    x = _rows(128, k, 4, 0.5)
    full = lin(x)
    for rows in ROWS:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).cuda()
    xq = (x.float() * inv).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    wq = torch.from_numpy(w).cuda().view(torch.float8_e4m3fn).double()
    ref = (xq @ wq.t()) * (np.float32(act) * np.float32(s))
    exact = checkpoint.matmul(checkpoint.A8, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err


@pytest.mark.parametrize("mode", [checkpoint.A4, checkpoint.A8])
def test_slices_meet_in_part_with_the_clusters_bits(mode):
    """SM 8.9 has no clusters: K slices add through ``part`` and the reduce, in the clusters' order: the same bits."""

    from tensorfold.cuda.kernels import qmm

    n, k, m = 5120, 17408, 37                                                   # 4 slices (a down projection)
    if mode == checkpoint.A4:
        packed, scale, g = _fp4_weight(n, k, 7)
        weight, scales = torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda()
        lin = Fp4Linear.from_checkpoint(weight, scales, g, act=0.0123)
    else:
        w = np.random.default_rng(7).integers(0, 256, size=(n, k), dtype=np.uint8)
        w[(w & 0x7F) >= 0x70] = 0x30
        lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), 0.0371, act=0.0071)
    sk = qmm.split_k(lin.n, lin.k)
    assert 1 < sk <= 8
    x = _rows(m, k, 5, 0.6)
    rows = checkpoint._rowsq(mode, x, lin.act)
    clusters = checkpoint.matmul(mode, x, lin)
    y = torch.empty_like(clusters)
    part = torch.empty((sk, m, lin.n), dtype=torch.float32, device="cuda")
    w, ws = (lin.words, lin.bs) if mode == checkpoint.A4 else (lin.w8, None)
    checkpoint._ext().lane(mode, rows[0], rows[1], w, ws, checkpoint.alpha(lin.act, lin.scale), y, part, lin.n, lin.k,
                           sk, lin.npad, qmm.bucket(m), False)
    assert torch.equal(y, clusters)


def test_head_tiles_keep_the_checkpoint_math():
    """The drafter's head slices (64-column tiles as views) keep the input scale and the FP4 mma's word order."""

    n, k = 640, 512
    packed, scale, g = _fp4_weight(n, k, 11)
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=0.02)
    x = _rows(9, k, 12)
    part = lin.tiles(2, 7)
    assert part.act == lin.act and torch.equal(part(x), lin(x)[:, 128:448])


CHUNKS = (1, 37, 128, 135)
TILES = (1, 2, 3, 11, 12, 13)               # the GEMM's tiles, then the warp-specialized GEMM's


def _chunked(fn, x):
    out, a = [], 0
    while a < x.shape[0]:
        for size in CHUNKS:
            out.append(fn(x[a:a + size].contiguous()))
            a += size
            if a >= x.shape[0]:
                break
    return torch.cat(out)[:x.shape[0]]


@pytest.mark.parametrize("n,k", [(128, 256), (1000, 5120), (12288, 5120)])
def test_fp4_prompt_rows_are_chunk_invariant_and_the_quantized_product(n, k):
    """The prompt GEMM: one K chain a row, so any chunking and every tile give a row the same bits."""

    from tensorfold.cuda.kernels import qmm

    packed, scale, g = _fp4_weight(n, k, 7 * n)
    act = 0.0123
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=act)
    x = _rows(300, k, 5, 0.7)
    full = checkpoint.prompt(checkpoint.A4, x, lin)
    for tile in (0, 11):
        assert torch.equal(_chunked(lambda c: checkpoint.prompt(checkpoint.A4, c, lin, tile=tile), x), full), tile
    for tile in TILES:
        assert torch.equal(checkpoint.prompt(checkpoint.A4, x, lin, tile=tile), full), tile
    ref = _fp4_ref(x, act, packed, scale, g)
    exact = checkpoint.prompt(checkpoint.A4, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err
    if qmm.split_k(n, k) == 1:                                                  # one K slice: the lane's chain too
        assert torch.equal(full[:128], lin(x[:128].contiguous()))


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120)])
def test_fp8_prompt_rows_are_chunk_invariant_and_the_quantized_product(n, k):
    rng = np.random.default_rng(n + 1)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    s, act = 0.0371, 0.0071
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s, act=act)
    x = _rows(300, k, 6, 0.5)
    full = checkpoint.prompt(checkpoint.A8, x, lin)
    for tile in (0, 11):
        assert torch.equal(_chunked(lambda c: checkpoint.prompt(checkpoint.A8, c, lin, tile=tile), x), full), tile
    for tile in TILES:
        assert torch.equal(checkpoint.prompt(checkpoint.A8, x, lin, tile=tile), full), tile
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).cuda()
    xq = (x.float() * inv).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    wq = torch.from_numpy(w).cuda().view(torch.float8_e4m3fn).double()
    ref = (xq @ wq.t()) * (np.float32(act) * np.float32(s))
    exact = checkpoint.prompt(checkpoint.A8, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err


@pytest.mark.parametrize("fp32", [True, False])
def test_fused_mlp_is_swiglu_then_down_and_chunk_invariant(monkeypatch, fp32):
    """gate|up -> SiLU(gate) * up -> down, quantized in the GEMM's epilogue: the fp64 product, chunk-invariant bits."""

    monkeypatch.setattr(checkpoint, "SWIGLU_FP32", fp32)
    hidden, inter, act, act_d = 256, 384, 0.0123, 0.0071
    lins, raw = [], []
    for n, k, a in ((inter, hidden, act), (inter, hidden, act), (hidden, inter, act_d)):
        packed, scale, g = _fp4_weight(n, k, n + k + len(lins))
        raw.append((packed, scale, g))
        lins.append(Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g,
                                              act=a))
    gate, up, down = lins
    x = _rows(300, hidden, 13, 0.7)
    y = checkpoint.mlp_prompt(x, gate, up, down)
    assert y is not None and y.shape == (300, hidden)
    assert torch.equal(_chunked(lambda c: checkpoint.mlp_prompt(c, gate, up, down), x), y)
    gq = checkpoint.prompt(checkpoint.A4, x, gate, f32=True)
    uq = checkpoint.prompt(checkpoint.A4, x, up, f32=True)
    if not fp32:
        gq, uq = gq.bfloat16().float(), uq.bfloat16().float()
    a = gq * torch.sigmoid(gq) * uq
    if not fp32:
        a = a.bfloat16().float()
    ref = _fp4_ref(a, act_d, *raw[2])                           # SwiGLU rows quantized as the epilogue does
    rel = (y.double() - ref).abs() / (ref.abs() + ref.abs().mean())
    assert (rel < 1e-2).double().mean().item() > 0.999, rel.max().item()   # exp's last ulp can flip a code
    assert ((y.double() - ref).norm() / ref.norm()).item() < 1e-2
