"""Lane matmul blocks keep today's bits: every 27B shape, NVFP4 and FP8, M 1..256, through clusters and part."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("the block-scaled FP4 mma needs an SM 12.x GPU", allow_module_level=True)

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear

A4, A8 = checkpoint.A4, checkpoint.A8
# the 27B's projections (name, n, k, mode), and a width that ends mid 128-column block past its last 64-column tile
SHAPES = [("qkv", 10240, 5120, A8), ("z", 6144, 5120, A8), ("out", 5120, 6144, A8), ("q", 12288, 5120, A8),
          ("k", 1024, 5120, A8), ("gate", 17408, 5120, A4), ("down", 5120, 17408, A4), ("head", 248320, 5120, A4),
          ("tail4", 900, 5120, A4), ("tail8", 900, 6144, A8)]
TILES = (16, 32, 64, 64128, 128064, 128128)
ROWS = 256


def _linear(n: int, k: int, mode: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == A4:
        packed = torch.randint(0, 256, (n, k // 2), generator=g, device="cuda", dtype=torch.uint8)
        scale = torch.randint(0x20, 0x50, (n, k // 16), generator=g, device="cuda", dtype=torch.uint8)
        return Fp4Linear.from_checkpoint(packed, scale, 0.0123, act=0.0123)
    w = torch.randint(0, 256, (n, k), generator=g, device="cuda", dtype=torch.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30                                               # no NaN or saturating codes
    return Fp8Linear.from_checkpoint(w.view(torch.float8_e4m3fn), 0.0371, act=0.0071)


def _rows(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=g) * 0.6
    x[:, ::97] *= 12.0
    return x.to(torch.bfloat16).cuda()


def _lane(mode: int, lin, rows, m: int, tile: int, part: bool) -> torch.Tensor:
    sk = qmm.split_k(lin.n, lin.k)
    y = torch.empty((m, lin.n), dtype=torch.bfloat16, device="cuda")
    p = torch.empty((sk, m, lin.n), dtype=torch.float32, device="cuda") if part and sk > 1 else None
    w, ws = (lin.words, lin.bs) if mode == A4 else (lin.w8, None)
    codes, scales = rows
    checkpoint._ext().lane(mode, codes[:m], scales, w, ws, checkpoint.alpha(lin.act, lin.scale), y, p, lin.n, lin.k,
                           sk, lin.npad, tile, False)
    return y


@pytest.mark.parametrize("name,n,k,mode", SHAPES, ids=[s[0] for s in SHAPES])
def test_every_block_and_row_count_keeps_todays_bits(name, n, k, mode):
    lin = _linear(n, k, mode, n + k)
    x = _rows(ROWS, k, n)
    rows = checkpoint._rowsq(mode, x, lin.act)                   # rows quantize alone: a prefix is its own rows
    ref = _lane(mode, lin, rows, ROWS, 16, True)
    sk, sms = qmm.split_k(n, k), checkpoint.sm_count()
    bad = []
    for m in range(1, ROWS + 1):
        tile = checkpoint.lane_tile(m, n, sk, sms)
        for part in (False, True):
            if not torch.equal(_lane(mode, lin, rows, m, tile, part), ref[:m]):
                bad.append((m, tile, part))
    for tile in TILES:
        for m in (1, 15, 33, 64, 65, 96, 127, 128, 129, 200, 256):
            for part in (False, True):
                if not torch.equal(_lane(mode, lin, rows, m, tile, part), ref[:m]):
                    bad.append((m, tile, part))
    assert not bad, bad[:10]
    for m in (1, 37, 64, 100, 128):                              # and through the module's own routing
        assert torch.equal(checkpoint.matmul(mode, x[:m].contiguous(), lin), ref[:m]), m


def test_a_128_row_block_reads_no_scale_past_mpad():
    """40 rows quantize with scales for one 64-row tile (mpad 64): a 128-row block zero-fills the rest."""

    lin = _linear(1024, 5120, A4, 3)
    rows = checkpoint._rowsq(A4, _rows(40, 5120, 4), lin.act)
    assert tuple(rows[1].shape) == (5120 // 64, 64, 4)
    assert torch.equal(_lane(A4, lin, rows, 40, 128128, False), _lane(A4, lin, rows, 40, 16, True))
