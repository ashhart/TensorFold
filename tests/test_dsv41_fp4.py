"""DeepSeek-V4.1's native KV numerics: NVFP4 compressed entries, MXFP4 indexer keys and queries, FP8 (UE8M0) window.

``dsv41_fp4_ref`` ports DeepSeek's quantizers (DeepSeek-V4.1-Flash ``inference/kernel.py``: ``fp4_act_quant``,
``act_quant``; Copyright (c) 2026 DeepSeek, MIT License) to PyTorch. Here it is checked against an independent NumPy
oracle (nearest value by distance, ties to the even code, scales from frexp), and on a GPU the engine's Triton store /
decode kernels are checked against it byte for byte.
"""

import numpy as np
import pytest
import torch

import dsv41_fp4_ref as Q

MAGS = np.array(Q.MAGS, dtype=np.float32)
_E4M3 = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()   # 0 .. 448, ascending


def _nearest_even(table: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Index of the nearest table value to each a (table ascending, a within it), ties to the even index."""

    hi = np.clip(np.searchsorted(table, a), 1, len(table) - 1)
    lo = hi - 1
    dl, dh = a - table[lo], table[hi] - a
    return np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))


def _e2m1(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * MAGS[_nearest_even(MAGS, np.abs(y))]


def _e4m3(v: np.ndarray) -> np.ndarray:
    return _E4M3[_nearest_even(_E4M3, np.minimum(np.abs(v), 448.0))] * np.sign(v)


def _log2_ceil(t: np.ndarray) -> np.ndarray:
    m, e = np.frexp(t.astype(np.float64))
    return np.where(m == 0.5, e - 1, e)


def _groups(x: torch.Tensor, g: int) -> np.ndarray:
    a = x.float().numpy()
    return a.reshape(-1, a.shape[-1] // g, g)


def oracle_nvfp4(x: torch.Tensor) -> np.ndarray:
    g = _groups(x, 16)
    amax = np.maximum(np.abs(g).max(-1), np.float32(6 * 2 ** -9))
    s = _e4m3(amax / np.float32(6)).astype(np.float32)[..., None]
    return (_e2m1(np.clip(g / s, -6, 6)) * s).reshape(x.shape)


def oracle_mx(x: torch.Tensor, fmax: float, floor: float, fp8: bool) -> np.ndarray:
    g = _groups(x, 32)
    amax = np.maximum(np.abs(g).max(-1), np.float32(floor))
    s = np.ldexp(np.float32(1), _log2_ceil(amax * np.float32(1 / fmax))).astype(np.float32)[..., None]
    y = np.clip(g / s, -fmax, fmax)
    return ((_e4m3(y) if fp8 else _e2m1(y)) * s).reshape(x.shape)


def _bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)


def test_e2m1_ties_go_to_the_even_code():
    y = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.2499, 0.7501, 5.0001, -0.25, -0.0, -0.1, -5.5, 6.0])
    want = [0, 2, 2, 4, 4, 6, 6, 0, 2, 7, 0, 0, 0, 15, 7]
    assert Q.e2m1_codes(y).tolist() == want
    assert torch.equal(Q.e2m1_values(Q.e2m1_codes(y)), torch.from_numpy(_e2m1(y.numpy())).float())


def test_nibble_order_and_scale_bytes():
    x = torch.zeros((1, 16), dtype=torch.bfloat16)
    x[0, 0], x[0, 1], x[0, 2] = 6.0, -0.5, 3.0                # amax 6: scale e4m3(1.0) = 0x38
    packed, sc, deq = Q.nvfp4(x)
    assert packed[0, :2].tolist() == [7 | (9 << 4), 5] and sc.tolist() == [[0x38]]
    assert torch.equal(deq, x)
    k = torch.zeros((1, 32), dtype=torch.bfloat16)
    k[0, 3] = 3.0                                            # 3 / 6 = 0.5: scale 2^-1, byte 126, 3 -> 6
    packed, sc, deq = Q.mxfp4(k)
    assert sc.tolist() == [[126]] and packed[0, 1].item() == 7 << 4 and torch.equal(deq, k)


def test_floors_zero_groups_and_signed_zero():
    z = torch.zeros((1, 512), dtype=torch.bfloat16)
    z[0, 5] = -0.0
    packed, sc, deq = Q.nvfp4(z)
    assert (sc == 1).all() and (packed == 0).all() and torch.equal(deq, z)       # 2^-9: the least e4m3 subnormal
    packed, sc, deq = Q.mxfp4(z[:, :128])
    assert (sc == 1).all() and (packed == 0).all()                                # 6 * 2^-126 / 6 -> 2^-126
    assert torch.equal(Q.mxfp8(z), z)


def test_nvfp4_clamps_when_the_scale_rounds_down():
    x = torch.zeros((1, 16), dtype=torch.bfloat16)
    x[0, 0] = 6.375                                          # 6.375 / 6 = 1.0625 -> e4m3 1.0: 6.375 / 1 clamps to 6
    packed, sc, deq = Q.nvfp4(x)
    assert sc.item() == 0x38 and deq[0, 0].item() == 6.0
    assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x)).float())


def test_mx_scales_next_to_powers_of_two():
    rows = []
    for k in range(-20, 12):
        for f in (1 - 2 ** -8, 1.0, 1 + 2 ** -7):            # bf16 neighbours of 6 * 2^k
            rows.append(6 * 2.0 ** k * f)
    x = torch.zeros((len(rows), 32), dtype=torch.bfloat16)
    x[:, 7] = torch.tensor(rows)
    _, sc, deq = Q.mxfp4(x)
    amax = x.float().abs().amax(-1).numpy()
    want = _log2_ceil(np.maximum(amax, np.float32(6 * 2 ** -126)) * np.float32(1 / 6)) + 127
    assert sc[:, 0].numpy().tolist() == want.tolist()
    assert torch.equal(deq.float(), torch.from_numpy(oracle_mx(x, 6, 6 * 2 ** -126, False)).float())
    assert torch.equal(Q.mxfp8(x).float(), torch.from_numpy(oracle_mx(x, 448, 1e-4, True)).float())


@pytest.mark.parametrize("scale", [1e-3, 0.3, 2.0, 40.0])
def test_random_rows_match_the_oracle(scale):
    g = torch.Generator().manual_seed(int(scale * 1000))
    x = (torch.randn((2048, 512), generator=g) * scale * torch.rand((2048, 1), generator=g)).to(torch.bfloat16)
    x[:7, :16] = 0.0
    packed, sc, deq = Q.nvfp4(x)
    assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x)).float())
    assert torch.equal(Q.dequant(packed, sc, 16, True), deq)
    k = x[:, :128].contiguous()
    packed, sc, deq = Q.mxfp4(k)
    assert torch.equal(deq.float(), torch.from_numpy(oracle_mx(k, 6, 6 * 2 ** -126, False)).float())
    assert torch.equal(Q.dequant(packed, sc, 32, False), deq)
    assert torch.equal(Q.mxfp8(x).float(), torch.from_numpy(oracle_mx(x, 448, 1e-4, True)).float())


def test_ties_reached_through_the_scale():
    """x / s landing exactly on an E2M1 midpoint (a scale of 1, 0.5 or 2^-9) rounds to the even code."""

    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    for s in (1.0, 0.5, 2.0 ** -9):
        x = torch.zeros((1, 16))
        x[0, :8], x[0, 8:] = mids * s, -mids * s
        packed, _, deq = Q.nvfp4(x.to(torch.bfloat16))
        codes = Q.unpack(packed)[0].tolist()
        assert codes == [0, 2, 2, 4, 4, 6, 6, 7, 0, 10, 10, 12, 12, 14, 14, 15]
        assert torch.equal(deq.float(), torch.from_numpy(oracle_nvfp4(x.to(torch.bfloat16))).float())
