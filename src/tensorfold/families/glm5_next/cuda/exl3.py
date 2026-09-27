"""EXL3 reference decoding preserves whole tiles and Hadamard blocks across ranks; format from ExLlamaV3, https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

MCG = 0xCBAC1FED
MASK = 0x8FFF8FFF
FLIP = 0x3B603B60
HAD = 128


@lru_cache(maxsize=1)
def mcg_values() -> np.ndarray:
    """The codebook: the fp16 value of every 16-bit state, [65536]."""

    s = np.arange(65536, dtype=np.uint64)
    x = (s * MCG) & 0xFFFFFFFF
    x = (x & MASK) ^ FLIP
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)            # the float64 sum is exact, so this is one fp16 rounding


@lru_cache(maxsize=1)
def tile_positions() -> tuple[np.ndarray, np.ndarray]:
    """(row, column) in its 16x16 tile of each of a tile's 256 values, in stream order."""

    p = np.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    return rows, cols


def states(trellis: torch.Tensor | np.ndarray, bits: int = 4) -> np.ndarray:
    """The 16-bit state of every value: [K/16, N/16, 256] uint32, from trellis int16 [K/16, N/16, 16 * bits]."""

    t = np.ascontiguousarray(trellis.cpu().numpy() if isinstance(trellis, torch.Tensor) else trellis)
    if t.dtype != np.int16 or t.shape[-1] != 16 * bits:
        raise ValueError(f"trellis must be int16 [..., {16 * bits}], got {t.dtype} {t.shape}")
    words = t.view(np.uint16).astype(np.uint64)
    words = words[..., 0::2] | (words[..., 1::2] << 16)             # little-endian pairs: 8 * bits words a tile
    nw = 8 * bits
    p = np.arange(256)
    first = p * bits + bits - 16 + 256 * bits                        # the state's first bit (made non-negative)
    last = first + 16                                                # one past its last bit
    i0, i1 = (first // 32) % nw, ((last - 1) // 32) % nw
    shift = ((last - 1) // 32 + 1) * 32 - last
    a, b = words[..., i0], words[..., i1]                            # [..., 256] each
    return (((a << 32) | b) >> shift.astype(np.uint64)) & 0xFFFF


def unpack(trellis: torch.Tensor | np.ndarray, bits: int = 4) -> torch.Tensor:
    """W_q [K, N] fp16, the weight in the rotated domain (what ExLlamaV3's ``reconstruct`` writes)."""

    s = states(trellis, bits)
    kt, nt = s.shape[0], s.shape[1]
    vals = mcg_values()[s.astype(np.int64)]                          # [kt, nt, 256] fp16
    rows, cols = tile_positions()
    w = np.zeros((kt, 16, nt, 16), dtype=np.float16)
    w[:, rows, :, cols] = vals.transpose(2, 0, 1)                    # [256, kt, nt] into (row, col) of each tile
    return torch.from_numpy(w.reshape(kt * 16, nt * 16))


@lru_cache(maxsize=4)
def hadamard(n: int = HAD) -> np.ndarray:
    """The n x n Sylvester Hadamard matrix of +1 and -1 (n a power of two): H[i, j] = (-1)^popcount(i & j)."""

    i = np.arange(n)
    parity = np.array([bin(v).count("1") & 1 for v in range(n)])
    return np.where(parity[(i[:, None] & i[None, :])] == 1, -1.0, 1.0)


def rotate(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply H / sqrt(128) to every block of 128 along ``dim`` (float64)."""

    h = torch.from_numpy(hadamard()) / np.sqrt(HAD)
    x = x.double().movedim(dim, -1)
    shape = x.shape
    x = (x.reshape(*shape[:-1], shape[-1] // HAD, HAD) @ h).reshape(shape)
    return x.movedim(-1, dim)


def dequantize(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """The layer's weight W [K, N] in float64: diag(suh) @ H_K @ W_q @ H_N @ diag(svh)."""

    wq = unpack(trellis, bits).double()
    w = rotate(wq, 0) * suh.double()[:, None]
    return rotate(w, 1) * svh.double()[None, :]


def forward(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """y = x @ W in float64, computed the way the kernels do it: rotate the input, multiply by W_q, rotate the output."""

    xh = rotate(x.double() * suh.double(), -1)
    return rotate(xh @ unpack(trellis, bits).double(), -1) * svh.double()
