"""Bounded Q8 prompt scratch: decode once, then reuse fixed-order BF16 dots."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .linear import _values, codebook

MAX_ELEMENTS = 32 << 20  # One shared buffer, at most 64 MiB per model.


@triton.jit
def _decode(W, GRID, OUT, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    w = _values(W, GRID, i // K, i % K, i < N * K, K, "Q8_0")
    tl.store(OUT + i, w.to(tl.bfloat16), i < N * K)


@triton.jit
def _gemm(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 32)
    acc = tl.zeros((128, 64), tl.float32)
    for start in range(tl.cdiv(K, 32)):
        ki = start * 32 + k
        a = tl.load(X + m[:, None] * K + ki[None, :], (m[:, None] < M) & (ki[None, :] < K), 0).to(tl.bfloat16)
        offset = n[None, :] * K + ki[:, None]
        mask = (ki[:, None] < K) & (n[None, :] < N)
        # Byte operands retain the packed kernel's kWidth=4 fragment ordering.
        # A direct BF16 load uses kWidth=2 and changes FP32 accumulation bits.
        words = W.to(tl.pointer_type(tl.uint8))
        lo = tl.load(words + offset * 2, mask, 0).to(tl.uint16)
        hi = tl.load(words + offset * 2 + 1, mask, 0).to(tl.uint16)
        b = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


class Workspace:
    """Shared by a model's linears, used sequentially on its current CUDA stream."""

    def __init__(self):
        self.w = None

    @staticmethod
    def accepts(rows: int, k: int, n: int) -> bool:
        # Short chunks cannot amortize decoding. Narrow-K projections prefer
        # the packed kernel; cap scratch instead of expanding resident weights.
        return rows >= 1024 and k >= 4096 and n >= 1024 and k * n <= MAX_ELEMENTS

    def matmul(self, weight, x: torch.Tensor, out: torch.Tensor) -> None:
        k, n = weight.shape
        if k * n > MAX_ELEMENTS:
            raise ValueError("Q8 prefill scratch exceeds 64 MiB")
        if self.w is None or self.w.numel() < k * n:
            self.w = torch.empty(k * n, device=x.device, dtype=torch.bfloat16)
        _decode[(triton.cdiv(k * n, 1024),)](
            weight.data, codebook(x.device), self.w, n, k, 1024, enable_fp_fusion=False
        )
        _gemm[(triton.cdiv(x.shape[0], 128), triton.cdiv(n, 64))](
            x, self.w, out, x.shape[0], n, k, num_warps=8, num_stages=3, enable_fp_fusion=False
        )
