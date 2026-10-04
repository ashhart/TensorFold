"""Bounded Q8 prompt scratch: decode once, then use BF16 operands and FP32 sums."""

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
def _gemm(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUPS: tl.constexpr = 1):
    group = tl.program_id(2)
    width = N // GROUPS
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = group * width + tl.program_id(1) * 128 + tl.arange(0, 128)
    k = tl.arange(0, 64)
    acc = tl.zeros((128, 128), tl.float32)
    for start in range(tl.cdiv(K, 64)):
        ki = start * 64 + k
        a = tl.load(X + m[:, None] * GROUPS * K + group * K + ki[None, :], (m[:, None] < M) & (ki[None, :] < K), 0).to(
            tl.bfloat16
        )
        offset = n[None, :] * K + ki[:, None]
        mask = (ki[:, None] < K) & (n[None, :] < (group + 1) * width)
        b = tl.load(W + offset, mask, 0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < (group + 1) * width))


class Workspace:
    """Shared by a model's linears, used sequentially on its current CUDA stream."""

    def __init__(self):
        self.w = None

    @staticmethod
    def accepts(rows: int, k: int, n: int) -> bool:
        # Amortize decoding without expanding the resident weights.
        return rows >= 1024 and k >= 1024 and n >= 1024 and k * n <= MAX_ELEMENTS

    def _prepare(self, weight, device):
        k, n = weight.shape
        if k * n > MAX_ELEMENTS:
            raise ValueError("Q8 prefill scratch exceeds 64 MiB")
        if self.w is None or self.w.numel() < k * n:
            self.w = torch.empty(k * n, device=device, dtype=torch.bfloat16)
        _decode[(triton.cdiv(k * n, 1024),)](weight.data, codebook(device), self.w, n, k, 1024, enable_fp_fusion=False)
        return k, n

    def matmul(self, weight, x: torch.Tensor, out: torch.Tensor) -> None:
        k, n = self._prepare(weight, x.device)
        _gemm[(triton.cdiv(x.shape[0], 128), triton.cdiv(n, 128))](
            x, self.w, out, x.shape[0], n, k, num_warps=8, num_stages=4, enable_fp_fusion=False
        )

    def grouped(self, weight, x: torch.Tensor, *, dtype=torch.bfloat16) -> torch.Tensor:
        """Validated grouped inputs share the same bounded scratch as dense linears."""

        k, n = self._prepare(weight, x.device)
        rows, groups, _ = x.shape
        out = torch.empty((rows, n), device=x.device, dtype=dtype)
        _gemm[(triton.cdiv(rows, 128), triton.cdiv(n // groups, 128), groups)](
            x, self.w, out, rows, n, k, groups, num_warps=8, num_stages=4, enable_fp_fusion=False
        )
        return out
