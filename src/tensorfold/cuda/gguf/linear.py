"""GGUF packed linears: weights stay packed, CUDA tiles decode only the values they consume."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
import triton
import triton.language as tl

from .codebook import IQ2_XXS_GRID

FORMATS = {"Q8_0": (32, 34), "Q2_K": (256, 84), "IQ2_XXS": (256, 66)}


@cache
def codebook(device: torch.device) -> torch.Tensor:
    values = [[(g >> (8 * j)) & 255 for j in range(8)] for g in IQ2_XXS_GRID]
    return torch.tensor(values, dtype=torch.uint8, device=device).flatten()


@triton.jit
def _half(W, offset, mask):
    # Every supported block and half offset is two-byte aligned.
    return tl.load(W.to(tl.pointer_type(tl.float16)) + offset // 2, mask, 0).to(tl.float32)


@triton.jit
def _values(W, GRID, row, k, mask, K: tl.constexpr, FORMAT: tl.constexpr):
    if FORMAT == "Q8_0":
        base = (row * (K // 32) + k // 32).to(tl.int64) * 34
        d = _half(W, base, mask)
        q = tl.load(W + base + 2 + k % 32, mask, 0).to(tl.int8, bitcast=True).to(tl.float32)
        return d * q
    elif FORMAT == "Q2_K":
        base = (row * (K // 256) + k // 256).to(tl.int64) * 84
        j = k % 256
        sc = tl.load(W + base + j // 16, mask, 0).to(tl.int32)
        bits = tl.load(W + base + 16 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
        q = (bits >> (2 * ((j % 128) // 32))) & 3
        d = _half(W, base + 80, mask)
        m = _half(W, base + 82, mask)
        return (d * (sc & 15)) * q.to(tl.float32) - m * (sc >> 4)
    else:
        base = (row * (K // 256) + k // 256).to(tl.int64) * 66
        j = k % 256
        sub = base + 2 + (j // 32) * 8
        grid = tl.load(W + sub + (j % 32) // 8, mask, 0).to(tl.int32)
        words = W.to(tl.pointer_type(tl.uint16))
        bits = tl.load(words + (sub + 4) // 2, mask, 0).to(tl.uint32)
        bits |= tl.load(words + (sub + 6) // 2, mask, 0).to(tl.uint32) << 16
        signs = (bits >> (7 * ((j % 32) // 8))) & 127
        parity = signs ^ (signs >> 4)
        parity ^= parity >> 2
        parity ^= parity >> 1
        signs |= (parity & 1) << 7
        mag = tl.load(GRID + grid * 8 + j % 8, mask, 0).to(tl.float32)
        sign = tl.where(((signs >> (j % 8)) & 1) != 0, -1.0, 1.0)
        d = _half(W, base, mask) * (0.5 + (bits >> 28).to(tl.float32)) * 0.25
        return d * mag * sign


@triton.jit
def _unpack(W, GRID, OUT, TOTAL: tl.constexpr, K: tl.constexpr, FORMAT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = _values(W, GRID, i // K, i % K, i < TOTAL, K, FORMAT)
    tl.store(OUT + i, v, i < TOTAL)


@triton.jit
def _mv(
    X,
    W,
    GRID,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values(
            W,
            GRID,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
            FORMAT,
        )
        acc += w * x[None, :]
    y = tl.sum(acc, 1)
    tl.store(Y + (r * SLOTS + slot) * N + cols, y, cols < N)


@triton.jit
def _mv_rows(
    X,
    W,
    G,
    Y,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    F: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a0 = tl.zeros((BN, BK), tl.float32)
    a1 = tl.zeros((BN, BK), tl.float32)
    if BM == 4:
        a2 = tl.zeros((BN, BK), tl.float32)
        a3 = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        w = _values(W, G, n[:, None], k[None, :], (n[:, None] < N) & (k[None, :] < K), K, F)
        x0 = tl.load(X + r * K + k, (r < R) & (k < K), 0).to(tl.float32)
        x1 = tl.load(X + (r + 1) * K + k, (r + 1 < R) & (k < K), 0).to(tl.float32)
        a0 += w * x0[None, :]
        a1 += w * x1[None, :]
        if BM == 4:
            x2 = tl.load(X + (r + 2) * K + k, (r + 2 < R) & (k < K), 0).to(tl.float32)
            x3 = tl.load(X + (r + 3) * K + k, (r + 3 < R) & (k < K), 0).to(tl.float32)
            a2 += w * x2[None, :]
            a3 += w * x3[None, :]
    tl.store(Y + r * N + n, tl.sum(a0, 1), (r < R) & (n < N))
    tl.store(Y + (r + 1) * N + n, tl.sum(a1, 1), (r + 1 < R) & (n < N))
    if BM == 4:
        tl.store(Y + (r + 2) * N + n, tl.sum(a2, 1), (r + 2 < R) & (n < N))
        tl.store(Y + (r + 3) * N + n, tl.sum(a3, 1), (r + 3 < R) & (n < N))


@triton.jit
def _mm(
    X,
    W,
    GRID,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ki = start * BK + k
        a = tl.load(X + m[:, None] * K + ki[None, :], (m[:, None] < M) & (ki[None, :] < K), 0)
        b = _values(W, GRID, n[None, :], ki[:, None], (n[None, :] < N) & (ki[:, None] < K), K, FORMAT).to(tl.bfloat16)
        acc = tl.dot(a.to(tl.bfloat16), b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@dataclass
class Packed:
    data: torch.Tensor
    shape: tuple[int, ...]  # GGML order: K, N, optional expert
    format: str

    def __post_init__(self):
        if self.format not in FORMATS or len(self.shape) not in (2, 3):
            raise ValueError(f"unsupported packed matrix {self.format} {self.shape}")
        block, size = FORMATS[self.format]
        elements = 1
        for d in self.shape:
            if d <= 0:
                raise ValueError("packed matrix dimensions must be positive")
            elements *= d
        if self.shape[0] % block or self.data.dtype != torch.uint8 or not self.data.is_contiguous():
            raise ValueError("packed matrix requires contiguous bytes and complete quantization blocks")
        if self.data.data_ptr() % 2:
            raise ValueError("packed matrix scales require two-byte aligned storage")
        if self.data.numel() != elements // block * size:
            raise ValueError("packed matrix payload length differs from its shape")

    def unpack(self) -> torch.Tensor:
        out = torch.empty(tuple(reversed(self.shape)), dtype=torch.float32, device=self.data.device)
        _unpack[(triton.cdiv(out.numel(), 256),)](
            self.data,
            codebook(self.data.device),
            out,
            out.numel(),
            self.shape[0],
            self.format,
            256,
            enable_fp_fusion=False,
        )
        return out

    def linear(
        self, x: torch.Tensor, picks: torch.Tensor | None = None, *, plan=None, dtype=torch.bfloat16, validated=False
    ) -> torch.Tensor:
        """Project dense rows or routed slots, leaving weights packed.

        ``validated`` is for a trusted router with checked expert ids; its supplied
        plan must have been routed with these same picks. Public calls check ids.
        """
        k, n = self.shape[:2]
        if x.ndim not in (2, 3) or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("packed linear requires CUDA floating input [rows,K] or [rows,slots,K]")
        if x.device != self.data.device or x.shape[-1] != k or x.shape[0] < 1:
            raise ValueError("linear input device, width, or row count differs from the packed matrix")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("packed linear output must be floating point")
        rows = x.shape[0]
        if len(self.shape) == 3 and picks is None:
            raise ValueError("expert matrix requires explicit expert picks")
        if len(self.shape) == 2 and picks is not None:
            raise ValueError("dense matrix does not take expert picks")
        if picks is not None and (
            picks.ndim != 2
            or picks.shape[0] != rows
            or picks.device != x.device
            or picks.dtype not in (torch.int32, torch.int64)
            or not picks.is_contiguous()
            or picks.shape[1] < 1
        ):
            raise ValueError("expert picks must be contiguous integer [rows,slots] on the input device")
        slots = 1 if picks is None else picks.shape[1]
        input_slots = x.ndim == 3
        if input_slots and x.shape[1] != slots:
            raise ValueError("input slots differ from expert picks")
        if picks is not None and not validated and (int(picks.min()) < 0 or int(picks.max()) >= self.shape[2]):
            raise ValueError("expert pick outside the matrix")
        if picks is not None:
            picks = picks.to(torch.int32)
        if picks is not None and rows > 16:
            from tensorfold.cuda.experts import Plan, route

            if plan is None:
                plan = Plan(rows, slots, self.shape[2], x.device, prefill=True)
                route(picks, plan, tile=64)
            elif not validated:
                if plan.experts != self.shape[2] or plan.slots != slots or plan.rows < rows or not plan.prefill:
                    raise ValueError("expert plan does not match the input")
                route(picks, plan, tile=64)
            out = torch.empty((rows, slots, n), dtype=dtype, device=x.device)
            # Small K tiles keep IQ2 grid/sign and Q2 scale decoding live ranges
            # short; wider output tiles reuse gathered rows.
            compact = self.format in ("IQ2_XXS", "Q2_K")
            bn, bk = (128 if input_slots else 64, 32) if compact else (32, 128)
            _group_mm[(plan.items.shape[0], triton.cdiv(n, bn))](
                x.contiguous(),
                self.data,
                codebook(x.device),
                plan.items,
                plan.members,
                plan.counts,
                out,
                n,
                k,
                slots,
                input_slots,
                self.format,
                self.shape[2],
                rows * slots,
                plan.tile,
                bn,
                bk,
                num_warps=8 if plan.tile == 64 else 4,
                num_stages=1 if compact else 3,
                enable_fp_fusion=False,
            )
            return out
        if picks is not None and x.ndim == 2:
            x = x[:, None, :].expand(rows, slots, k).contiguous()
        else:
            x = x.contiguous()
        out = torch.empty((rows, slots, n), dtype=dtype, device=x.device)
        if rows > 16 and picks is None:
            # Q8 scales cover 32 values; avoid carrying four blocks' decoding
            # through one dot. More rows then reuse each decoded weight tile.
            q8 = self.format == "Q8_0"
            bm, bn, bk = (128, 64, 32) if q8 else (64, 32, 128)
            _mm[(triton.cdiv(rows, bm), triton.cdiv(n, bn))](
                x,
                self.data,
                codebook(x.device),
                out,
                rows,
                n,
                k,
                self.format,
                bm,
                bn,
                bk,
                num_warps=8 if q8 else 4,
                num_stages=1 if q8 else 3,
                enable_fp_fusion=False,
            )
        elif picks is None and rows > 1:
            # Share packed weight reads; each row retains the serial accumulator
            # shape, block order and reduction (tensor-core dots would change it).
            bm = 2 if rows == 2 else 4
            _mv_rows[(triton.cdiv(rows, bm), triton.cdiv(n, 4))](
                x, self.data, codebook(x.device), out, rows, n, k, self.format,
                bm, 4, 256, num_warps=4, enable_fp_fusion=False,
            )
        else:
            if picks is None:
                picks = torch.zeros((rows, 1), dtype=torch.int32, device=x.device)
            _mv[(rows, triton.cdiv(n, 4) * slots)](
                x,
                self.data,
                codebook(x.device),
                picks,
                out,
                n,
                k,
                slots,
                self.format,
                self.shape[2] if len(self.shape) == 3 else 1,
                256,
                4,
                enable_fp_fusion=False,
            )
        return out[:, 0] if slots == 1 else out


@triton.jit
def _group_mm(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    INPUT_SLOTS: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        m = tl.arange(0, BM)
        valid = (m < count) & (first + m >= 0) & (first + m < PAIRS)
        pairs = tl.load(MEMBERS + first + m, valid, 0)
        valid = valid & (pairs >= 0) & (pairs < PAIRS) & (expert >= 0) & (expert < EXPERTS)
        inputs = pairs if INPUT_SLOTS else pairs // SLOTS
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values(
                W,
                GRID,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
                FORMAT,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


@triton.jit
def _float_mv(X, W, Y, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * K + k, k < K, 0).to(tl.float32)
        w = tl.load(W + n[:, None].to(tl.int64) * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc += x[None, :] * w
    tl.store(Y + r * N + n, tl.sum(acc, 1), n < N)


def linear(x: torch.Tensor, weight: Packed | torch.Tensor, *, dtype=torch.bfloat16) -> torch.Tensor:
    """Dense projection; narrow rows use the same FP32 reductions at every decode width."""
    if isinstance(weight, Packed):
        return weight.linear(x, dtype=dtype)
    if weight.ndim != 2 or x.ndim != 2 or x.shape[1] != weight.shape[1] or x.device != weight.device:
        raise ValueError("dense linear expects input [rows, K] and weight [N, K] on one device")
    n, k = weight.shape
    if x.shape[0] > 16:
        # Compressors require FP32 projections. Disable TF32 at engine startup.
        return (x.float() @ weight.float().T).to(dtype)
    out = torch.empty((x.shape[0], n), dtype=dtype, device=x.device)
    _float_mv[(x.shape[0], triton.cdiv(n, 4))](
        x.contiguous(), weight.contiguous(), out, n, k, 256, 4, enable_fp_fusion=False
    )
    return out
