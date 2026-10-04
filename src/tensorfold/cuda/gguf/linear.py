"""GGUF packed linears: weights stay packed, CUDA tiles decode only the values they consume."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .codebook import IQ2_XXS_GRID

if TYPE_CHECKING:
    from .prefill import Workspace

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
    GROUPS: tl.constexpr = 1,
):
    r = tl.program_id(0) * BM
    group = tl.program_id(2)
    width = N // GROUPS
    n = group * width + tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a0 = tl.zeros((BN, BK), tl.float32)
    a1 = tl.zeros((BN, BK), tl.float32)
    if BM == 4:
        a2 = tl.zeros((BN, BK), tl.float32)
        a3 = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        w = _values(W, G, n[:, None], k[None, :], (n[:, None] < (group + 1) * width) & (k[None, :] < K), K, F)
        x0 = tl.load(X + r * GROUPS * K + group * K + k, (r < R) & (k < K), 0).to(tl.float32)
        x1 = tl.load(X + (r + 1) * GROUPS * K + group * K + k, (r + 1 < R) & (k < K), 0).to(tl.float32)
        a0 += w * x0[None, :]
        a1 += w * x1[None, :]
        if BM == 4:
            x2 = tl.load(X + (r + 2) * GROUPS * K + group * K + k, (r + 2 < R) & (k < K), 0).to(tl.float32)
            x3 = tl.load(X + (r + 3) * GROUPS * K + group * K + k, (r + 3 < R) & (k < K), 0).to(tl.float32)
            a2 += w * x2[None, :]
            a3 += w * x3[None, :]
    tl.store(Y + r * N + n, tl.sum(a0, 1), (r < R) & (n < (group + 1) * width))
    tl.store(Y + (r + 1) * N + n, tl.sum(a1, 1), (r + 1 < R) & (n < (group + 1) * width))
    if BM == 4:
        tl.store(Y + (r + 2) * N + n, tl.sum(a2, 1), (r + 2 < R) & (n < (group + 1) * width))
        tl.store(Y + (r + 3) * N + n, tl.sum(a3, 1), (r + 3 < R) & (n < (group + 1) * width))


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
    GROUPS: tl.constexpr = 1,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    group = tl.program_id(2)
    width = N // GROUPS
    n = group * width + tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ki = start * BK + k
        a = tl.load(X + m[:, None] * GROUPS * K + group * K + ki[None, :], (m[:, None] < M) & (ki[None, :] < K), 0)
        b = _values(
            W, GRID, n[None, :], ki[:, None], (n[None, :] < (group + 1) * width) & (ki[:, None] < K), K, FORMAT
        ).to(tl.bfloat16)
        acc = tl.dot(a.to(tl.bfloat16), b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < (group + 1) * width))


@dataclass
class Packed:
    data: torch.Tensor
    shape: tuple[int, ...]  # GGML order: K, N, optional expert
    format: str
    layout: str = "raw"  # raw | tile (BN blocks) | soa (IQ2/Q2 field split; bn carries section sizes)
    bn: int = 0
    workspace: Workspace | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.format not in FORMATS or len(self.shape) not in (2, 3):
            raise ValueError(f"unsupported packed matrix {self.format} {self.shape}")
        if self.layout not in ("raw", "tile", "soa"):
            raise ValueError(f"unsupported packed layout {self.layout}")
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
        if self.layout == "raw":
            if self.data.numel() != elements // block * size:
                raise ValueError("packed matrix payload length differs from its shape")
        elif self.layout == "soa":
            if self.format not in ("IQ2_XXS", "Q2_K") or len(self.shape) != 3 or self.bn <= 0:
                raise ValueError("soa layout requires IQ2/Q2 experts with section sizes in bn")
            nblk = self.shape[2] * self.shape[1] * (self.shape[0] // 256)
            if self.format == "IQ2_XXS":
                if self.bn % 64 or self.data.numel() != self.bn + nblk * 64:
                    raise ValueError("IQ2 soa payload length differs from its shape")
            else:
                dm_bytes = self.bn & 0xFFFFFFFF
                sc_bytes = self.bn >> 32
                if dm_bytes % 64 or sc_bytes % 64 or self.data.numel() != dm_bytes + sc_bytes + nblk * 64:
                    raise ValueError("Q2 soa payload length differs from its shape")
        else:
            if self.bn <= 0 or len(self.shape) != 3:
                raise ValueError("tiled packed matrix requires bn and a 3D expert shape")
            nb = (self.shape[1] + self.bn - 1) // self.bn
            kg = self.shape[0] // 256
            if self.data.numel() != self.shape[2] * nb * kg * self.bn * size:
                raise ValueError("tiled packed matrix payload length differs from its shape")

    def unpack(self) -> torch.Tensor:
        if self.layout != "raw":
            raise ValueError("unpack requires raw GGUF layout")
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

    def linear_grouped(self, x: torch.Tensor, *, dtype=torch.bfloat16) -> torch.Tensor:
        """Project [rows, groups, K] against consecutive output groups of a Q8 matrix."""

        k, n = self.shape[:2]
        if self.format != "Q8_0" or self.layout != "raw" or len(self.shape) != 2:
            raise ValueError("grouped linear requires a dense raw Q8 matrix")
        if (
            x.ndim != 3
            or not x.is_cuda
            or x.device != self.data.device
            or x.shape[0] < 1
            or x.shape[1] < 1
            or x.shape[-1] != k
            or n % x.shape[1]
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("grouped input requires CUDA floating [rows, groups, K] with matching matrix groups")
        x = x.contiguous()
        rows, groups, _ = x.shape
        if self.workspace is not None and self.workspace.accepts(rows, k, n):
            return self.workspace.grouped(self, x, dtype=dtype)
        out = torch.empty((rows, n), device=x.device, dtype=dtype)
        if rows > 16:
            _mm[(triton.cdiv(rows, 128), triton.cdiv(n // groups, 64), groups)](
                x,
                self.data,
                codebook(x.device),
                out,
                rows,
                n,
                k,
                self.format,
                128,
                64,
                32,
                groups,
                num_warps=8,
                num_stages=1,
                enable_fp_fusion=False,
            )
        else:
            bm = 2 if rows <= 2 else 4
            _mv_rows[(triton.cdiv(rows, bm), triton.cdiv(n // groups, 4), groups)](
                x,
                self.data,
                codebook(x.device),
                out,
                rows,
                n,
                k,
                self.format,
                bm,
                4,
                256,
                groups,
                num_warps=4,
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
            if self.layout == "soa" and self.format == "IQ2_XXS" and plan.tile == 64:
                # Staged CUDA D2R beats Triton SoA on the prefill-sized path; opt out with =0.
                import os

                if x.dtype == dtype == torch.bfloat16 and os.environ.get("TENSORFOLD_GGUF_CUDA_PREFILL", "1") != "0":
                    from .cuda_prefill import iq2_soa_group_prefill

                    iq2_soa_group_prefill(x, self, plan, out, slots=slots, input_slots=input_slots)
                    return out
                bn, bk = (128 if input_slots else 64, 32)
                _group_mm_iq2_soa[(plan.items.shape[0], triton.cdiv(n, bn))](
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
                    self.shape[2],
                    rows * slots,
                    plan.tile,
                    bn,
                    bk,
                    self.bn,
                    num_warps=8,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
                return out
            if self.layout == "tile" and self.format in ("IQ2_XXS", "Q2_K") and plan.tile == 64 and self.bn > 0:
                from .tiled import group_prefill_tiled

                group_prefill_tiled(x, self, plan, out, slots=slots, input_slots=input_slots)
                return out
            if self.layout == "soa" and self.format == "Q2_K" and plan.tile == 64:
                import os

                # Q8_1×Q2 dp4a path (opt-in): correct but not yet faster than Triton BF16 SoA.
                if input_slots and os.environ.get("TENSORFOLD_GGUF_Q8", "") == "1":
                    from .cuda_prefill import q2_soa_q8_group_prefill, quantize_q8_1

                    x_flat = x.reshape(rows * slots, k).contiguous()
                    x8 = quantize_q8_1(x_flat)
                    q2_soa_q8_group_prefill(x8, self, plan, out, slots=slots)
                    return out
                cuda_q2 = os.environ.get("TENSORFOLD_GGUF_CUDA_Q2", os.environ.get("TENSORFOLD_GGUF_CUDA_PREFILL", "1"))
                if x.dtype == dtype == torch.bfloat16 and cuda_q2 != "0":
                    from .cuda_prefill import q2_soa_group_prefill

                    q2_soa_group_prefill(x, self, plan, out, slots=slots, input_slots=input_slots)
                    return out
                # Wider N/K tiles reuse decoded Q2 scales across more accumulators.
                bn, bk = (256 if input_slots else 128, 64)
                _group_mm_q2_soa[(plan.items.shape[0], triton.cdiv(n, bn))](
                    x.contiguous(),
                    self.data,
                    plan.items,
                    plan.members,
                    plan.counts,
                    out,
                    n,
                    k,
                    slots,
                    input_slots,
                    self.shape[2],
                    rows * slots,
                    plan.tile,
                    bn,
                    bk,
                    self.bn & 0xFFFFFFFF,
                    self.bn >> 32,
                    num_warps=8,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
                return out
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
            if self.format == "Q8_0" and self.workspace is not None and self.workspace.accepts(rows, k, n):
                self.workspace.matmul(self, x, out)
                return out[:, 0]
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
                x,
                self.data,
                codebook(x.device),
                out,
                rows,
                n,
                k,
                self.format,
                bm,
                4,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            if picks is None:
                picks = torch.zeros((rows, 1), dtype=torch.int32, device=x.device)
            # Share expert weight reads across decode/verify rows that pick the same
            # expert (dense already uses _mv_rows); keep each row's FP32 reduction.
            if len(self.shape) == 3 and self.layout in ("raw", "soa") and rows > 1:
                from tensorfold.cuda.experts import Plan, route

                plan = Plan(rows, slots, self.shape[2], x.device, prefill=False)
                route(picks, plan, tile=16)
                flat = x.reshape(rows * slots, k) if x.ndim == 3 else x
                if self.layout == "soa" and self.format == "IQ2_XXS":
                    _mv_group_iq2_soa[(plan.items.shape[0], triton.cdiv(n, 4))](
                        flat.contiguous(),
                        self.data,
                        codebook(x.device),
                        plan.items,
                        plan.members,
                        plan.counts,
                        out.reshape(rows * slots, n),
                        n,
                        k,
                        self.shape[2],
                        rows * slots,
                        plan.tile,
                        4,
                        256,
                        self.bn,
                        enable_fp_fusion=False,
                    )
                elif self.layout == "soa" and self.format == "Q2_K":
                    _mv_group_q2_soa[(plan.items.shape[0], triton.cdiv(n, 4))](
                        flat.contiguous(),
                        self.data,
                        plan.items,
                        plan.members,
                        plan.counts,
                        out.reshape(rows * slots, n),
                        n,
                        k,
                        self.shape[2],
                        rows * slots,
                        plan.tile,
                        4,
                        256,
                        self.bn & 0xFFFFFFFF,
                        self.bn >> 32,
                        enable_fp_fusion=False,
                    )
                else:
                    _mv_group[(plan.items.shape[0], triton.cdiv(n, 4))](
                        flat.contiguous(),
                        self.data,
                        codebook(x.device),
                        plan.items,
                        plan.members,
                        plan.counts,
                        out.reshape(rows * slots, n),
                        n,
                        k,
                        self.format,
                        self.shape[2],
                        rows * slots,
                        plan.tile,
                        4,
                        256,
                        enable_fp_fusion=False,
                    )
            elif self.layout == "soa" and self.format == "IQ2_XXS":
                _mv_iq2_soa[(rows, triton.cdiv(n, 4) * slots)](
                    x,
                    self.data,
                    codebook(x.device),
                    picks,
                    out,
                    n,
                    k,
                    slots,
                    self.shape[2] if len(self.shape) == 3 else 1,
                    256,
                    4,
                    self.bn,
                    enable_fp_fusion=False,
                )
            elif self.layout == "soa" and self.format == "Q2_K":
                _mv_q2_soa[(rows, triton.cdiv(n, 4) * slots)](
                    x,
                    self.data,
                    picks,
                    out,
                    n,
                    k,
                    slots,
                    self.shape[2] if len(self.shape) == 3 else 1,
                    256,
                    4,
                    self.bn & 0xFFFFFFFF,
                    self.bn >> 32,
                    enable_fp_fusion=False,
                )
            else:
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
def _mv_q2_soa(
    X,
    W,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    dmh = W.to(tl.pointer_type(tl.float16))
    sc = W + DM_BYTES
    qs = W + DM_BYTES + SC_BYTES
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values_q2_soa(
            dmh,
            sc,
            qs,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
        )
        acc += w * x[None, :]
    tl.store(Y + (r * SLOTS + slot) * N + cols, tl.sum(acc, 1), cols < N)


@triton.jit
def _mv_group_q2_soa(
    X,
    W,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
):
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        dmh = W.to(tl.pointer_type(tl.float16))
        sc = W + DM_BYTES
        qs = W + DM_BYTES + SC_BYTES
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values_q2_soa(
                        dmh,
                        sc,
                        qs,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                    )
                    x0 = tl.load(X + p0 * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1 * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2 * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3 * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _mv_iq2_soa(
    X,
    W,
    GRID,
    PICKS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    DQ_BYTES: tl.constexpr,
):
    r = tl.program_id(0)
    pair = tl.program_id(1)
    slot = pair % SLOTS
    cols = (pair // SLOTS) * BN + tl.arange(0, BN)
    expert = tl.load(PICKS + r * SLOTS + slot).to(tl.int64)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    dq = W.to(tl.pointer_type(tl.float16))
    qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + r * SLOTS * K + slot * K + k, k < K, 0).to(tl.float32)
        w = _values_iq2_soa(
            dq,
            qs,
            GRID,
            expert * N + cols[:, None],
            k[None, :],
            (cols[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
            K,
        )
        acc += w * x[None, :]
    tl.store(Y + (r * SLOTS + slot) * N + cols, tl.sum(acc, 1), cols < N)


@triton.jit
def _mv_group_iq2_soa(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DQ_BYTES: tl.constexpr,
):
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        dq = W.to(tl.pointer_type(tl.float16))
        qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values_iq2_soa(
                        dq,
                        qs,
                        GRID,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                    )
                    x0 = tl.load(X + p0.to(tl.int64) * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1.to(tl.int64) * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2.to(tl.int64) * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3.to(tl.int64) * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _mv_group(
    X,
    W,
    GRID,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Decode-width grouped GEMV: `_mv_rows` math, expert-offset weights, Plan members."""
    item = tl.program_id(0)
    if item < tl.load(COUNTS):
        expert = tl.load(ITEMS + item * 3).to(tl.int64)
        first = tl.load(ITEMS + item * 3 + 1)
        count = tl.minimum(tl.maximum(tl.load(ITEMS + item * 3 + 2), 0), BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        # Waves of 4 match dense `_mv_rows`: one weight decode feeds four [BN,BK] lanes.
        for base in tl.static_range(0, BM, 4):
            if base < count:
                p0 = tl.load(MEMBERS + first + base, base < count, 0)
                p1 = tl.load(MEMBERS + first + base + 1, base + 1 < count, 0)
                p2 = tl.load(MEMBERS + first + base + 2, base + 2 < count, 0)
                p3 = tl.load(MEMBERS + first + base + 3, base + 3 < count, 0)
                ok0 = (base < count) & (p0 >= 0) & (p0 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok1 = (base + 1 < count) & (p1 >= 0) & (p1 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok2 = (base + 2 < count) & (p2 >= 0) & (p2 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                ok3 = (base + 3 < count) & (p3 >= 0) & (p3 < PAIRS) & (expert >= 0) & (expert < EXPERTS)
                a0 = tl.zeros((BN, BK), tl.float32)
                a1 = tl.zeros((BN, BK), tl.float32)
                a2 = tl.zeros((BN, BK), tl.float32)
                a3 = tl.zeros((BN, BK), tl.float32)
                for start in range(tl.cdiv(K, BK)):
                    k = start * BK + kk
                    w = _values(
                        W,
                        GRID,
                        expert * N + n[:, None],
                        k[None, :],
                        (n[:, None] < N) & (k[None, :] < K) & (expert >= 0) & (expert < EXPERTS),
                        K,
                        FORMAT,
                    )
                    x0 = tl.load(X + p0.to(tl.int64) * K + k, ok0 & (k < K), 0).to(tl.float32)
                    x1 = tl.load(X + p1.to(tl.int64) * K + k, ok1 & (k < K), 0).to(tl.float32)
                    x2 = tl.load(X + p2.to(tl.int64) * K + k, ok2 & (k < K), 0).to(tl.float32)
                    x3 = tl.load(X + p3.to(tl.int64) * K + k, ok3 & (k < K), 0).to(tl.float32)
                    a0 += w * x0[None, :]
                    a1 += w * x1[None, :]
                    a2 += w * x2[None, :]
                    a3 += w * x3[None, :]
                tl.store(Y + p0.to(tl.int64) * N + n, tl.sum(a0, 1), ok0 & (n < N))
                tl.store(Y + p1.to(tl.int64) * N + n, tl.sum(a1, 1), ok1 & (n < N))
                tl.store(Y + p2.to(tl.int64) * N + n, tl.sum(a2, 1), ok2 & (n < N))
                tl.store(Y + p3.to(tl.int64) * N + n, tl.sum(a3, 1), ok3 & (n < N))


@triton.jit
def _values_iq2_soa(DQ, QS, GRID, row, k, mask, K: tl.constexpr):
    """IQ2 from SoA: DQ half scales, QS 8×u64 code groups per 256-K block."""
    blk = (row * (K // 256) + k // 256).to(tl.int64)
    j = k % 256
    d = tl.load(DQ + blk, mask, 0).to(tl.float32)
    word = tl.load(QS + blk * 8 + j // 32, mask, 0)
    # Each u64 holds 4 grid bytes + 4 bytes of packed signs/scale-nibble.
    grid = ((word >> (8 * ((j % 32) // 8))) & 255).to(tl.int32)
    # signs live in bytes 4..7 as a little-endian uint32 (matches raw block).
    bits = (word >> 32).to(tl.uint32)
    signs = (bits >> (7 * ((j % 32) // 8))) & 127
    parity = signs ^ (signs >> 4)
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs |= (parity & 1) << 7
    mag = tl.load(GRID + grid * 8 + j % 8, mask, 0).to(tl.float32)
    sign = tl.where(((signs >> (j % 8)) & 1) != 0, -1.0, 1.0)
    d = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.25
    return d * mag * sign


@triton.jit
def _values_q2_soa(DMH, SC, QS, row, k, mask, K: tl.constexpr):
    """Q2_K from SoA: DMH float16[2] per block, SC 16 bytes, QS 64 bytes."""
    blk = (row * (K // 256) + k // 256).to(tl.int64)
    j = k % 256
    d = tl.load(DMH + blk * 2, mask, 0).to(tl.float32)
    m = tl.load(DMH + blk * 2 + 1, mask, 0).to(tl.float32)
    sc = tl.load(SC + blk * 16 + j // 16, mask, 0).to(tl.int32)
    bits = tl.load(QS + blk * 64 + (j // 128) * 32 + j % 32, mask, 0).to(tl.int32)
    q = (bits >> (2 * ((j % 128) // 32))) & 3
    return (d * (sc & 15)) * q.to(tl.float32) - m * (sc >> 4)


@triton.jit
def _group_mm_iq2_soa(
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
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DQ_BYTES: tl.constexpr,
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
        dq = W.to(tl.pointer_type(tl.float16))
        qs = (W + DQ_BYTES).to(tl.pointer_type(tl.uint64))
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values_iq2_soa(
                dq,
                qs,
                GRID,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


@triton.jit
def _group_mm_q2_soa(
    X,
    W,
    ITEMS,
    MEMBERS,
    COUNTS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SLOTS: tl.constexpr,
    INPUT_SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAIRS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DM_BYTES: tl.constexpr,
    SC_BYTES: tl.constexpr,
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
        dmh = W.to(tl.pointer_type(tl.float16))
        sc = W + DM_BYTES
        qs = W + DM_BYTES + SC_BYTES
        for start in range(tl.cdiv(K, BK)):
            ki = start * BK + k
            a = tl.load(X + inputs[:, None].to(tl.int64) * K + ki[None, :], (valid[:, None]) & (ki[None, :] < K), 0)
            b = _values_q2_soa(
                dmh,
                sc,
                qs,
                expert * N + n[None, :],
                ki[:, None],
                (n[None, :] < N) & (ki[:, None] < K) & (expert >= 0) & (expert < EXPERTS),
                K,
            ).to(tl.bfloat16)
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        tl.store(Y + pairs[:, None].to(tl.int64) * N + n[None, :], acc, valid[:, None] & (n[None, :] < N))


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
