"""Validated packed projections select native CUDA or Triton kernels without expanding weights."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import triton

from .expert_kernels import (
    _group_mm,
    _group_mm_iq2_soa,
    _group_mm_q2_soa,
    _mv_group,
    _mv_group_iq2_soa,
    _mv_group_q2_soa,
    _mv_iq2_soa,
    _mv_q2_soa,
)
from .kernels import _float_mm, _float_mv, _mm, _mv, _mv_rows, _unpack, codebook

if TYPE_CHECKING:
    from .prefill import Workspace

FORMATS = {"Q8_0": (32, 34), "Q2_K": (256, 84), "IQ2_XXS": (256, 66)}


@dataclass
class Packed:
    data: torch.Tensor
    shape: tuple[int, ...]  # GGML order: K, N, optional expert
    format: str
    layout: str = "raw"  # raw | soa (IQ2/Q2 field split; bn carries section sizes)
    bn: int = 0
    workspace: Workspace | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.format not in FORMATS or len(self.shape) not in (2, 3):
            raise ValueError(f"unsupported packed matrix {self.format} {self.shape}")
        if self.layout not in ("raw", "soa"):
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
            if self.layout == "soa" and self.format == "Q2_K" and plan.tile == 64:
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


def linear(x: torch.Tensor, weight: Packed | torch.Tensor, *, dtype=torch.bfloat16) -> torch.Tensor:
    """Dense projection; narrow rows use the same FP32 reductions at every decode width."""
    if isinstance(weight, Packed):
        return weight.linear(x, dtype=dtype)
    if weight.ndim != 2 or x.ndim != 2 or x.shape[1] != weight.shape[1] or x.device != weight.device:
        raise ValueError("dense linear expects input [rows, K] and weight [N, K] on one device")
    n, k = weight.shape
    if x.shape[0] > 16:
        if x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.float16:
            out = torch.empty((x.shape[0], n), dtype=dtype, device=x.device)
            _float_mm[(triton.cdiv(x.shape[0], 128), triton.cdiv(n, 64))](
                x.contiguous(),
                weight.contiguous(),
                out,
                x.shape[0],
                n,
                k,
                num_warps=4,
                num_stages=3,
                enable_fp_fusion=False,
            )
            return out
        # Other operand formats require FP32 projections; TF32 stays disabled.
        return (x.float() @ weight.float().T).to(dtype)
    out = torch.empty((x.shape[0], n), dtype=dtype, device=x.device)
    _float_mv[(x.shape[0], triton.cdiv(n, 4))](
        x.contiguous(), weight.contiguous(), out, n, k, 256, 4, enable_fp_fusion=False
    )
    return out
