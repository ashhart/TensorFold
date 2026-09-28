"""The EXL3 Flash Next matrices: trellis projections, the pack's fp16 tensors, side-by-side stacks and their shared scratch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3 import prefill as x3prefill
from tensorfold.cuda.exl3.linear import Exl3Linear

from .exl3_pack import MOE_WINDOW

ROWS = 128               # most rows one call of the EXL3 linear or a split fp16 matmul takes


class Scratch:
    """Buffers for every EXL3 and fp16 matrix of the model (calls run one after another on one stream)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots                 # routed experts a row, plus the shared expert
        self.users: list[Any] = []
        self.xh = self.z = self.tmp = self.part = None
        self.moe: x3experts.Scratch | None = None
        self.ple_host = self.ple_dev = self.ple_emb = None
        self.prefill = x3prefill.Workspace()

    def allocate(self, device, *, experts: x3experts.Exl3RoutedExperts, rows: int, ple_words: int, ple_heads: int,
                 ple_dim: int) -> None:
        """Size the buffers for the model's matrices, routed windows of ``MOE_WINDOW`` rows and n-gram rows for ``rows``."""

        xh = z = tmp = part = 1
        for u in self.users:
            if isinstance(u, X3):
                xh = max(xh, ROWS * u.lin.k)
                if u.lin.split[0] > 1:
                    z = max(z, u.lin.split[0] * ROWS * u.lin.n)
            elif isinstance(u, F16):
                part = max(part, u.sk * ROWS * u.n)
            elif isinstance(u, Stack) and len(u.parts) > 1:
                tmp = max(tmp, ROWS * max(p.n for p in u.parts))
        self.xh = torch.empty((xh,), dtype=torch.float16, device=device)
        self.z = torch.empty((z,), dtype=torch.float32, device=device)
        self.tmp = torch.empty((tmp,), dtype=torch.bfloat16, device=device)
        self.part = torch.empty((part,), dtype=torch.float32, device=device)
        self.moe = x3experts.Scratch(experts, MOE_WINDOW, self.slots, device=device)
        if ple_words:
            pin = torch.cuda.is_available()
            self.ple_host = torch.zeros((rows * ple_heads, ple_words), dtype=torch.int16, pin_memory=pin)
            self.ple_dev = torch.zeros((rows * ple_heads, ple_words), dtype=torch.int16, device=device)
            self.ple_emb = torch.empty((rows, ple_dim), dtype=torch.float16, device=device)

    def nbytes(self) -> int:
        own = [self.xh, self.z, self.tmp, self.part, self.ple_dev, self.ple_emb]
        moe = [] if self.moe is None else [self.moe.xg, self.moe.xu, self.moe.xd, self.moe.z, self.moe.y,
                                          self.moe.ids, self.moe.members_buf]
        return sum(t.numel() * t.element_size() for t in own + moe if t is not None) + self.prefill.nbytes()


@triton.jit(do_not_specialize=["M"])
def _f16_mm(X, W, OUT, M, N, x_stride, o_stride, K: tl.constexpr, KS: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    """Program (m block, n block, K slice): fp32 sums of fp16(x) * w over the slice's K in BK steps; OUT or fp32 slices [KS, M, N]."""

    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ps = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * KS
    for kk in range(0, KS, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + kk + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + kk + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x.to(tl.float16), tl.trans(w), acc)
    if F32:
        tl.store(OUT + ps * M * N + rm[:, None] * N + rn[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
    else:
        tl.store(OUT + rm[:, None] * o_stride + rn[None, :], acc.to(OUT.dtype.element_ty),
                 mask=m_ok[:, None] & n_ok[None, :])


@triton.jit(do_not_specialize=["M"])
def _reduce(P, OUT, M, N, o_stride, SK: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SK):
        acc += tl.load(P + s * M * N + r * N + c, mask=ok, other=0.0)
    tl.store(OUT + r * o_stride + c, acc.to(OUT.dtype.element_ty), mask=ok)


def f16_split(n: int, k: int, target: int = 96) -> int:
    """K slices for an (n, k) fp16 matrix: a function of the shape only."""

    tiles = -(-n // 64)
    sk = 1
    while sk < 32 and tiles * sk < target and k % (sk * 2 * 64) == 0 and k // (sk * 2) >= 256:
        sk *= 2
    return sk


@dataclass
class F16:
    """y = x @ w.T for a matrix the pack leaves unquantized (fp16): fixed tiles and K slices, so a row is its own."""

    w: torch.Tensor
    n: int
    k: int
    sk: int
    sc: Scratch = field(repr=False)

    def nbytes(self) -> int:
        return self.w.numel() * 2

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """Unreduced fp32 slices [sk, M, n] (shared scratch) for a consumer that sums them in order; M <= ROWS."""

        m = x.shape[0]
        if m > ROWS:
            raise ValueError(f"F16.partials takes at most {ROWS} rows")
        out = self.sc.part[:self.sk * m * self.n].view(self.sk, m, self.n)
        self._launch(x, out, True)
        return out

    def _launch(self, x: torch.Tensor, out: torch.Tensor, f32: bool) -> None:
        m = x.shape[0]
        if x.stride(1) != 1 or x.shape[1] != self.k:
            raise ValueError(f"F16: x {tuple(x.shape)} does not match K={self.k}")
        grid = (triton.cdiv(m, 16), triton.cdiv(self.n, 64), self.sk)
        _f16_mm[grid](x, self.w, out, m, self.n, x.stride(0), self.n if f32 else out.stride(0),
                      K=self.k, KS=self.k // self.sk, BM=16, BN=64, BK=64, F32=f32, num_warps=4, num_stages=3)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if out.stride(-1) != 1:
            raise ValueError("F16: output rows must be contiguous")
        if self.sk == 1:
            self._launch(x, out, False)
            return out
        for r0 in range(0, x.shape[0], ROWS):                 # rows are independent, so slices keep their bits
            xs, m = x[r0:r0 + ROWS], min(ROWS, x.shape[0] - r0)
            part = self.sc.part[:self.sk * m * self.n].view(self.sk, m, self.n)
            self._launch(xs, part, True)
            _reduce[(m, triton.cdiv(self.n, 256))](part, out[r0:r0 + m], m, self.n, out.stride(0), SK=self.sk,
                                                   BLOCK=256, num_warps=2)
        return out

    prefill = __call__


@dataclass
class X3:
    """A trellis projection: the row-invariant EXL3 linear in 128-row calls, or the prompt GEMM (``prefill``)."""

    lin: Exl3Linear
    sc: Scratch = field(repr=False)
    head: bool = False       # read one row a prompt: the decode linear, so the prompt path needs no decoded copy

    @property
    def n(self) -> int:
        return self.lin.n

    @property
    def k(self) -> int:
        return self.lin.k

    def nbytes(self) -> int:
        lin = self.lin
        parts = [lin.words, lin.suh, lin.svh, lin.counters] + ([lin.bias] if lin.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in parts)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        lin, sk = self.lin, self.lin.split[0]
        for r0 in range(0, x.shape[0], ROWS):
            r1 = min(x.shape[0], r0 + ROWS)
            rows = r1 - r0
            lin(x[r0:r1].contiguous(), out=out[r0:r1], xh=self.sc.xh[:rows * lin.k].view(rows, lin.k),
                z=self.sc.z[:sk * rows * lin.n] if sk > 1 else None)
        return out

    def prefill(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.head:
            return self(x, out)
        return x3prefill.matmul(self.lin, x, out, self.sc.prefill)


def x3(sc: Scratch, pk, prefix: str, device, *, head: bool = False) -> X3:
    lin = Exl3Linear.from_tensors(pk.get(prefix + ".trellis"), pk.scales(prefix, "su", "suh"),
                                  pk.scales(prefix, "sv", "svh"), pk.codebook(prefix),
                                  pk.get(prefix + ".bias") if pk.has(prefix + ".bias") else None, device)
    got = X3(lin, sc, head)
    sc.users.append(got)
    return got


def f16(sc: Scratch, rows: list[torch.Tensor], device) -> F16:
    w = torch.cat([r.to(torch.float16) for r in rows]).to(device).contiguous()
    n, k = w.shape
    got = F16(w, n, k, f16_split(n, k), sc)
    sc.users.append(got)
    return got


@dataclass
class Stack:
    """Matrices reading the same input, written side by side into one output (the MLX path's stacked Q4)."""

    parts: list[Any]
    sc: Scratch = field(repr=False)

    @property
    def n(self) -> int:
        return sum(p.n for p in self.parts)

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self.parts)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if len(self.parts) == 1 and out.is_contiguous():
            return self.parts[0](x, out)
        m = x.shape[0]
        if m == 1:                    # one row: a part's slice of ``out`` needs no copy (its row stride is unused)
            at = 0
            for p in self.parts:
                p(x, out[:, at:at + p.n])
                at += p.n
            return out
        for r0 in range(0, m, ROWS):
            xs, rows = x[r0:r0 + ROWS], min(ROWS, m - r0)
            at = 0
            for p in self.parts:
                tmp = self.sc.tmp[:rows * p.n].view(rows, p.n)
                p(xs, tmp)
                out[r0:r0 + rows, at:at + p.n].copy_(tmp)
                at += p.n
        return out

    def prefill(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        at = 0
        for p in self.parts:
            p.prefill(x, out[:, at:at + p.n])
            at += p.n
        return out


def stack(sc: Scratch, parts: list[Any]) -> Stack:
    got = Stack(parts, sc)
    sc.users.append(got)
    return got


@triton.jit
def _embed(IDS, T, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(IDS + r).to(tl.int64)
    v = tl.load(T + t * D + d).to(tl.bfloat16)
    for s in tl.static_range(S):
        tl.store(OUT + r * (S * D) + s * D + d, v)


def embed(ids: torch.Tensor, table: torch.Tensor, dims: int, copies: int, out: torch.Tensor) -> torch.Tensor:
    """ids (R,) int32 -> out (R, copies * dims) bf16: the unquantized embedding row, repeated per stream."""

    _embed[(ids.shape[0], dims // 256)](ids, table, out, D=dims, S=copies, BLOCK=256, num_warps=2)
    return out


@triton.jit
def _ple_rows(PK, HB, OUT, WORDS: tl.constexpr, KB: tl.constexpr, HEADS: tl.constexpr, DH: tl.constexpr,
              BLOCK: tl.constexpr, K_INV: tl.constexpr, K_BIAS: tl.constexpr):
    """Row r * HEADS + h (an fp16 scale word, then DH tail-biting KB-bit values) -> fp16(fp16(mul1(state)) * scale + bias[h])."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    row = (r * HEADS + h).to(tl.int64)
    base = PK + row * WORDS
    i = tl.arange(0, BLOCK)
    ok = i < DH
    scale = tl.load(base).to(tl.float16, bitcast=True).to(tl.float32)
    state = tl.zeros((BLOCK,), dtype=tl.int64)
    for m in tl.static_range(16):
        pos = i - (m // KB)
        pos = tl.where(pos < 0, pos + DH, pos)
        sb = pos * KB + (m % KB)
        word = tl.load(base + 1 + (sb >> 4), mask=ok, other=0).to(tl.int64) & 0xFFFF
        state |= ((word >> (sb & 15)) & 1) << m
    prod = (state * 0x83DCD12D) & 0xFFFFFFFF
    hs = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    cb = ((1024 + hs).to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
    b = tl.load(HB + h * DH + i, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * (HEADS * DH) + h * DH + i, (cb * scale + b).to(tl.float16), mask=ok)


def _fp16_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


K_INV, K_BIAS = _fp16_bits(0x1EEE), _fp16_bits(0xC931)


def ple_rows(rows: int, packed: torch.Tensor, head_bias: torch.Tensor, heads: int, dh: int, bits: int,
             out: torch.Tensor) -> torch.Tensor:
    """Staged table rows (row r * heads + h) -> out [rows, heads * dh] fp16, as ExLlamaV3's ``ngram_dequant``."""

    _ple_rows[(rows, heads)](packed, head_bias, out, WORDS=packed.shape[1], KB=bits, HEADS=heads, DH=dh,
                             BLOCK=triton.next_power_of_2(dh), K_INV=K_INV, K_BIAS=K_BIAS, num_warps=4)
    return out
