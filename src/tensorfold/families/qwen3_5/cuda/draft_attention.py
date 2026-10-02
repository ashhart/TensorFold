"""DFlash2 block attention for every stream in one launch: each block reads its stream's context keys in place."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import torch

from tensorfold.cuda.build import volta
import triton
import triton.language as tl


@triton.jit
def _block_attention(Q, KB, VB, TABLE, LENS, O, scale, window, R,
                     G: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, D: tl.constexpr,
                     BN: tl.constexpr, CAUSAL: tl.constexpr):
    """Program (stream j, kv head g): G query heads x L block rows against the context keys a row's window keeps, then the block's keys."""

    j = tl.program_id(0)
    g = tl.program_id(1)
    M: tl.constexpr = G * LP
    rows = tl.arange(0, M)
    head = g * G + rows // LP
    r = rows % LP
    live = r < L
    d = tl.arange(0, D)
    qrow = (j * L + r).to(tl.int64)
    q = tl.load(Q + head[:, None].to(tl.int64) * R * D + qrow[:, None] * D + d[None, :], mask=live[:, None], other=0.0)
    s = tl.load(LENS + j)
    # contexts are fresh torch tensors (16-byte aligned): the hint makes their loads 16 bytes wide, same arithmetic
    kc = tl.multiple_of(tl.load(TABLE + 2 * j).to(tl.pointer_type(tl.bfloat16)), 16) + g.to(tl.int64) * s * D
    vc = tl.multiple_of(tl.load(TABLE + 2 * j + 1).to(tl.pointer_type(tl.bfloat16)), 16) + g.to(tl.int64) * s * D
    m_i = tl.full((M,), float("-inf"), tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    for n0 in range(0, s, BN):
        kidx = n0 + tl.arange(0, BN)
        inside = kidx < s
        k = tl.load(kc + kidx[None, :] * D + d[:, None], mask=inside[None, :], other=0.0)
        qk = tl.dot(q, k) * scale
        seen = inside[None, :] & (s + r[:, None] - kidx[None, :] < window + 1)
        qk = tl.where(seen, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc + kidx[:, None] * D + d[None, :], mask=inside[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    cols = tl.arange(0, LP)
    krow = (j * L + cols).to(tl.int64)
    in_block = cols < L
    kb = tl.load(KB + g.to(tl.int64) * R * D + krow[None, :] * D + d[:, None], mask=in_block[None, :], other=0.0)
    qk = tl.dot(q, kb) * scale
    seen = in_block[None, :]
    if CAUSAL:
        seen = seen & (cols[None, :] <= r[:, None])
    qk = tl.where(seen, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp(qk - m_safe[:, None])
    alpha = tl.exp(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, 1)
    vb = tl.load(VB + g.to(tl.int64) * R * D + krow[:, None] * D + d[None, :], mask=in_block[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
    out = (acc / l_i[:, None]).to(tl.bfloat16)
    tl.store(O + qrow[:, None] * (G * HKV * D) + head[:, None] * D + d[None, :], out, mask=live[:, None])


@lru_cache(maxsize=1)
def _volta_ext():
    """The sm_70 block attention (draft_attention_volta.cu): tl.dot has no tensor-core path on Volta."""

    from pathlib import Path

    from tensorfold.cuda.build import VOLTA, load

    return load(name="tensorfold_draft_attention_volta_v1", need=VOLTA,
                sources=[str(Path(__file__).parent / "draft_attention_volta.cu")], extra_cuda_cflags=["-O3"], verbose=False)


def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keys: Sequence[torch.Tensor],
                    values: Sequence[torch.Tensor], length: int, window: int, scale: float,
                    causal: bool = False) -> torch.Tensor:
    """q [H, S*L, D], k and v [Hkv, S*L, D] (S streams' blocks of ``length`` rows), stream s's context [Hkv, n_s, D] -> [S*L, H*D] bf16."""

    heads, rows, dim = q.shape
    kv_heads = k.shape[0]
    streams = rows // length
    if len(keys) != streams or len(values) != streams or rows != streams * length or heads % kv_heads:
        raise ValueError("one context a stream, and blocks of equal length")
    for kc, vc in zip(keys, values):
        if kc.shape != vc.shape or kc.shape[0] != kv_heads or kc.shape[2] != dim or not kc.is_contiguous() \
                or not vc.is_contiguous() or kc.dtype != torch.bfloat16 or (kc.data_ptr() | vc.data_ptr()) % 16:
            raise ValueError("contexts are contiguous, 16-byte aligned bf16 [Hkv, n, D] keys and values")
    host = torch.tensor([p for kc, vc in zip(keys, values) for p in (kc.data_ptr(), vc.data_ptr())] +
                        [kc.shape[1] for kc in keys], dtype=torch.int64).pin_memory()
    dev = host.to(q.device, non_blocking=True)
    table, lens = dev[:2 * streams], dev[2 * streams:].to(torch.int32)
    if volta(q.device):
        lens64 = dev[2 * streams:]
        return _volta_ext().block_attention(q.contiguous(), k.contiguous(), v.contiguous(), table, lens64, length,
                                            window, scale, causal, max(kc.shape[1] for kc in keys))
    out = torch.empty((rows, heads * dim), dtype=torch.bfloat16, device=q.device)
    group = heads // kv_heads
    _block_attention[(streams, kv_heads)](q, k, v, table, lens, out, scale, window, rows, G=group, HKV=kv_heads,
                                          L=length, LP=max(16, triton.next_power_of_2(length)), D=dim,
                                          BN=64, CAUSAL=causal, num_warps=4, num_stages=2)
    return out


@triton.jit
def _append(TABLE, SIZES, NEW, R, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """Program (stream, head, row block): out rows are the last ones of [context | its new rows], copied once."""

    j = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(2) * BR + tl.arange(0, BR)
    old_n = tl.load(SIZES + 4 * j)
    add = tl.load(SIZES + 4 * j + 1)
    first = tl.load(SIZES + 4 * j + 2)
    keep = tl.load(SIZES + 4 * j + 3)
    old = tl.multiple_of(tl.load(TABLE + 2 * j).to(tl.pointer_type(tl.bfloat16)), 16)
    out = tl.multiple_of(tl.load(TABLE + 2 * j + 1).to(tl.pointer_type(tl.bfloat16)), 16)
    d = tl.arange(0, D)
    live = rows < keep
    src = rows + old_n + add - keep
    from_old = src < old_n
    a = tl.load(old + (h * old_n + src)[:, None] * D + d[None, :], mask=(live & from_old)[:, None], other=0.0)
    b = tl.load(NEW + (h * R + first + src - old_n)[:, None] * D + d[None, :], mask=(live & ~from_old)[:, None],
                other=0.0)
    tl.store(out + (h * keep + rows)[:, None] * D + d[None, :], tl.where(from_old[:, None], a, b), mask=live[:, None])


def append(new: torch.Tensor, olds: Sequence[torch.Tensor | None], sizes: Sequence[int], window: int) -> list[torch.Tensor]:
    """new [Hkv, R, D] (each stream's rows in turn) after each stream's context [Hkv, n, D] -> its last ``window`` rows."""

    heads, rows, dim = new.shape
    outs, table, meta, first = [], [], [], 0
    for old, add in zip(olds, sizes):
        if old is not None and (not old.is_contiguous() or old.data_ptr() % 16):
            raise ValueError("a context to extend is a contiguous, 16-byte aligned [Hkv, n, D] tensor")
        n = 0 if old is None else old.shape[1]
        keep = min(window, n + add)
        out = torch.empty((heads, keep, dim), dtype=new.dtype, device=new.device)
        table += [out.data_ptr() if old is None else old.data_ptr(), out.data_ptr()]
        meta += [n, add, first, keep]
        outs.append(out)
        first += add
    host = torch.tensor(table + meta, dtype=torch.int64).pin_memory()
    dev = host.to(new.device, non_blocking=True)
    streams = len(outs)
    _append[(streams, heads, triton.cdiv(max(o.shape[1] for o in outs), 64))](dev[:2 * streams], dev[2 * streams:], new,
                                                                              rows, H=heads, D=dim, BR=64, num_warps=4)
    return outs
