"""A KV source's compressor: pooled latents, their index keys and compressed entries, stored by position."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.families.glm5_next.cuda import qmm

from . import norms, quant, rope
from .buffers import Buffers, State
from .norms import _normed
from .weights import LayerW


@triton.jit
def _pool(CMP, TAIL, W, LAT, EPOS, POS, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    """Pooling (M:473-485): row r at an odd q pools group q // 2 from row r - 1 (or the committed tail) and itself."""

    r = tl.program_id(0)
    q = tl.load(POS) + r
    d = tl.arange(0, BLOCK)
    ok = d < D
    cur = CMP + r.to(tl.int64) * (2 * D)
    kv1 = tl.load(cur + d, mask=ok, other=0.0)
    z1 = tl.load(cur + D + d, mask=ok, other=0.0)
    if r > 0:
        kv0 = tl.load(cur - 2 * D + d, mask=ok, other=0.0)
        z0 = tl.load(cur - D + d, mask=ok, other=0.0)
    else:
        kv0 = tl.load(TAIL + d, mask=ok, other=0.0)
        z0 = tl.load(TAIL + D + d, mask=ok, other=0.0)
    m = tl.maximum(z0, z1)                          # softmax over the group's two scores, per channel
    e0 = libdevice.exp(z0 - m)
    e1 = libdevice.exp(z1 - m)
    s = e0 + e1
    pooled = kv0 * tl.math.div_rn(e0, s) + kv1 * tl.math.div_rn(e1, s)
    pooled = tl.where(q % 2 == 1, pooled, 0.0)
    lat = pooled.to(tl.bfloat16).to(tl.float32)
    tl.store(LAT + r.to(tl.int64) * D + d, _normed(lat, W, d, ok, eps, D), mask=ok)
    tl.store(EPOS + r, q - q % 2)                   # the group's first position, where its entry is rotated


@triton.jit
def _store(LAT, KI, COMP, INDEX_K, POS, RATIO: tl.constexpr, D: tl.constexpr, DK: tl.constexpr,
           BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Row r's entry and index key into slot (pos + r) // RATIO when its position completes a group."""

    r = tl.program_id(0)
    q = tl.load(POS) + r
    if q % RATIO == RATIO - 1:
        j = (q // RATIO).to(tl.int64)
        d = tl.arange(0, BLOCK)
        tl.store(COMP + j * D + d, tl.load(LAT + r.to(tl.int64) * D + d, mask=d < D), mask=d < D)
        k = tl.arange(0, BLOCK_K)
        tl.store(INDEX_K + j * DK + k, tl.load(KI + r.to(tl.int64) * DK + k, mask=k < DK), mask=k < DK)


def compress(lw: LayerW, xa: torch.Tensor, state: State, buf: Buffers, table: torch.Tensor, eps: float) -> None:
    """Pool, index-key and entry rows of KV source ``lw`` from ``xa`` [R, D] at ``state.pos_dev`` + r."""

    layer, comp, idx = lw.index, lw.attn.comp, lw.attn.idx
    rows, hd = xa.shape[0], comp.norm.shape[0]
    if comp.ratio != state.ratio.get(layer) or idx is None or idx.wk is None:
        raise ValueError(f"layer {layer}: not a KV source of ratio {comp.ratio} with an index-K projection")
    if not 0 < rows <= buf.rows or state.pos + rows > state.capacity:
        raise ValueError(f"compress: {rows} rows at {state.pos} for {buf.rows}-row buffers, {state.capacity} slots")
    lat, ki = buf.lat[:rows], buf.kI[:rows]
    block = triton.next_power_of_2(hd)
    if comp.ratio == 2:
        cmp = buf.cmp[state.pooled.index(layer), :rows]
        qmm.matmul(xa, comp.wkv, out=cmp.view(rows, 2 * hd), f32=True)
        _pool[(rows,)](cmp, state.tail[state.pooled.index(layer)], comp.norm, lat, buf.epos, state.pos_dev, eps,
                       D=hd, BLOCK=block, num_warps=4, enable_fp_fusion=False)
        positions = buf.epos[:rows]
    else:
        qmm.matmul(xa, comp.wkv, out=lat)
        norms.rmsnorm(lat, comp.norm, eps, lat)
        positions = state.pos_dev                   # entry j = q, rotated at q
    qmm.matmul(lat, idx.wk, out=ki)                 # the index key reads the latent before RoPE (M:744, 749-750)
    norms.rmsnorm(ki, idx.k_norm, eps, ki)
    quant.fp4_qdq_1x32_e8m0(rope.apply(ki, positions, table))
    quant.fp4_qdq_1x16_e4m3(rope.apply(lat, positions, table))
    dk = ki.shape[1]
    _store[(rows,)](lat, ki, state.comp[layer], state.index_k[layer], state.pos_dev, RATIO=comp.ratio, D=hd, DK=dk,
                    BLOCK=block, BLOCK_K=triton.next_power_of_2(dk), num_warps=4)


def commit_tail(state: State, buf: Buffers, pos: int, keep: int) -> None:
    """After a forward at ``pos`` keeps ``keep`` rows: an open group's first row becomes each ratio-2 source's tail."""

    if not state.pooled:
        return
    if (pos + keep) % 2:
        state.tail.copy_(buf.cmp[:, keep - 1])
        state.tail_valid.fill_(1)
    else:
        state.tail_valid.zero_()
