"""The 27B's prefill: bits never depend on chunking but differ from decode's, so only prompt-end states resume."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.cuda.kernels.prefill_attention import attention

from . import glue
from . import prefill_bf16, prefill_glue
from .forward import State
from .qmm_fast import tile
from .weights import QLinear, Weights

CHUNK = 4096
TAP_LAYERS = (5, 19, 33, 47, 61)


def _mm(x, w: QLinear, f32: bool = False) -> torch.Tensor:
    """``x``: a quantized input (e4m3, group sums, row scales) from ``prefill_glue``, or bf16 rows for an EXL3 pack."""

    if not isinstance(w, QLinear):
        return w.prefill(x)
    return shared.prefill_matmul8(x, tile(w), f32=f32)


def _row_mm(x, w: QLinear, tp: bool) -> torch.Tensor:
    if not tp:
        return _mm(x, w)
    from .distributed import gather_rank_partials

    return gather_rank_partials(_mm(x, w))                 # bf16 partials: half the bytes of fp32 over the link


def _grow(st: State, i: int, need: int) -> tuple[torch.Tensor, torch.Tensor]:
    kbuf, vbuf = st.kv[i]
    if kbuf.shape[0] < need:
        cap = max(need, 2 * kbuf.shape[0], 1024)
        if st.limit:
            cap = max(need, min(cap, st.limit))
        grown_k, grown_v = kbuf.new_empty((cap, *kbuf.shape[1:])), vbuf.new_empty((cap, *vbuf.shape[1:]))
        grown_k[:st.pos] = kbuf[:st.pos]
        grown_v[:st.pos] = vbuf[:st.pos]
        st.kv[i] = (grown_k, grown_v)
    return st.kv[i]


@torch.no_grad()
def prefill_chunk(w: Weights, tokens: torch.Tensor, st: State, *, tp: bool = False, capture_taps: bool = False,
                  last: bool = True):
    """Commit ``tokens`` at [st.pos, st.pos + W) into ``st``, replacing its list entries, never writing through them."""

    c = w.config
    pg = prefill_bf16 if w.quant == "exl3" else prefill_glue
    W = int(tokens.shape[0])
    p0 = st.pos
    keep = c.conv_kernel - 1
    dev = tokens.device
    pos = torch.arange(p0, p0 + W, device=dev, dtype=torch.int32)
    windows = (torch.arange(W, device=dev, dtype=torch.int32)[:, None]
               + torch.arange(keep + 1, device=dev, dtype=torch.int32)[None, :])
    x = glue.embedding(tokens.to(torch.int32), w.embed)
    pending: torch.Tensor | None = None
    taps: list[torch.Tensor] = []
    for i, layer in enumerate(w.layers):
        x, h = pg.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z).reshape(W, c.v_heads, c.dv)
                b = _mm(h, gdn.b)
                a = _mm(h, gdn.a)
            q, k, v, g, beta = glue.gdn_pre(qkv, st.conv[i], gdn.conv, windows, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk)
            final = torch.empty_like(st.rec[i])
            yr = deltanet.chain(q, k, v, g, beta, st.rec[i], final)
            r = _row_mm(pg.gated_norm(yr, z, gdn.norm, c.eps), gdn.out, tp)
            st.conv[i] = torch.cat([st.conv[i], qkv[-keep:]])[-keep:].contiguous()
            st.rec[i] = final
        else:
            attn = layer.attn
            qg = _mm(h, attn.q)
            if attn.kv is not None:
                kv = _mm(h, attn.kv)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k)
                value = _mm(h, attn.v).reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim)
            kbuf, vbuf = _grow(st, i, p0 + W)
            kbuf[p0:p0 + W] = key.view(W, c.kv_heads, c.head_dim)
            vbuf[p0:p0 + W] = value
            out = attention(q.view(W, c.heads, c.head_dim), kbuf, vbuf, p0, scale=c.head_dim ** -0.5)
            r = _row_mm(pg.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim), attn.o, tp)
        x, h = pg.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _row_mm(pg.swiglu(_mm(h, layer.gate), _mm(h, layer.up)), layer.down, tp)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    st.pos = p0 + W
    normed = None
    if last:
        _, normed, _ = glue.add_rmsnorm(x[-1:].contiguous(), pending[-1:].contiguous(), w.norm, c.eps)
    return normed, (torch.cat(taps, dim=-1) if capture_taps else None)


def chunks(start: int, end: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """Even chunks of at most ``size`` rows (a short one costs a whole weight pass); any bounds give the same bits."""

    n = -(-(end - start) // size)
    return [(start + (end - start) * j // n, start + (end - start) * (j + 1) // n) for j in range(n)] if n else []


@torch.no_grad()
def prefill_state(w: Weights, prompt: Sequence[int], st: State, *, tp: bool = False, draft=None,
                  size: int = CHUNK) -> torch.Tensor:
    """Commit prompt[st.pos:] into ``st``; the drafter gets taps only for rows its window keeps at the prompt's end."""

    dev = w.norm.device
    ids = torch.tensor(list(prompt[st.pos:]), dtype=torch.int32, device=dev)
    base, normed = st.pos, None
    tap_from = base
    if draft is not None and len(prompt) - draft.window > base:
        tap_from = len(prompt) - draft.window
        draft.skip(tap_from - base)
    spans = chunks(base, len(prompt), size)
    for j, (a, b) in enumerate(spans):
        want = draft is not None and b > tap_from
        normed, taps = prefill_chunk(w, ids[a - base:b - base], st, tp=tp, capture_taps=want, last=j == len(spans) - 1)
        if want:
            draft.add_taps(taps[max(0, tap_from - a):])
    return normed
