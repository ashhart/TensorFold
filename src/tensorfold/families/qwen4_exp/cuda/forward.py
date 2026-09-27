"""Flash Next forward on CUDA: every kernel treats each row alone, so a window row equals the serial step."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda import moe as moe_mod
from tensorfold.cuda.kernels import gdn as shared_gdn

from . import attention as attn_mod
from . import gdn as gdn_mod
from . import gdn_io, glue, qmm
from .state import ATT_ROWS, CAND, Buffers, State
from .weights import HC, LayerW, Weights


def _gather(w: Weights, b: Buffers, part: torch.Tensor, flat: torch.Tensor, R: int) -> torch.Tensor:
    """All ranks' fp32 partials [R, D] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = part.shape[1]
    out = flat[:b.world * R * d]
    w.comm.all_gather(part[:R], out)
    return out.view(b.world, R, d)


def _mm(x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, out: torch.Tensor, b: Buffers, **kw) -> torch.Tensor:
    if not isinstance(q, qmm.Q4):                 # an EXL3 pack's matrix (``exl3_mm``): prompts on its prompt path
        return q.prefill(x, out) if b.prefill else q(x, out)
    mm = qmm.prefill_matmul if b.prefill else qmm.matmul
    return mm(x, q, xs, out=out, part=b.part, **kw)


def _embed(w: Weights, ids: torch.Tensor, copies: int, out: torch.Tensor) -> torch.Tensor:
    if len(w.embed) == 1:     # an EXL3 checkpoint's unquantized embedding
        from .exl3_mm import embed

        return embed(ids, w.embed[0], w.cfg.hidden, copies, out)
    return glue.embed(ids, *w.embed, w.cfg.hidden, copies=copies, out=out)


def hc_block(hc: HC, b: Buffers, R: int, eps: float, streams: int, low: int, mode: int, inject_prev,
             inject_out, h: torch.Tensor, branch=None, y=None, wts=None) -> None:
    """Write the pending branch back into the streams h (in place), then the hyper-connection's read-out: b.mixed [R, D] (+ group sums), and its inject gates into ``inject_out``."""

    glue.hc_writeback(h[:R], h[:R], b.pss[:R], streams, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
    _readout(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None)


FUSED_ROWS = 16      # decode windows: the read-out in 3 kernels; wider windows (prefill) in 5, the same bits


def _readout(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """normed streams -> down -> SiLU / inject -> up -> mix: b.mixed [R, D] and its group sums."""

    if R <= FUSED_ROWS and not b.prefill and isinstance(hc.down, qmm.Q4):
        _readout_fused(hc, b, h, R, eps, streams, low, inject)
    else:
        _readout_plain(hc, b, h, R, eps, streams, low, inject)


def _readout_fused(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The norm inside the down projection, the mix inside the up projection."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    got = qmm.hc_down(h[:R], b.pss[:R], hc.scale, b.normed[:R], hc.down, eps, streams, out=out, part=b.part)
    if got.dim() == 3:
        glue.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    qmm.hc_upmix(b.act[:R], b.xs_act[:R], hc.up, b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_plain(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The norm, the down projection with SiLU and the inject gates, the up projection, the mix: separate kernels."""

    glue.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps)
    _down_act(hc, b, R, streams, low, inject)
    _mm(b.act[:R], hc.prefill_up if b.prefill else hc.up, b.xs_act[:R], b.up[:R], b)
    glue.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _down_act(hc: HC, b: Buffers, R: int, streams: int, low: int, inject) -> None:
    """A hyper-connection's down projection, then SiLU and the inject gates: b.act, b.xs_act (and ``inject``). With a split K the slice sum is fused into the activation kernel (the same bits as reduce, then act)."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    if isinstance(hc.down, qmm.Q4):
        got = _mm(b.normed[:R], hc.prefill_down if b.prefill else hc.down, b.xs_normed[:R], out, b, reduce=False)
    elif b.prefill:                                   # an EXL3 pack's fp16 matrix: summed slices, any row count
        got = hc.down(b.normed[:R], out)
    else:
        got = hc.down.partials(b.normed[:R])          # fp32 slices [SK, R, N] that the activation sums in order
    if got.dim() == 3:
        glue.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)


Seg = tuple[State, int, int]     # a stream's committed state and its rows [a0, a1) of the window


def gdn_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int) -> None:
    c = w.cfg
    g = layer.gdn
    li = segs[0][0].lin_index[layer.index]
    if b.prefill:
        _mm(b.mixed[:R], g.proj, b.xs_mixed[:R], b.proj[0, :R], b)
        for st, a0, a1 in segs:
            _prefill_chain(g, st, li, b, a0, a1, c)
        return _out_proj(w, b, b.gout[:R], g.out, b.gxs[:R], R)
    _mm(b.mixed[:R], g.proj, b.xs_mixed[:R], b.proj[li, :R], b)
    for st, a0, a1 in segs:
        cur = st.cur[li]
        gdn_mod.chain(b.proj[li, a0:a1], st.conv[li], g.conv, st.rec[cur, li], g.a_log, g.dt_bias, g.norm, c.eps,
                      a1 - a0, st.scratch[li], st.rec[1 - cur, li], b.gout[a0:a1], b.gxs[a0:a1])
    return _out_proj(w, b, b.gout[:R], g.out, b.gxs[:R], R)


def _prefill_chain(g, st: State, li: int, b: Buffers, a0: int, a1: int, c) -> None:
    """A prompt chunk's DeltaNet; the layer commits at once (a chunk keeps every row)."""

    n, p, cur = a1 - a0, b.proj[0, a0:a1], st.cur[li]
    b.conv_ptr.fill_(st.conv[li].data_ptr())
    q, k, v, gt, beta = gdn_io.front(p, b.conv_ptr, b.sid[:n], b.windows[:n], g.conv, g.a_log, g.dt_bias, c.nk)
    y = shared_gdn.chain(q, k, v, gt, beta, st.rec[cur, li], st.rec[1 - cur, li])
    gdn_io.back(y, p, g.norm, c.eps, b.gout[a0:a1], b.gxs[a0:a1])
    st.cur[li] = 1 - cur
    shift_windows(st.conv[li:li + 1], b.proj[0:1, a0:a1], n, c.conv_dim)


def _out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int):
    """A block's output projection: (1, bf16 branch) on one GPU; (3, gathered fp32 partials) across ranks."""

    if not isinstance(q, qmm.Q4):                     # an EXL3 pack (one GPU): the bf16 branch
        return 1, _mm(x, q, xs, b.branch[:R], b)
    if w.comm is None:
        got = _mm(x, q, xs, b.branch[:R], b, reduce=False)
        if got.dim() == 3:
            return 4, got            # K slices: the write-back sums them in order (the bits of reduce, then round)
        return 1, got
    _mm(x, q, xs, b.part_branch[:R], b, f32=True)
    return 3, _gather(w, b, b.part_branch, b.g_branch, R)


def _caches(layer: LayerW, st: State, mtp: bool) -> tuple:
    """The layer's caches and committed length (on the device and on the host)."""

    if mtp:
        return st.mtp_kc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, st.mtp_len
    ai = st.att_index[layer.index]
    return st.kc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, st.pos


def attn_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, mtp: bool,
               context: int | None = None):
    """Attention over each stream's own caches; ``context`` bounds the launches (a graph's bucket)."""

    c = w.cfg
    a = layer.attn
    _mm(b.mixed[:R], a.proj, b.xs_mixed[:R], b.pa[:R], b)
    scale = c.head_dim ** -0.5
    for st, a0, a1 in segs:
        cache, ikc, pooled, pos, host_pos = _caches(layer, st, mtp)
        keys = context if context is not None else host_pos + a1 - a0
        glue.attn_prep(b.pa[a0:a1], pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, b.q[a0:], cache.k, cache.v,
                       b.iq[a0:], ikc, c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim,
                       index_heads=c.index_heads, index_dim=c.index_dim, ks=cache.ks, vs=cache.vs,
                       kvq=cache.quantized)
        if b.prefill:
            if b.attn.qsa:
                attn_mod.qsa_pool(ikc, pooled, pos, a.ik_scale, w.inv_freq, c.eps, b.attn, a1 - a0)
            for r0 in range(a0, a1, ATT_ROWS):
                n = min(ATT_ROWS, a1 - r0)
                b.pos_blk.fill_(host_pos + r0 - a0)
                ends = host_pos + r0 - a0 + n
                if b.attn.qsa:
                    attn_mod.qsa_rows(b.iq[r0:r0 + n], pooled, b.pos_blk, b.attn, n, context=ends)
                attn_mod.attention(b.q[r0:r0 + n], cache.k, cache.v, b.pos_blk, b.attn, n, scale,
                                   out=b.attn_o[r0:r0 + n], context=ends, ks=cache.ks, vs=cache.vs,
                                   kvq=cache.quantized)
            continue
        if b.attn.qsa:
            attn_mod.qsa_select(b.iq[a0:a1], ikc, pooled, pos, a.ik_scale, w.inv_freq, c.eps, b.attn, a1 - a0,
                                context=keys)
        o = attn_mod.attention(b.q[a0:a1], cache.k, cache.v, pos, b.attn, a1 - a0, scale, context=keys,
                               ks=cache.ks, vs=cache.vs, kvq=cache.quantized)
        if len(segs) > 1:                       # the scratch output is the next stream's too
            b.attn_o[a0:a1].copy_(o[:a1 - a0])
    o = b.attn_o if b.prefill or len(segs) > 1 else o
    glue.attn_gate(o[:R], b.pa[:R], b.gated[:R], b.xs_gated[:R], q_heads=c.heads, head_dim=c.head_dim)
    return _out_proj(w, b, b.gated[:R], a.o, b.xs_gated[:R], R)


def ple_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int) -> None:
    """h += the n-gram embedding branch, each stream through its own conv tail (rows staged by ``stage``)."""

    c = w.cfg
    p = layer.ple
    if w.x3 is not None:                              # an EXL3 pack: the rows' codec, fp16 key/value weights
        from .exl3_mm import ple_rows

        emb = ple_rows(R, w.x3.ple_dev, p.table.head_bias, p.ngram.heads, p.ngram.dims, p.table.bits,
                       w.x3.ple_emb[:R])
        _mm(emb, p.key, None, b.ple_keys[:R], b)
        _mm(emb, p.value, None, b.ple_vals[:R], b)
    else:
        glue.ple_embed(R, b.ple_w, b.ple_s, b.ple_b, p.ngram.heads, p.ngram.dims, b.ple_emb[:R], b.xs_ple[:R])
        _mm(b.ple_emb[:R], p.key, b.xs_ple[:R], b.ple_keys[:R], b)
        _mm(b.ple_emb[:R], p.value, b.xs_ple[:R], b.ple_vals[:R], b)
    glue.ple_gate(b.ple_keys[:R], b.ple_vals[:R], b.h[:R], p.norm_key, p.norm_query, b.ple_gated[:R],
                  b.ple_pss[:R], c.eps, c.streams)
    for st, a0, a1 in segs:
        glue.ple_conv(b.ple_gated[a0:a1], b.ple_pss[a0:a1], p.norm_conv, st.ple_tail, p.conv, b.h[a0:a1],
                      b.h[a0:a1], b.ple_nrow[a0:a1], c.eps, c.streams, c.ngram_size)


def stage_ple_rows(p, b: Buffers, ids: np.ndarray, at: int = 0) -> None:
    """Copy the rows' n-gram table entries (host memory map) to the GPU buffers, from staging row ``at``."""

    words, scales, biases = p.table.gather(ids)
    n = words.shape[0]
    rows = slice(at, at + n)
    b.ple_hw[rows].numpy()[:] = words.view(np.int32)
    b.ple_hs[rows].numpy()[:] = scales.view(np.int16)
    b.ple_hb[rows].numpy()[:] = biases.view(np.int16)
    b.ple_w[rows].copy_(b.ple_hw[rows], non_blocking=True)
    b.ple_s[rows].copy_(b.ple_hs[rows].view(torch.bfloat16), non_blocking=True)
    b.ple_b[rows].copy_(b.ple_hb[rows].view(torch.bfloat16), non_blocking=True)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> tuple:
    """Routed experts + the shared expert. Returns the pending write-back: (2, slots y, weights) on one GPU, (3, gathered fp32 partials, None) across ranks."""

    m = layer.moe
    if w.x3 is not None:                              # an EXL3 pack: each expert at its own width, one GPU
        return _exl3_moe(m, w, b, R)
    buf = moe_mod.moe(b.mixed[:R], m.router, m.experts, b.moe, w.cfg.top_k, w.cfg.experts)
    if w.comm is None:
        return 2, buf.y[:R], buf.wts[:R]
    glue.moe_partial(buf.y[:R], buf.wts[:R], b.part_moe, R)
    return 3, _gather(w, b, b.part_moe, b.g_moe, R), None


def _exl3_moe(m, w: Weights, b: Buffers, R: int) -> tuple:
    """Routed and shared experts on the grouped EXL3 kernel in windows (rows are independent); prompts keep bf16 slots."""

    from tensorfold.cuda.exl3.experts import routed

    from .exl3_pack import MOE_WINDOW

    buf = b.moe
    moe_mod.router(b.mixed[:R], m.router, buf.logits[:R])
    moe_mod.select_rows(buf.logits[:R], buf, w.cfg.top_k, w.cfg.experts)
    if not b.prefill and R <= MOE_WINDOW:
        y = routed(b.mixed[:R], buf.pick[:R], None, m.experts, w.x3.moe, None, R)
        return 2, y.view(R, buf.slots, -1), buf.wts[:R]
    for r0 in range(0, R, MOE_WINDOW):
        n = min(MOE_WINDOW, R - r0)
        y = routed(b.mixed[r0:r0 + n], buf.pick[r0:r0 + n], None, m.experts, w.x3.moe, None, n)
        buf.y[r0:r0 + n].copy_(y.view(n, buf.slots, -1))
    return 2, buf.y[:R], buf.wts[:R]


def _writeback(h: torch.Tensor, b: Buffers, R: int, c, pending) -> None:
    """Apply a pending branch to the streams (in place), no read-out."""

    mode, a, wts, inj = pending
    if mode == 2:
        glue.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, 2, inject=inj[:R], y=a, wts=wts)
    else:
        glue.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, mode, branch=a, inject=inj[:R])


def layer_forward(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, pending, *,
                  mtp: bool = False, context: int | None = None):
    """One decoder layer on b.h[:R]; ``pending`` = the previous MoE's (mode, branch, weights, inject) or None. Returns the new pending write-back."""

    c = w.cfg
    h = b.h
    if layer.ple is not None:
        if pending is not None:
            _writeback(h, b, R, c, pending)
            pending = None
        ple_block(layer, w, segs, b, R)
    if pending is None:
        hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 0, None, b.inj_a, h)
    else:
        mode, a, wts, inj = pending
        if mode == 2:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 2, inj[:R], b.inj_a, h, y=a, wts=wts)
        else:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, mode, inj[:R], b.inj_a, h, branch=a)
    if layer.linear:
        mode, branch = gdn_block(layer, w, segs, b, R)
    else:
        mode, branch = attn_block(layer, w, segs, b, R, mtp, context)
    hc_block(layer.mlp_hc, b, R, c.eps, c.streams, c.low, mode, b.inj_a[:R], b.inj_m, h, branch=branch)
    moe_mode, a, wts = moe_block(layer, w, b, R)
    return (moe_mode, a, wts, b.inj_m)


def finish(w: Weights, mixer: HC, b: Buffers, R: int, pending, logits: bool = True) -> torch.Tensor | None:
    """The last write-back (b.streams: the residual streams before the final mixer), the mixer and the head."""

    c = w.cfg
    b.streams[:R].copy_(b.h[:R])
    _writeback(b.streams, b, R, c, pending)
    if b.prefill:                 # the mixer and the head for the last row only (row 0 of the scratch)
        b.pss[0].copy_(b.pss[R - 1])
        _readout(mixer, b, b.streams[R - 1:R], 1, c.eps, c.streams, c.low, None)
        R = 1
    else:
        _readout(mixer, b, b.streams, R, c.eps, c.streams, c.low, None)
    if not logits:
        return None
    out = _mm(b.mixed[:R], w.head, b.xs_mixed[:R], b.logits[:R], b)
    if w.comm is not None:
        candidates(w, b, out, R, offset=int(w.meta["vocab_offset"]))
    return out


def candidates(w: Weights, b: Buffers, logits: torch.Tensor, R: int, *, id_map: torch.Tensor | None = None,
               offset: int = 0) -> None:
    """Gather each rank's top CAND values, global ids as int32 bits and log-sum-exp into b.cand_all [world, R, 2 CAND + 1] inside the step graph, avoiding a sampling collective."""

    lf = logits.float()
    vals, idx = torch.topk(lf, CAND, dim=-1, sorted=False)
    ids = (id_map[idx] if id_map is not None else idx + offset).to(torch.int32)
    c = b.cand[:R]
    c[:, :CAND] = vals
    c[:, CAND:2 * CAND] = ids.view(torch.float32)
    c[:, 2 * CAND:] = torch.logsumexp(lf, dim=-1, keepdim=True)
    w.comm.all_gather(c, b.cand_all[:b.world * R * (2 * CAND + 1)])


def stage(w: Weights, b: Buffers, windows: Sequence[tuple[State, Sequence[int]]]) -> list[Seg]:
    """Host work before a forward (token ids, n-gram rows into static buffers); returns each stream's segment."""

    segs: list[Seg] = []
    for st, tokens in windows:
        a0 = segs[-1][2] if segs else 0
        if st.pos + len(tokens) > st.capacity:
            raise ValueError("context past the cache capacity")
        segs.append((st, a0, a0 + len(tokens)))
    R = segs[-1][2]
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    b.staged.synchronize()               # the previous step's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.asarray([t for _, tokens in windows for t in tokens], dtype=np.int32)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    for layer in w.layers:
        if layer.ple is not None:
            p = layer.ple
            for (st, tokens), (_, a0, _) in zip(windows, segs):
                toks = np.asarray(tokens, dtype=np.int64)
                ids = p.ngram.ids(st.ple_history, toks)
                st.ple_last = (st.ple_history, toks)
                if w.x3 is not None:
                    from .exl3_pack import stage_ple

                    stage_ple(p.table, w.x3, ids, at=a0 * (ids.size // len(toks)))
                else:
                    stage_ple_rows(p, b, ids, at=a0 * (ids.size // len(toks)))       # ids [rows, heads]
    b.staged.record()
    return segs


def compute(w: Weights, segs: Sequence[Seg], b: Buffers, *, logits: bool = True, context: int | None = None):
    """The forward's GPU work on staged rows (capturable); ``context`` bounds the attention launches."""

    c = w.cfg
    R = segs[-1][2]
    _embed(w, b.ids[:R], c.streams, b.h[:R])
    pending = None
    for layer in w.layers:
        pending = layer_forward(layer, w, segs, b, R, pending, context=context)
    return finish(w, w.mixer, b, R, pending, logits=logits)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True):
    """Rows for ``tokens`` at positions st.pos .. st.pos + R - 1: logits [R, V] bf16 (a view of b.logits) and the residual streams b.streams[:R]. The committed state is unchanged until ``commit``."""

    return compute(w, stage(w, b, [(st, tokens)]), b, logits=logits)


@triton.jit
def _shift_windows(OLD, NEW, keep, OLD_L, NEW_L, NEW_ROW, C: tl.constexpr, T: tl.constexpr, TP: tl.constexpr,
                   BLOCK: tl.constexpr):
    """Program (layer, channel block): window rows j < T become rows keep + j of [old (T rows); new rows]."""

    li = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    ch = cb * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, TP)
    src = keep + j
    from_old = src < T
    ok = j < T
    old = tl.load(OLD + li * OLD_L + tl.where(from_old, src, 0)[:, None] * C + ch[None, :],
                  mask=(ok & from_old)[:, None], other=0.0)
    new = tl.load(NEW + li * NEW_L + tl.where(from_old, 0, src - T)[:, None] * NEW_ROW + ch[None, :],
                  mask=(ok & ~from_old)[:, None], other=0.0)
    rows = tl.where(from_old[:, None], old, new)
    tl.debug_barrier()
    tl.store(OLD + li * OLD_L + j[:, None] * C + ch[None, :], rows, mask=ok[:, None])


def shift_windows(old: torch.Tensor, new: torch.Tensor, keep: int, channels: int) -> None:
    """old [L, T, C] (in place), new [L, R, W >= C] (the first C columns of each row are the window's)."""

    layers, taps, _ = old.shape
    block = 256
    _shift_windows[(layers, triton.cdiv(channels, block))](
        old, new, keep, old.stride(0), new.stride(0), new.stride(1), C=channels, T=taps,
        TP=triton.next_power_of_2(taps), BLOCK=block, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int, at: int = 0) -> None:
    """Keep the first ``keep`` of the R rows the last forward (buffers ``b``) ran for ``st``, from window row ``at``."""

    c = w.cfg
    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)          # a prompt chunk's DeltaNet layers committed during the forward
    if n:
        for li in range(n):
            cur = st.cur[li]
            if keep < R:
                gdn_mod.replay(st.rec[cur, li], st.scratch[li], keep, st.rec[1 - cur, li])
            st.cur[li] = 1 - cur
        shift_windows(st.conv, b.proj[:, at:at + R], keep, c.conv_dim)
    if st.ple_last is not None:
        history, tokens = st.ple_last
        st.ple_history = np.concatenate([history, tokens[:keep]])[-(c.ngram_size - 1):]
        st.ple_last = None
        tail = st.ple_tail
        shift_windows(tail[None], b.ple_nrow[None, at:at + R], keep, tail.shape[1])
    st.set_pos(st.pos + keep)
