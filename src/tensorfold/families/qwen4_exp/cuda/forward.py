"""Flash Next forward on CUDA for a chain of R consecutive tokens (1 = a serial step, 2-8 = an MTP verify
window, up to ``Buffers.rows`` for prefill chunks), and the commit that keeps a prefix of it.

Every kernel treats each row on its own (``qmm``, ``moe``, ``glue``, ``gdn``, ``attention``), so row r of
a window gets the bits of the serial step at its position. The committed state is read-only during a
forward, except for attention cache slots past the committed length (the window's keys), which a
later round overwrites. ``commit`` then keeps the first ``keep`` rows:

- DeltaNet: the forward writes the state after its last row into the layer's other state buffer; a
  shorter keep replays the kept rows from the committed state into that buffer (same update routine,
  same bits). Either way the layer's current buffer flips.
- conv windows (DeltaNet and the n-gram conv): rows [keep, keep + taps) of [old window; the rows].
- attention: the committed length advances by ``keep``.

Buffers are static per window size and the committed length is read on the device, so a forward can
be captured in a CUDA graph (``graphs.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from . import attention as attn_mod
from . import gdn as gdn_mod
from . import glue, moe as moe_mod, qmm
from .weights import HC, LayerW, Weights


# -- per-window buffers --------------------------------------------------------------------------------
CAND = 32      # tensor parallel: candidates a rank gathers per row for sampling (top-k 20 plus the sampler's margin 8)


class Buffers:
    """Scratch for windows of up to ``rows`` rows (views [:R] serve smaller windows)."""

    def __init__(self, w: Weights, rows: int, capacity: int) -> None:
        c = w.cfg
        dev = w.device
        wide = c.streams * c.hidden
        self.rows = rows
        bf, f32 = torch.bfloat16, torch.float32
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        self.h = torch.empty((rows, wide), dtype=bf, device=dev)
        self.pss = torch.empty((rows, c.hidden // 256, c.streams), dtype=f32, device=dev)
        self.normed = torch.empty((rows, wide), dtype=bf, device=dev)
        self.xs_normed = torch.empty((rows, wide // 32), dtype=f32, device=dev)
        self.dn = torch.empty((rows, c.low + c.streams), dtype=bf, device=dev)
        self.dn_mix = torch.empty((rows, c.low), dtype=bf, device=dev)      # a mixer's down (no inject rows)
        self.act = torch.empty((rows, c.low), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, c.low // 32), dtype=f32, device=dev)
        self.inj_a = torch.empty((rows, c.streams), dtype=bf, device=dev)
        self.inj_m = torch.empty((rows, c.streams), dtype=bf, device=dev)
        self.up = torch.empty((rows, wide), dtype=bf, device=dev)
        self.mixed = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.xs_mixed = torch.empty((rows, c.hidden // 32), dtype=f32, device=dev)
        self.branch = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        attn_width = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
        self.pa = torch.empty((rows, attn_width), dtype=bf, device=dev)
        self.q = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.iq = torch.empty((rows, c.index_heads, c.index_dim), dtype=bf, device=dev)
        self.attn = attn_mod.AttnScratch(rows, c.heads, c.head_dim, capacity, dev, budget=c.index_budget,
                                         ratio=c.index_ratio)
        self.gated = torch.empty((rows, c.heads * c.head_dim), dtype=bf, device=dev)
        self.xs_gated = torch.empty((rows, c.heads * c.head_dim // 32), dtype=f32, device=dev)
        self.moe = moe_mod.MoEBuffers(rows, _MoECfg(c), dev)
        self.streams = torch.empty((rows, wide), dtype=bf, device=dev)
        self.logits = torch.empty((rows, w.head.n), dtype=bf, device=dev)
        world = int(w.meta.get("world", 1))
        self.world = world
        if world > 1:                  # tensor parallel: fp32 partials and their rank-ordered gathers
            self.part_branch = torch.empty((rows, c.hidden), dtype=f32, device=dev)
            self.part_moe = torch.empty((rows, c.hidden), dtype=f32, device=dev)
            self.g_branch = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.g_moe = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.cand = torch.empty((rows, 2 * CAND + 1), dtype=f32, device=dev)
            self.cand_all = torch.empty((world * rows * (2 * CAND + 1),), dtype=f32, device=dev)
        # n-gram embedding
        nrow = rows * 2 * c.heads_per_ngram
        dh = c.ple_dim // (2 * c.heads_per_ngram)
        self.ple_w = torch.zeros((nrow, dh // 8), dtype=torch.int32, device=dev)
        self.ple_s = torch.zeros((nrow, dh // 32), dtype=bf, device=dev)
        self.ple_b = torch.zeros((nrow, dh // 32), dtype=bf, device=dev)
        pin = torch.cuda.is_available()
        self.ple_hw = torch.zeros((nrow, dh // 8), dtype=torch.int32, pin_memory=pin)
        self.ple_hs = torch.zeros((nrow, dh // 32), dtype=torch.int16, pin_memory=pin)
        self.ple_hb = torch.zeros((nrow, dh // 32), dtype=torch.int16, pin_memory=pin)
        self.ple_emb = torch.empty((rows, c.ple_dim), dtype=bf, device=dev)
        self.xs_ple = torch.empty((rows, c.ple_dim // 32), dtype=f32, device=dev)
        self.ple_keys = torch.empty((rows, wide), dtype=bf, device=dev)
        self.ple_vals = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.ple_gated = torch.empty((rows, wide), dtype=bf, device=dev)
        self.ple_pss = torch.empty((rows, c.streams), dtype=f32, device=dev)
        self.ple_nrow = torch.empty((rows, wide), dtype=bf, device=dev)
        # split-K partials for the largest matmul of a window
        self.part = torch.empty((32 * max(rows, 4) * 2560,), dtype=f32, device=dev)
        # MTP
        self.mtp_e = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_xe = torch.empty((rows, c.hidden // 32), dtype=f32, device=dev)
        self.mtp_eo = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_hn = torch.empty((rows, wide), dtype=bf, device=dev)
        self.mtp_xh = torch.empty((rows, wide // 32), dtype=f32, device=dev)
        self.mtp_hs = torch.empty((rows * c.streams, c.hidden), dtype=bf, device=dev)
        self.mtp_in = torch.empty((rows, wide), dtype=bf, device=dev)          # the MTP's input streams


class _MoECfg:
    def __init__(self, c) -> None:
        self.num_experts_per_tok = c.top_k
        self.num_experts = c.experts
        self.moe_intermediate_size = c.moe_width
        self.hidden_size = c.hidden


# -- committed state -----------------------------------------------------------------------------------
class State:
    """Committed caches of one sequence (and of the MTP head's attention layer)."""

    def __init__(self, w: Weights, capacity: int, max_rows: int) -> None:
        c = w.cfg
        dev = w.device
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        lin = [l for l in w.layers if l.linear]
        att = [l for l in w.layers if not l.linear]
        self.lin_index = {l.index: i for i, l in enumerate(lin)}
        self.att_index = {l.index: i for i, l in enumerate(att)}
        n = len(lin)
        self.conv = torch.zeros((n, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, c.nv, c.dv, c.dk), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, max_rows, gdn_mod.widths(c.nk, c.nv)[1]), dtype=torch.bfloat16, device=dev)
        self.scratch = [gdn_mod.GDNScratch(max_rows, dev, c.nk, c.nv) for _ in range(n)]
        self.kc = [torch.zeros((capacity, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev) for _ in att]
        self.vc = [torch.zeros_like(x) for x in self.kc]
        self.ikc = [torch.zeros((capacity, c.index_dim), dtype=torch.bfloat16, device=dev) for _ in att]
        nb = -(-capacity // c.index_ratio)
        self.pooled = [torch.zeros((nb, c.index_dim), dtype=torch.bfloat16, device=dev) for _ in att]
        wide = c.streams * c.hidden
        self.ple_tail = torch.zeros(((c.ple_kernel - 1) * c.ngram_size, wide), dtype=torch.bfloat16, device=dev)
        self.ple_history = c.ngram(0).initial_history() if c.ple_layers else None
        self.ple_last: tuple[np.ndarray, np.ndarray] | None = None
        # MTP head (its own attention cache; ``mtp_len`` entries, the last ``mtp_drafted`` of them chained drafts)
        self.mtp_len = 0
        self.mtp_drafted = 0
        self.mtp_pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        if w.mtp is not None:
            self.mtp_kc = torch.zeros((capacity, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)
            self.mtp_vc = torch.zeros_like(self.mtp_kc)
            self.mtp_ikc = torch.zeros((capacity, c.index_dim), dtype=torch.bfloat16, device=dev)
            self.mtp_pooled = torch.zeros((-(-capacity // c.index_ratio), c.index_dim), dtype=torch.bfloat16,
                                          device=dev)

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def reset(self, w: Weights) -> None:
        """An empty sequence in the same buffers (captured graphs keep pointing at them)."""

        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.ple_tail.zero_()
        self.ple_history = w.cfg.ngram(0).initial_history() if w.cfg.ple_layers else None
        self.ple_last = None
        self.set_pos(0)
        self.mtp_drafted = 0
        self.set_mtp_len(0)

    def clone(self) -> "State":
        """An independent copy (tests and A/B checks)."""

        import copy

        other = copy.copy(self)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(other, name, value.clone())
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                setattr(other, name, [v.clone() for v in value])
        other.cur = list(self.cur)
        other.scratch = [copy.copy(sc) for sc in self.scratch]
        for sc_new, sc in zip(other.scratch, self.scratch):
            for name, value in vars(sc).items():
                setattr(sc_new, name, value.clone())
        return other

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos.fill_(n)

    def snapshot(self) -> dict:
        """What the committed sequence keeps outside the caches' rows: the DeltaNet states, the conv and n-gram
        windows, the lengths. With the cache rows below ``pos`` still in place, ``restore`` brings the sequence
        back; a copy of about 113 MB (the DeltaNet states of 36 layers)."""

        p = self.cur[0] if self.cur else 0
        if any(c != p for c in self.cur):
            raise RuntimeError("DeltaNet layers out of step")
        return {"pos": self.pos, "rec": self.rec[p].clone(), "conv": self.conv.clone(),
                "ple_tail": self.ple_tail.clone(),
                "ple_history": None if self.ple_history is None else self.ple_history.copy(),
                "mtp_len": self.mtp_len - self.mtp_drafted}

    def restore(self, snap: dict) -> None:
        self.rec[0].copy_(snap["rec"])
        self.cur = [0] * len(self.cur)
        self.conv.copy_(snap["conv"])
        self.ple_tail.copy_(snap["ple_tail"])
        self.ple_history = None if snap["ple_history"] is None else snap["ple_history"].copy()
        self.ple_last = None
        self.set_pos(snap["pos"])
        self.mtp_drafted = 0
        self.set_mtp_len(snap["mtp_len"])


# -- blocks ----------------------------------------------------------------------------------------------
def _gather(w: Weights, b: Buffers, part: torch.Tensor, flat: torch.Tensor, R: int) -> torch.Tensor:
    """All ranks' fp32 partials [R, D] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = part.shape[1]
    out = flat[:b.world * R * d]
    w.comm.all_gather(part[:R], out)
    return out.view(b.world, R, d)


def _mm(x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, out: torch.Tensor, b: Buffers) -> torch.Tensor:
    return qmm.matmul(x, q, xs, out=out, part=b.part)


def hc_block(hc: HC, b: Buffers, R: int, eps: float, streams: int, low: int, mode: int, inject_prev,
             inject_out, h: torch.Tensor, branch=None, y=None, wts=None) -> None:
    """Write the pending branch back into the streams h (in place), then the hyper-connection's read-out:
    b.mixed [R, D] (+ group sums), and its inject gates into ``inject_out``."""

    glue.hc_writeback(h[:R], h[:R], b.pss[:R], streams, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
    _readout(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None)


FUSED_ROWS = 16      # decode windows: the read-out in 3 kernels; wider windows (prefill) in 5, the same bits


def _readout(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """normed streams -> down -> SiLU / inject -> up -> mix: b.mixed [R, D] and its group sums."""

    if R <= FUSED_ROWS:
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
    _mm(b.act[:R], hc.up, b.xs_act[:R], b.up[:R], b)
    glue.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _down_act(hc: HC, b: Buffers, R: int, streams: int, low: int, inject) -> None:
    """A hyper-connection's down projection, then SiLU and the inject gates: b.act, b.xs_act (and ``inject``).
    With a split K the slice sum is fused into the activation kernel (the same bits as reduce, then act)."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    got = qmm.matmul(b.normed[:R], hc.down, b.xs_normed[:R], out=out, part=b.part, reduce=False)
    if got.dim() == 3:
        glue.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        glue.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)


def gdn_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int) -> None:
    c = w.cfg
    g = layer.gdn
    li = st.lin_index[layer.index]
    p = st.proj[li, :R]
    _mm(b.mixed[:R], g.proj, b.xs_mixed[:R], p, b)
    cur = st.cur[li]
    sc = st.scratch[li]
    gdn_mod.chain(p, st.conv[li], g.conv, st.rec[cur, li], g.a_log, g.dt_bias, g.norm, c.eps, R, sc,
                  st.rec[1 - cur, li])
    return _out_proj(w, b, sc.out[:R], g.out, sc.xs[:R], R)


def _out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int):
    """A block's output projection: (1, bf16 branch) on one GPU; (3, gathered fp32 partials) across ranks."""

    if w.comm is None:
        got = qmm.matmul(x, q, xs, out=b.branch[:R], part=b.part, reduce=False)
        if got.dim() == 3:
            return 4, got            # K slices: the write-back sums them in order (the bits of reduce, then round)
        return 1, got
    qmm.matmul(x, q, xs, out=b.part_branch[:R], part=b.part, f32=True)
    return 3, _gather(w, b, b.part_branch, b.g_branch, R)


def attn_block(layer: LayerW, w: Weights, kc, vc, ikc, pooled, pos_dev: torch.Tensor, b: Buffers, R: int,
               context: int):
    c = w.cfg
    a = layer.attn
    _mm(b.mixed[:R], a.proj, b.xs_mixed[:R], b.pa[:R], b)
    glue.attn_prep(b.pa[:R], pos_dev, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, b.q, kc, vc, b.iq, ikc,
                   c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, index_heads=c.index_heads,
                   index_dim=c.index_dim)
    if b.attn.qsa:
        attn_mod.qsa_select(b.iq[:R], ikc, pooled, pos_dev, a.ik_scale, w.inv_freq, c.eps, b.attn, R)
    o = attn_mod.attention(b.q[:R], kc, vc, pos_dev, b.attn, R, c.head_dim ** -0.5)
    glue.attn_gate(o[:R], b.pa[:R], b.gated[:R], b.xs_gated[:R], q_heads=c.heads, head_dim=c.head_dim)
    return _out_proj(w, b, b.gated[:R], a.o, b.xs_gated[:R], R)


def ple_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int) -> None:
    """h += the n-gram embedding branch (model.PLELayer), rows in order through the dilated conv. The rows'
    table entries were staged by ``stage`` (host work, outside a captured graph)."""

    c = w.cfg
    p = layer.ple
    glue.ple_embed(R, b.ple_w, b.ple_s, b.ple_b, p.ngram.heads, p.ngram.dims, b.ple_emb[:R], b.xs_ple[:R])
    _mm(b.ple_emb[:R], p.key, b.xs_ple[:R], b.ple_keys[:R], b)
    _mm(b.ple_emb[:R], p.value, b.xs_ple[:R], b.ple_vals[:R], b)
    glue.ple_gate(b.ple_keys[:R], b.ple_vals[:R], b.h[:R], p.norm_key, p.norm_query, b.ple_gated[:R],
                  b.ple_pss[:R], c.eps, c.streams)
    glue.ple_conv(b.ple_gated[:R], b.ple_pss[:R], p.norm_conv, st.ple_tail, p.conv, b.h[:R], b.h[:R],
                  b.ple_nrow[:R], c.eps, c.streams, c.ngram_size)


def stage_ple_rows(p, b: Buffers, ids: np.ndarray) -> None:
    """Copy the rows' n-gram table entries (host memory map) to the GPU buffers."""

    words, scales, biases = p.table.gather(ids)
    n = words.shape[0]
    b.ple_hw[:n].numpy()[:] = words.view(np.int32)
    b.ple_hs[:n].numpy()[:] = scales.view(np.int16)
    b.ple_hb[:n].numpy()[:] = biases.view(np.int16)
    b.ple_w[:n].copy_(b.ple_hw[:n], non_blocking=True)
    b.ple_s[:n].copy_(b.ple_hs[:n].view(torch.bfloat16), non_blocking=True)
    b.ple_b[:n].copy_(b.ple_hb[:n].view(torch.bfloat16), non_blocking=True)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> tuple:
    """Routed experts + the shared expert. Returns the pending write-back: (2, slots y, weights) on one GPU,
    (3, gathered fp32 partials, None) across ranks."""

    m = layer.moe
    buf = b.moe
    sub = _Sub(buf, R)
    moe_mod.router(b.mixed[:R], m.router, buf.logits[:R])
    moe_mod.select(buf.logits[:R], sub, w.cfg.top_k, w.cfg.experts)
    qmm.moe_gateup(b.mixed[:R], b.xs_mixed[:R], m.experts, sub.group, buf.act, buf.axs)
    qmm.moe_down(buf.act, buf.axs, m.experts, sub.group, buf.y)
    if w.comm is None:
        return 2, buf.y, sub.wts
    glue.moe_partial(buf.y, sub.wts, b.part_moe, R)
    return 3, _gather(w, b, b.part_moe, b.g_moe, R), None


class _Sub:
    """A MoE buffer set restricted to R rows (grouping sized for R, not for the buffer's maximum)."""

    def __new__(cls, buf, R: int):
        if R == buf.rows:
            return buf
        subs = buf.__dict__.setdefault("_subs", {})          # kept on the buffer: never outlives it
        sub = subs.get(R)
        if sub is None:
            sub = object.__new__(moe_mod.MoEBuffers)
            sub.rows, sub.slots = R, buf.slots
            sub.maxu = min(R * (buf.slots - 1), buf.group.ids.shape[0] - 1) + 1
            sub.logits, sub.pick, sub.wts = buf.logits[:R], buf.pick[:R], buf.wts[:R]
            sub.group = qmm.Group(buf.group.ids[:sub.maxu], buf.group.count,
                                  buf.group.members.as_strided((sub.maxu, R), (R, 1)))
            sub.act, sub.axs, sub.y = buf.act, buf.axs, buf.y
            subs[R] = sub
        return sub


def _writeback(h: torch.Tensor, b: Buffers, R: int, c, pending) -> None:
    """Apply a pending branch to the streams (in place), no read-out."""

    mode, a, wts, inj = pending
    if mode == 2:
        glue.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, 2, inject=inj[:R], y=a, wts=wts)
    else:
        glue.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, mode, branch=a, inject=inj[:R])


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, pending, *,
                  mtp: bool = False, context: int = 0):
    """One decoder layer on b.h[:R]; ``pending`` = the previous MoE's (mode, branch, weights, inject) or None.
    Returns the new pending write-back."""

    c = w.cfg
    h = b.h
    if layer.ple is not None:
        if pending is not None:
            _writeback(h, b, R, c, pending)
            pending = None
        ple_block(layer, w, st, b, R)
    if pending is None:
        hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 0, None, b.inj_a, h)
    else:
        mode, a, wts, inj = pending
        if mode == 2:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 2, inj[:R], b.inj_a, h, y=a, wts=wts)
        else:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, mode, inj[:R], b.inj_a, h, branch=a)
    if layer.linear:
        mode, branch = gdn_block(layer, w, st, b, R)
    elif mtp:
        mode, branch = attn_block(layer, w, st.mtp_kc, st.mtp_vc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, b, R,
                                  context)
    else:
        ai = st.att_index[layer.index]
        mode, branch = attn_block(layer, w, st.kc[ai], st.vc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, b, R,
                                  context)
    hc_block(layer.mlp_hc, b, R, c.eps, c.streams, c.low, mode, b.inj_a[:R], b.inj_m, h, branch=branch)
    moe_mode, a, wts = moe_block(layer, w, b, R)
    return (moe_mode, a, wts, b.inj_m)


def finish(w: Weights, mixer: HC, b: Buffers, R: int, pending, logits: bool = True) -> torch.Tensor | None:
    """The last write-back (b.streams: the residual streams before the final mixer), the mixer and the head."""

    c = w.cfg
    b.streams[:R].copy_(b.h[:R])
    _writeback(b.streams, b, R, c, pending)
    _readout(mixer, b, b.streams, R, c.eps, c.streams, c.low, None)
    if not logits:
        return None
    out = _mm(b.mixed[:R], w.head, b.xs_mixed[:R], b.logits[:R], b)
    if w.comm is not None:
        candidates(w, b, out, R, offset=int(w.meta["vocab_offset"]))
    return out


def candidates(w: Weights, b: Buffers, logits: torch.Tensor, R: int, *, id_map: torch.Tensor | None = None,
               offset: int = 0) -> None:
    """Tensor parallel: each row's top CAND logits over this rank's vocabulary slice (values, then global ids as
    int32 bits) and the slice's log-sum-exp, all-gathered into b.cand_all [world, R, 2 CAND + 1]. It runs inside
    the step's CUDA graph, so sampling costs one device-to-host copy and no collective of its own."""

    lf = logits.float()
    vals, idx = torch.topk(lf, CAND, dim=-1, sorted=False)
    ids = (id_map[idx] if id_map is not None else idx + offset).to(torch.int32)
    c = b.cand[:R]
    c[:, :CAND] = vals
    c[:, CAND:2 * CAND] = ids.view(torch.float32)
    c[:, 2 * CAND:] = torch.logsumexp(lf, dim=-1, keepdim=True)
    w.comm.all_gather(c, b.cand_all[:b.world * R * (2 * CAND + 1)])


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: token ids and the n-gram rows into the static device buffers (pinned, async)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    if st.pos + R > st.capacity:
        raise ValueError("context past the cache capacity")
    b.staged.synchronize()               # the previous step's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.asarray(tokens, dtype=np.int32)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    toks = np.asarray(tokens, dtype=np.int64)
    for layer in w.layers:
        if layer.ple is not None:
            p = layer.ple
            ids = p.ngram.ids(st.ple_history, toks)
            st.ple_last = (st.ple_history, toks)
            stage_ple_rows(p, b, ids)
    b.staged.record()
    return R


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, context: int | None = None):
    """The GPU work of a forward on staged rows (capturable: static buffers, device-side positions)."""

    c = w.cfg
    glue.embed(b.ids[:R], *w.embed, c.hidden, copies=c.streams, out=b.h[:R])
    pending = None
    ctx = st.pos + R if context is None else context
    for layer in w.layers:
        pending = layer_forward(layer, w, st, b, R, pending, context=ctx)
    return finish(w, w.mixer, b, R, pending, logits=logits)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True):
    """Rows for ``tokens`` at positions st.pos .. st.pos + R - 1: logits [R, V] bf16 (a view of b.logits) and
    the residual streams b.streams[:R]. The committed state is unchanged until ``commit``."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits)


# -- commit ------------------------------------------------------------------------------------------------
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
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the first ``keep`` rows of the last forward (on ``st`` with buffers ``b``) of R rows."""

    c = w.cfg
    if not 1 <= keep <= R:
        raise ValueError("keep must be in 1..R")
    n = len(st.cur)
    if n:
        for li in range(n):
            cur = st.cur[li]
            if keep < R:
                gdn_mod.replay(st.rec[cur, li], st.scratch[li], keep, st.rec[1 - cur, li])
            st.cur[li] = 1 - cur
        shift_windows(st.conv, st.proj, keep, c.conv_dim)
    if st.ple_last is not None:
        history, tokens = st.ple_last
        st.ple_history = np.concatenate([history, tokens[:keep]])[-(c.ngram_size - 1):]
        st.ple_last = None
        tail = st.ple_tail
        shift_windows(tail[None], b.ple_nrow[None], keep, tail.shape[1])
    st.set_pos(st.pos + keep)
