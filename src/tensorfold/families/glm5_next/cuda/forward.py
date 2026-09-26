"""GLM-5.3-Flash forward on CUDA for a chain of R consecutive tokens (1 = a serial step, 2-8 = an MTP verify
window, up to ``Buffers.rows`` for prefill chunks), tensor parallel over ``w.world`` ranks, and the commit
that keeps a prefix of it.

Every kernel treats each row on its own (``qmm``, ``glue``, ``kda``, ``attention``), so row r of a window gets
the bits of the serial step at its position. Each rank computes its heads and its share of every MLP and
expert; block outputs leave a rank as fp32 partials, and ``glue.hc_post`` adds the gathered partials rank 0
first and rounds once. The committed state is read-only during a forward except for attention cache slots
past the committed length (the window's keys and values), which a later round overwrites. ``commit`` keeps
the first ``keep`` rows:

- KDA: the chain kernel writes the state after its last row into the layer's other state buffer; a shorter
  keep replays the kept rows from the committed state into that buffer (same update routine, same bits).
  Either way the layer's current buffer flips. Conv windows: rows [keep, keep + 3) of [old window; rows].
- DSA: the committed length advances by ``keep``.

Sparse attention is exact here while every query sees at most ``Config.dense_limit`` keys: below that DSA's
indexer keeps every visible key, so attention is dense and the indexer need not run. Longer contexts need the
engine built with ``long_context`` (``sparse.py``).
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from . import glue, kda as kda_mod, qmm, sparse
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights


class Buffers:
    """Scratch for windows of up to ``rows`` rows (views [:R] serve smaller windows)."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.rows = rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        self.attn = AttnScratch(rows, HL, c.qk_dim, capacity, dev)
        self.hin = torch.empty((rows, c.hidden), dtype=torch.bfloat16, device=dev)      # MTP input rows
        self.zero_first = False          # MTP: this step starts at position 0 (its embedding is zeroed)
        self.x = torch.empty((rows, S * D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.xs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.post = torch.empty((rows, S), dtype=f32, device=dev)
        self.comb = torch.empty((rows, S * S), dtype=f32, device=dev)
        self.hcpart = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=f32, device=dev)
        # KDA
        self.ka = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.xs_fa = torch.empty((rows, 2), dtype=f32, device=dev)
        self.xs_ga = torch.empty((rows, 2), dtype=f32, device=dev)
        self.kxs = torch.empty((rows, LL * 128 // 64), dtype=f32, device=dev)
        # DSA
        self.dp = torch.empty((rows, c.q_lora + c.kv_lora), dtype=bf, device=dev)
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.xs_qr = torch.empty((rows, c.q_lora // 64), dtype=f32, device=dev)
        self.lat = torch.empty((rows, c.kv_lora), dtype=bf, device=dev)
        self.xs_lat = torch.empty((rows, c.kv_lora // 64), dtype=f32, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.kn = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.xs_ao = torch.empty((rows, HL * c.v_dim // 64), dtype=f32, device=dev)
        # DSA indexer (long contexts)
        self.ikr = torch.empty((rows, c.index_dim + c.index_heads), dtype=bf, device=dev)
        self.igr = torch.empty((rows, c.index_dim), dtype=f32, device=dev)
        self.qi = torch.empty((rows, c.index_heads * c.index_dim), dtype=bf, device=dev)
        # dense MLP
        dl = c.dense_width // w.world
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, dl // 64), dtype=f32, device=dev)
        # MoE
        slots = c.top_k + 1
        ml = c.moe_width // w.world
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        self.eact = torch.empty((rows, slots, ml), dtype=bf, device=dev)
        self.eaxs = torch.empty((rows, slots, ml // 64), dtype=f32, device=dev)
        self.ey = torch.empty((rows, slots, D), dtype=f32, device=dev)
        self._groups: dict[int, qmm.Group] = {}
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sk = torch.empty((8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mxs = torch.empty((rows, 2 * D // 64), dtype=f32, device=dev)
        self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        self._parents: dict[int, torch.Tensor] = {}
        # DFlash2 taps: the mean of the streams after chosen layers (``set_taps``), filled by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.experts = c.experts
        self.top_k = c.top_k

    def set_taps(self, layers: tuple[int, ...], hidden: int) -> None:
        self.tap_at = {}
        for i, layer in enumerate(layers):
            self.tap_at.setdefault(layer, []).append(i)
        self.taps = [torch.empty((self.rows, hidden), dtype=torch.bfloat16, device=self.ids.device) for _ in layers]

    def group(self, R: int) -> qmm.Group:
        g = self._groups.get(R)
        if g is None:
            dev = self.ids.device
            maxu = min(R * self.top_k, self.experts) + 1
            g = qmm.Group(torch.zeros((maxu,), dtype=torch.int32, device=dev),
                          torch.zeros((1,), dtype=torch.int32, device=dev),
                          torch.full((maxu, R), -1, dtype=torch.int32, device=dev))
            self._groups[R] = g
        return g

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer)."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.conv = torch.zeros((n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch_set = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.scratch = self.scratch_set.views if n else []
        self.kc = [torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.vc = [torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            self.mtp_kc = torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev)
            self.mtp_vc = torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev)
        # DSA indexer caches (long contexts only): per layer (and the MTP layer, last) keys, gates, pool keys
        self.index = None
        if w.meta.get("long_context"):
            n_idx = len(dsa_layers) + (1 if w.mtp is not None else 0)
            mk = lambda n: torch.zeros((n, c.index_dim), dtype=torch.bfloat16, device=dev)   # noqa: E731
            self.index = [(mk(capacity), mk(capacity), mk(capacity // 4 + 2)) for _ in range(n_idx)]

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() for x in self.vc]
        if self.index is not None:
            other.index = [tuple(x.clone() for x in trio) for trio in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone()
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Every rank's fp32 partial b.part[:R] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = b.part.shape[1]
    if w.comm is None:
        return b.part[:R].view(1, R, d)
    out = b.gath[:b.world * R * d]
    w.comm.all_gather(b.part[:R].reshape(-1), out)
    return out.view(b.world, R, d)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int) -> torch.Tensor:
    qmm.matmul(x, q, xs, out=b.part[:R], f32=True, part=b.sk)
    return gather(w, b, R)


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    p = st.proj[li, :R]
    qmm.matmul(b.normed[:R], k.proj, b.xs[:R], out=p, part=b.sk)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    qmm.matmul(fa, k.fb, qmm.group_sums(fa, b.xs_fa[:R]), out=b.ka[:R], part=b.sk)
    qmm.matmul(ga, k.gb, qmm.group_sums(ga, b.xs_ga[:R]), out=b.kg[:R], part=b.sk)
    cur = st.cur[li]
    out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log, k.dt_bias,
                        k.norm, c.eps, c.lower, R, st.scratch[li], st.rec[1 - cur, li])
    return out_proj(w, b, out, k.o, qmm.group_sums(out, b.kxs[:R]), R)


def dsa_block(layer: LayerW, w: Weights, kc: torch.Tensor, vc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers,
              R: int, nch: int | None, index=None, host_pos: int | None = None) -> torch.Tensor:
    """``pos_dev``: the committed length on the device; ``nch``: attention chunks to visit (None: the capacity's).
    ``index``: this layer's indexer caches (long contexts): the window's index keys and pools are always written;
    rows past 2050 then attend to their top-512 pools (``host_pos`` given, eager only)."""

    c = w.cfg
    a = layer.dsa
    qmm.matmul(b.normed[:R], a.proj, b.xs[:R], out=b.dp[:R], part=b.sk)
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    qmm.matmul(b.qr[:R], a.q_b, b.xs_qr[:R], out=b.q[:R].view(R, HL * c.qk_dim), part=b.sk)
    qmm.matmul(b.lat[:R], a.kv_k, b.xs_lat[:R], out=b.kn[:R].view(R, HL * c.qk_dim), part=b.sk)
    qmm.matmul(b.lat[:R], a.kv_v, b.xs_lat[:R], out=b.vn[:R].view(R, HL * c.v_dim), part=b.sk)
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        qmm.matmul(b.normed[:R], ix.kw, b.xs[:R], out=b.ikr[:R], part=b.sk)
        glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:
        o = b.attn.out[:R]                     # every row sparse: the dense pass is skipped
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        qmm.matmul(b.qr[:R], ix.qb, b.xs_qr[:R], out=b.qi[:R], part=b.sk)
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R)


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    qmm.matmul(b.normed[:R], m.gu, b.xs[:R], out=b.gu[:R], part=b.sk)
    glue.swiglu(b.gu[:R], b.act[:R], b.xs_act[:R], w.cfg.limit)
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    m = layer.moe
    grp = b.group(R)
    glue.router(b.normed[:R], m.router, b.mlog[:R])
    glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], grp.ids, grp.count, grp.members, c.top_k, c.experts,
                c.routed_scale, c.norm_topk)
    qmm.moe_gateup(b.normed[:R], b.xs[:R], m.experts, grp, b.eact, b.eaxs, c.limit)
    qmm.moe_down(b.eact, b.eaxs, m.experts, grp, b.ey)
    glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None) -> None:
    c = w.cfg
    x = b.x[:R]
    h = layer.attn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    if layer.kind == "kda":
        g = kda_block(layer, w, st, b, R)
    else:
        di = st.dsa_index[layer.index]
        g = dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch,
                      st.index[di] if st.index is not None else None, host_pos)
    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])
    h = layer.ffn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    check_room(w, st, R)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = list(tokens)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return R


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None):
    """The GPU work of a forward on staged rows (capturable: static buffers, device-side positions). ``host_pos``
    (eager calls): the committed length, which long contexts need to switch rows to sparse attention."""

    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    glue.stream_mean(b.x[:R], b.hidden[:R])
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    return qmm.matmul(b.fnormed[:R], w.head, b.fxs[:R], out=b.logits[:R], part=b.sk)


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Rows for ``tokens`` at positions st.pos .. st.pos + R - 1: logits [R, V/world] bf16 (a view of b.logits)
    and the hidden rows b.hidden[:R] (the mean of the streams before the final norm: the MTP head's input).
    The committed state is unchanged until ``commit``."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@triton.jit
def _row(CONV, PROJ, l, src, c, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr):
    old = tl.load(CONV + l * conv_layer + src * C + c, mask=(src < TAPS) & (c < C), other=0.0)
    new = tl.load(PROJ + l * proj_layer + (src - TAPS) * proj_row + c, mask=(src >= TAPS) & (c < C), other=0.0)
    return tl.where(src < TAPS, old, new)


@triton.jit
def _conv_shift(CONV, PROJ, keep, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program (layer, channel block): the 3 window rows become rows keep .. keep + 2 of [old window; new rows]."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v0 = _row(CONV, PROJ, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v1 = _row(CONV, PROJ, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v2 = _row(CONV, PROJ, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
    tl.store(CONV + l * conv_layer + c, v0, mask=c < C)
    tl.store(CONV + l * conv_layer + C + c, v1, mask=c < C)
    tl.store(CONV + l * conv_layer + 2 * C + c, v2, mask=c < C)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the first ``keep`` rows of the last forward's R rows (every KDA layer at once)."""

    if not 1 <= keep <= R:
        raise ValueError("keep must be in 1..R")
    n = len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        C = st.conv.shape[2]
        taps = st.conv.shape[1]
        if taps != 3:
            raise ValueError("the conv shift kernel is written for 4-tap convolutions")
        _conv_shift[(n, triton.cdiv(C, 1024))](st.conv, st.proj, keep, st.conv.stride(0), st.proj.stride(0),
                                               st.proj.stride(1), C=C, TAPS=taps, BLOCK=1024, num_warps=4)
    st.set_pos(st.pos + keep)
