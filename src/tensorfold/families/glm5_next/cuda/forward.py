"""GLM-5.3-Flash's tensor-parallel forward and commit; kernels keep rows apart, so row r has the serial step's bits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
from tensorfold.cuda.geometry import MLA_PROMPT_ATT_ROWS as PROMPT_ATT_ROWS   # a dense latent call's prompt rows
from tensorfold.cuda.kernels import prefill_attention, qmm as shared

from . import exl3_generic, glue, kda as kda_mod, latent, prof, qmm, sparse
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights


@dataclass
class Cut:
    """The recurrent state and convolution windows at an interior prompt row."""

    point: int
    rec: torch.Tensor
    conv: torch.Tensor


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.rows, self.prefill = rows, prefill
        head_rows = 1 if prefill else rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        if latent.ENABLED:
            # Dense attention only ever covers contexts up to the dense limit; longer rows go sparse.
            self.attn = None
            # a prompt chunk's dense pass runs PROMPT_ATT_ROWS rows at a time: its fp32 partials hold that many rows
            self.lat_s = latent.LatentScratch(rows, HL, latent.chunks_for(min(capacity, 2560) + rows), dev,
                                              lw=c.kv_lora, part_rows=PROMPT_ATT_ROWS if prefill else rows)
        else:
            self.attn = AttnScratch(1 if prefill else rows, HL, c.qk_dim, capacity, dev)
        if prefill:
            kda_layers = [l for l in w.layers if l.kind == "kda"]
            width = kda_layers[0].kda.proj.n if kda_layers else 0
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
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
        self.eact = torch.empty((rows * slots, ml), dtype=bf, device=dev)
        exl3 = c.quant == "exl3"
        self.ey = None if exl3 else torch.empty((rows, slots, D), dtype=bf if prefill else f32, device=dev)
        self.plan = None if exl3 else grouped.Plan(rows, slots, c.experts + 1, dev, prefill=prefill)
        self.exl3 = None
        if c.quant == "exl3":            # EXL3 routed experts, and the shared expert as a BF16 MLP
            sl = c.shared_width // w.world
            shape = SimpleNamespace(dims=D, width=ml, count=c.experts)
            self.exl3 = Exl3Scratch(shape, rows, slots, device=dev)
            self.ey = self.exl3.y.view(rows, slots, D)
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sk = torch.empty((1 if prefill and not exl3 else 8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
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

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


class Caches:
    """Every per-token cache of the model over ``rows`` tokens of a shared pool (``pool.Arena``): per DSA layer its
    latents (or keys and values without the latent cache); the MTP layer's; and with long contexts per indexed layer
    (the DSA layers, then the MTP layer) its index keys, gates and pooled keys (a row a pool of 4 tokens, 2 rows of
    pad). Element types and layouts are decided here; a ``State`` views them over its extent (``bind``)."""

    def __init__(self, w: Weights, rows: int) -> None:
        from .pool import Arena, Plane

        c = w.cfg
        dev = w.device
        HL = c.heads // w.world
        dsa = len([l for l in w.layers if l.kind == "dsa"])
        self.latent = latent.ENABLED
        planes: list[Plane] = []

        def add(shape, div: int = 1, pad: int = 0) -> int:
            planes.append(Plane(torch.zeros((rows // div + pad, *shape), dtype=torch.bfloat16, device=dev), div, pad))
            return len(planes) - 1

        def attention_planes() -> tuple[int, int | None]:
            if self.latent:          # one 512-wide latent a token and layer (kc), no separate values (vc)
                return add((c.kv_lora,)), None
            return add((HL, c.qk_dim)), add((HL, c.v_dim))

        self.kc, self.vc = [], []
        for _ in range(dsa):
            k, v = attention_planes()
            self.kc.append(k)
            self.vc.append(v)
        self.mtp = attention_planes() if w.mtp is not None else None
        # DSA indexer caches (long contexts only): per layer (and the MTP layer, last) keys, gates, pool keys
        self.index = None
        if w.meta.get("long_context"):
            n_idx = dsa + (1 if w.mtp is not None else 0)
            self.index = [(add((c.index_dim,)), add((c.index_dim,)), add((c.index_dim,), 4, 2)) for _ in range(n_idx)]
        self.arena = Arena(rows, planes)
        self.rows = rows

    def bind(self, st: State, base: int, size: int) -> None:
        """Point ``st``'s caches at tokens [base, base + size) of the pool."""

        view = lambda i: None if i is None else self.arena.view(i, base, size)     # noqa: E731
        st.kc = [view(i) for i in self.kc]
        st.vc = [view(i) for i in self.vc]
        if self.mtp is not None:
            st.mtp_kc, st.mtp_vc = view(self.mtp[0]), view(self.mtp[1])
        st.index = None if self.index is None else [tuple(view(i) for i in trio) for trio in self.index]


class Slots:
    """What each of ``count`` concurrent streams holds whatever its length: KDA recurrent states [count, 2, layers,
    H, 128, 128] (two parities) and conv windows [count, layers, 3, C], the decode window's KDA projections and replay
    scratch (so a window's commit can come after another stream's forward), and the device positions a captured graph
    reads. A ``State`` in slot s views slot s."""

    def __init__(self, w: Weights, count: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        LL = c.lin_heads // w.world
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.count, self.rows, self.layers = count, rows, n
        self.conv = torch.zeros((count, n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((count, 2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.proj = torch.zeros((count, n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch = [kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None for _ in range(count)]
        self.pos_dev = torch.zeros((count, 1), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((count, 1), dtype=torch.int32, device=dev)

    def nbytes(self) -> int:
        t = [self.conv, self.rec, self.proj, self.pos_dev, self.mtp_pos_dev]
        s = [x for set_ in self.scratch if set_ is not None for x in (set_.out, set_.k, set_.v, set_.g, set_.b)]
        return sum(x.numel() * x.element_size() for x in t + s)


def slot_bytes(w: Weights, rows: int) -> int:
    """Device bytes one more stream slot takes (``Slots``), without allocating it."""

    c = w.cfg
    LL = c.lin_heads // w.world
    kda_layers = [l for l in w.layers if l.kind == "kda"]
    n = len(kda_layers)
    width = kda_layers[0].kda.proj.n if kda_layers else 0
    DK = DV = 128
    scratch = rows * LL * DV * 2 + rows * LL * DK * 4 + rows * LL * DV * 2 + rows * LL * DK * 4 + rows * LL * 4
    return (n * (c.conv - 1) * 3 * LL * 128 * 2 + 2 * n * LL * 128 * 128 * 4 + n * rows * width * 2 + n * scratch
            + 16)


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer): views of its extent [base, base +
    capacity) of the pool's ``Caches`` and of its stream slot (``Slots``). Without ``caches`` / ``slots`` it makes its
    own (one sequence of ``capacity`` tokens, one slot), as it always held tensors of its own."""

    def __init__(self, w: Weights, capacity: int, rows: int, *, caches: Caches | None = None, base: int = 0,
                 slots: Slots | None = None, slot: int = 0) -> None:
        self.caches = caches if caches is not None else Caches(w, capacity)
        self.slots = slots if slots is not None else Slots(w, 1, rows)
        self.slot = slot
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        sl = self.slots
        self.pos = 0
        self.pos_dev = sl.pos_dev[slot]
        self.mtp_pos_dev = sl.mtp_pos_dev[slot]
        self.conv = sl.conv[slot]
        self.rec = sl.rec[slot]
        self.cur = [0] * n
        self.proj = sl.proj[slot]
        self.scratch_set = sl.scratch[slot]
        self.scratch = self.scratch_set.views if n else []
        self.latent = latent.ENABLED
        self.mtp_len = 0
        self.mtp_drafted = 0
        self.bind(base, capacity)

    def bind(self, base: int, capacity: int) -> None:
        """View the pool's tokens [base, base + capacity) (a new extent, grown or moved: the rows are the caller's)."""

        self.base, self.capacity = int(base), int(capacity)
        self.caches.bind(self, self.base, self.capacity)

    @property
    def graph_key(self) -> tuple[int, int]:
        """What a captured graph baked in besides the shared buffers: the slot and the extent's base."""
        return self.slot, self.base

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
        """A copy with tensors of its own (no longer the pool's or the slot's views; not for captured graphs)."""
        import copy

        other = copy.copy(self)
        other.proj = self.proj.clone()
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() if x is not None else None for x in self.vc]
        if self.index is not None:
            other.index = [tuple(x.clone() for x in trio) for trio in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone() if self.mtp_vc is not None else None
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


def mm(b: Buffers, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False) -> torch.Tensor:
    """A projection: 4-bit ones of a prompt chunk on the shared prefill matmul, the rest on ``qmm.matmul``."""

    if b.prefill and isinstance(q, qmm.Q4):
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return qmm.matmul(x, q, xs, out=out, f32=f32, part=b.sk)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int) -> torch.Tensor:
    mm(b, x, q, xs, b.part[:R], f32=True)
    return gather(w, b, R)


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, cut: Cut | None = None) -> torch.Tensor:
    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    p = b.kproj[0, :R] if b.prefill else st.proj[li, :R]
    mm(b, b.normed[:R], k.proj, b.xs[:R], p)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    pre = b.prefill
    mm(b, fa, k.fb, None if pre else qmm.group_sums(fa, b.xs_fa[:R]), b.ka[:R])
    mm(b, ga, k.gb, None if pre else qmm.group_sums(ga, b.xs_ga[:R]), b.kg[:R])
    cur = st.cur[li]
    if cut is None:
        out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                            k.dt_bias, k.norm, c.eps, c.lower, R, b.kscratch if pre else st.scratch[li],
                            st.rec[1 - cur, li])
    else:
        n = cut.point
        first = kda_mod.chain(p[:n], k.b_off, b.ka[:n], b.kg[:n], st.conv[li], k.conv, st.rec[cur, li],
                              k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n, b.kscratch, cut.rec[li]).clone()
        _shift_conv(cut.conv[li:li + 1], b.kproj[:, :n], n)
        rest = kda_mod.chain(p[n:], k.b_off, b.ka[n:R], b.kg[n:R], cut.conv[li], k.conv, cut.rec[li],
                             k.a_log, k.dt_bias, k.norm, c.eps, c.lower, R - n, b.kscratch, st.rec[1 - cur, li])
        out = torch.cat((first, rest))
    if pre:                              # a prompt chunk keeps every row: the layer commits now
        st.cur[li] = 1 - cur
        _shift_conv(st.conv[li:li + 1], b.kproj[:, :R], R)
    return out_proj(w, b, out, k.o, None if pre else qmm.group_sums(out, b.kxs[:R]), R)


def kda_segments(layer: LayerW, w: Weights, b: Buffers, R: int, seg: torch.Tensor, slots: Slots,
                 proj: torch.Tensor, scratch, li: int) -> torch.Tensor:
    """kda_block's decode path for a multi-stream window (``seg``: kda.segment_table, set): the projections on every
    row at once (``proj`` [layers, rows, width]: the window's, which the commit's conv shift reads), the chain per
    segment from its slot's state (``kda.chain_segments``: each segment's bits alone; always the wide path, as
    decode windows run by default), ``scratch``: the window's replay scratch (a KDAScratchSet)."""

    c = w.cfg
    k = layer.kda
    p = proj[li, :R]
    with prof.timed("kda: projections"):
        mm(b, b.normed[:R], k.proj, b.xs[:R], p)
        fa = p[:, k.fa_off:k.fa_off + 128]
        ga = p[:, k.ga_off:k.ga_off + 128]
        mm(b, fa, k.fb, qmm.group_sums(fa, b.xs_fa[:R]), b.ka[:R])
        mm(b, ga, k.gb, qmm.group_sums(ga, b.xs_ga[:R]), b.kg[:R])
    with prof.timed("kda: recurrence"):
        out = kda_mod.chain_segments(seg, p, k.b_off, b.ka[:R], b.kg[:R], slots.conv[:, li], k.conv,
                                     slots.rec[:, :, li], k.a_log, k.dt_bias, k.norm, c.eps, c.lower, R,
                                     scratch.views[li])
    with prof.timed("kda: out + all-gather"):
        return out_proj(w, b, out, k.o, qmm.group_sums(out, b.kxs[:R]), R)


def dsa_block(layer: LayerW, w: Weights, kc: torch.Tensor, vc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers,
              R: int, nch: int | None, index=None, host_pos: int | None = None,
              sparse_np: int | None = None) -> torch.Tensor:
    """Write every index key and pool; rows past the dense limit attend to their top-512 pools (host_pos eager, or sparse_np in a captured graph)."""

    c = w.cfg
    a = layer.dsa
    mm(b, b.normed[:R], a.proj, b.xs[:R], b.dp[:R])
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    if a.absorb is not None:
        return _dsa_latent(a, w, kc, pos_dev, b, R, nch, index, host_pos, sparse_np)
    if sparse_np is not None:
        raise ValueError("sparse CUDA graphs need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.lat[:R], a.kv_k, b.xs_lat[:R], b.kn[:R].view(R, HL * c.qk_dim))
    mm(b, b.lat[:R], a.kv_v, b.xs_lat[:R], b.vn[:R].view(R, HL * c.v_dim))
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
        glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:     # every row sparse: the dense pass is skipped
        o = torch.empty((R, HL, c.v_dim), dtype=torch.bfloat16, device=b.q.device) if b.prefill else b.attn.out[:R]
    elif b.prefill:
        o = prefill_attention.attention(b.q[:R], kc, vc, host_pos, scale=c.qk_dim ** -0.5)
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, None if b.prefill else qmm.group_sums(o, b.xs_ao[:R]), R)


def dense_attention(qa: torch.Tensor, lc: torch.Tensor, pos_dev: torch.Tensor, s: latent.LatentScratch, *,
                    scale: float, nch: int, out: torch.Tensor) -> torch.Tensor:
    """``latent.attention`` in blocks of ``part_rows`` rows, each at its first row's position: one call's bits a row."""

    R, step = qa.shape[0], s.part_rows
    hb = latent.head_block(R)
    if R <= step:
        return latent.attention(qa, lc, pos_dev, s, scale=scale, nch=nch, out=out, hb=hb)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        at = pos_dev if r0 == 0 else pos_dev + r0          # the block's first row's position, on the device
        latent.attention(qa[r0:r1], lc, at, s, scale=scale, nch=nch, out=out[r0:r1], hb=hb)
    return out


def _dsa_latent(a, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int, nch: int | None,
                index, host_pos: int | None, sparse_np: int | None = None) -> torch.Tensor:
    """DSA on the latent cache: the same indexer and selection, attention over latents with kv_b's key blocks absorbed into the query."""

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.latent_write(b.lat[:R], lc, pos_dev)
    # sparse_np: every row is past the dense limit (a captured graph); else the host position decides
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
            glue.router(b.normed[:R], ix.gate, b.igr[:R])
            sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        # Rows past the dense limit are recomputed sparsely below, so the dense pass needs only the chunks up to it.
        with prof.timed("dsa: dense attention"):
            dense_attention(qa, lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol)
    if sparse_rows:
        with prof.timed("dsa: select tokens"):
            mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
            tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                                  pk.shape[0] - 2, pos_dev, bucket=sparse_np)
        with prof.timed("dsa: sparse attention"):
            latent.sparse_attention(qa, lc, tokens, counts, ol, scale)
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R)


def dsa_segments(layer: LayerW, w: Weights, lc: torch.Tensor, b: Buffers, R: int, rows, sel=None, index=None, *,
                 select: bool = True) -> torch.Tensor | None:
    """dsa_block on the latent cache for a multi-stream window (``rows``: segments.SegRows, already set): several
    streams' rows back to back, each stream's latents (lc), index keys, gates and pooled keys (index: arenas) in
    its own extent. Each segment's rows get the bits of that segment alone through dsa_block (the projections,
    absorb and expand keep rows apart; the attention kernels are latent.seg_* / sparse.seg_*). Graphs: one per R
    covers every position mix. ``select`` False skips the token selection (eager windows with no sparse row:
    ``rows.any_sparse()``); ``sel``: segments.SelectScratch, needed with an index."""

    c = w.cfg
    a = layer.dsa
    if a.absorb is None:
        raise ValueError("segmented windows need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.normed[:R], a.proj, b.xs[:R], b.dp[:R])
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.seg_latent_write(b.lat[:R], lc, rows)
    tokens = counts = None
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
            glue.router(b.normed[:R], ix.gate, b.igr[:R])
            sparse.seg_index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, rows)
        if select:
            with prof.timed("dsa: select tokens"):
                mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
                tokens, counts = sparse.seg_select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, rows, sel)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    with prof.timed("dsa: attention"):
        ol = latent.seg_attention(qa, lc, rows, tokens, counts, s, scale=c.qk_dim ** -0.5, out=s.ol[:R])
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R)


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    mm(b, b.normed[:R], m.gu, b.xs[:R], b.gu[:R])
    glue.swiglu(b.gu[:R], b.act[:R], b.xs_act[:R], w.cfg.limit)
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        if m.shared is None:
            grouped.route(b.pick[:R], b.plan)
    if m.shared is not None:
        # EXL3: the routed slots through the trellis kernels, the shared expert (last slot) through BF16 matmuls
        exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit)
        s = m.shared
        mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R])
        glue.swiglu(b.sgu[:R], b.sact[:R], b.sxs[:R], c.limit)
        mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True)
        b.ey[:R, c.top_k].copy_(b.sy[:R])
        glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
        return gather(w, b, R)
    with prof.timed("moe: gate/up"):
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)
    with prof.timed("moe: combine"):
        glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None, cut: Cut | None = None,
                  mixer=None) -> None:
    """``mixer``: the layer's KDA / DSA block as ``mixer(layer)`` instead of on ``st`` (a multi-stream window's
    segmented blocks, ``verify.BatchedVerify``); everything else is the same calls."""

    c = w.cfg
    x = b.x[:R]
    h = layer.attn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    if mixer is not None:
        g = mixer(layer)
    elif layer.kind == "kda":
        with prof.timed("kda"):
            g = kda_block(layer, w, st, b, R, cut)
    else:
        di = st.dsa_index[layer.index]
        with prof.timed("dsa (total)"):
            g = dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch,
                          st.index[di] if st.index is not None else None, host_pos, sparse_np)
    with prof.timed("hc"):
        glue.hc_post(x, x, g, b.post[:R], b.comb[:R])
        h = layer.ffn_hc
        glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
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
            host_pos: int | None = None, sparse_np: int | None = None, cut: Cut | None = None, mixer=None):
    """Run capturable GPU work on static buffers and device positions; eager long contexts use host_pos (graphs sparse_np) to select sparse attention; ``mixer``: each layer's KDA / DSA block (``layer_forward``; ``st`` unused)."""

    if cut is not None and (not b.prefill or not 0 < cut.point < R):
        raise ValueError("a prompt cut must lie inside a prefill chunk")
    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, cut, mixer)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    glue.stream_mean(b.x[:R], b.hidden[:R])
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits and hidden buffer views for token rows, leaving committed state unchanged until commit."""

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


def _shift_conv(conv: torch.Tensor, proj: torch.Tensor, keep: int) -> None:
    """conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [L, R, W >= C])."""

    n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    _conv_shift[(n, triton.cdiv(C, 1024))](conv, proj, keep, conv.stride(0), proj.stride(0), proj.stride(1), C=C,
                                           TAPS=taps, BLOCK=1024, num_warps=4)


@triton.jit
def _conv_shift_seg(CONV, PROJ, SEG, conv_slot, conv_layer, proj_layer, proj_row, C: tl.constexpr,
                    TAPS: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    """Program (layer, channel block, segment): ``_conv_shift`` of the segment's conv slot over its own rows."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = SEG + tl.program_id(2) * COLS
    row0 = tl.load(e + 0).to(tl.int64)
    rows = tl.load(e + 1)
    cslot = tl.load(e + 4).to(tl.int64)
    keep = tl.load(e + 5)
    if rows > 0:
        conv = CONV + cslot * conv_slot
        proj = PROJ + row0 * proj_row
        v0 = _row(conv, proj, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
        v1 = _row(conv, proj, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
        v2 = _row(conv, proj, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
        tl.store(conv + l * conv_layer + c, v0, mask=c < C)
        tl.store(conv + l * conv_layer + C + c, v1, mask=c < C)
        tl.store(conv + l * conv_layer + 2 * C + c, v2, mask=c < C)


def conv_shift_segments(conv: torch.Tensor, proj: torch.Tensor, seg: torch.Tensor) -> None:
    """``_shift_conv`` per segment of a multi-stream window: conv [conv slots, L, 3, C] (in place; each [3, C]
    contiguous), proj [L, R, W >= C] the window's projection rows; segment i (``kda.segment_table``) has its conv
    slot take rows keep .. keep + 2 of [conv[slot]; its own proj rows]. Segments of 0 rows are skipped."""

    _, n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    if conv.stride(3) != 1 or conv.stride(2) != C:
        raise ValueError("each conv slot's [3, C] window must be contiguous")
    if seg.dtype != torch.int32 or seg.dim() != 2 or seg.shape[1] != kda_mod.SEG_COLS or not seg.is_contiguous():
        raise ValueError("segments: a contiguous int32 [nseg, SEG_COLS] table")
    _conv_shift_seg[(n, triton.cdiv(C, 1024), seg.shape[0])](
        conv, proj, seg, conv.stride(0), conv.stride(1), proj.stride(0), proj.stride(1), C=C, TAPS=taps,
        COLS=kda_mod.SEG_COLS, BLOCK=1024, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
