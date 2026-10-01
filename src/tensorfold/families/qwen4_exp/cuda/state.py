"""Flash Next's static per-window buffers, so a step can be graph-captured, and one sequence's committed state."""

from __future__ import annotations

import numpy as np
import torch

from tensorfold.cuda import moe as moe_mod

from . import attention as attn_mod
from . import gdn as gdn_mod
from . import kvcache
from .weights import Weights


# -- per-window buffers --------------------------------------------------------------------------------
CAND = 32      # tensor parallel: candidates a rank gathers per row for sampling (top-k 20 plus the sampler's margin 8)
ATT_ROWS = 256  # prompt attention runs in blocks of this many rows (its partials scale with rows x context)
ENDS = 16       # prompts one prompt pass can end (each ending prompt's last row gets the head)


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int, *, prefill: bool = False,
                 moe_prefill: bool | None = None) -> None:
        c = w.cfg
        dev = w.device
        wide = c.streams * c.hidden
        self.rows, self.prefill = rows, prefill
        self.rope_rows = None          # an image prompt chunk's [rows, 3] positions, else None
        head_rows = ENDS if prefill else rows
        bf, f32 = torch.bfloat16, torch.float32
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.last = torch.zeros((rows,), dtype=torch.int32, device=dev)          # each stream's last row
        self.last_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
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
        self.attn = attn_mod.AttnScratch(min(rows, ATT_ROWS) if prefill else rows, c.heads, c.head_dim, capacity,
                                         dev, budget=c.index_budget, ratio=c.index_ratio)
        self.gated = torch.empty((rows, c.heads * c.head_dim), dtype=bf, device=dev)
        self.xs_gated = torch.empty((rows, c.heads * c.head_dim // 32), dtype=f32, device=dev)
        # the experts' prefill arithmetic for decode windows too (``moe_prefill``): their rows can share a pass's launch
        self.moe = moe_mod.MoEBuffers(rows, _MoECfg(c), dev, prefill=prefill if moe_prefill is None else moe_prefill)
        # DeltaNet projections and outputs and attention outputs; ``commit`` reads the projections' conv channels
        lin = 1 if prefill else sum(1 for layer in w.layers if layer.linear)     # a prompt chunk commits each layer
        self.proj = torch.zeros((lin, rows, gdn_mod.widths(c.nk, c.nv)[1]), dtype=bf, device=dev)
        if prefill:                    # the DeltaNet rows' conv taps ([conv state (3) | the chunk's rows]) and stream
            self.windows = (torch.arange(rows, dtype=torch.int32, device=dev)[:, None]
                            + torch.arange(c.conv_kernel, dtype=torch.int32, device=dev)).contiguous()
            self.sid = torch.zeros((rows,), dtype=torch.int32, device=dev)
            self.conv_ptr = torch.zeros((1,), dtype=torch.int64, device=dev)
            self.pos_blk = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.gout = torch.empty((rows, c.nv * gdn_mod.DV), dtype=bf, device=dev)
        self.gxs = torch.empty((rows, c.nv * gdn_mod.DV // 32), dtype=f32, device=dev)
        self.attn_o = torch.empty((rows, c.heads, c.head_dim), dtype=bf, device=dev)
        self.streams = torch.empty((rows, wide), dtype=bf, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        world = int(w.meta.get("world", 1))
        self.world = world
        if world > 1:                  # tensor parallel: fp32 partials and their rank-ordered gathers
            self.part_branch = torch.empty((rows, c.hidden), dtype=f32, device=dev)
            self.part_moe = torch.empty((rows, c.hidden), dtype=f32, device=dev)
            self.g_branch = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.g_moe = torch.empty((world * rows * c.hidden,), dtype=f32, device=dev)
            self.cand = torch.empty((head_rows, 2 * CAND + 1), dtype=f32, device=dev)
            self.cand_all = torch.empty((world * head_rows * (2 * CAND + 1),), dtype=f32, device=dev)
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
        self.ple_v = torch.zeros((nrow, dh), dtype=bf, device=dev)          # a bf16 table's rows, as they ship
        self.ple_hv = torch.zeros((nrow, dh), dtype=bf, pin_memory=pin)
        self.ple_emb = torch.empty((rows, c.ple_dim), dtype=bf, device=dev)
        self.xs_ple = torch.empty((rows, c.ple_dim // 32), dtype=f32, device=dev)
        self.ple_keys = torch.empty((rows, wide), dtype=bf, device=dev)
        self.ple_vals = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.ple_gated = torch.empty((rows, wide), dtype=bf, device=dev)
        self.ple_pss = torch.empty((rows, c.streams), dtype=f32, device=dev)
        self.ple_nrow = torch.empty((rows, wide), dtype=bf, device=dev)
        # split-K partials for the largest matmul of a window (prompt chunks take K whole)
        self.part = torch.empty((1 if prefill else 32 * max(rows, 4) * 2560,), dtype=f32, device=dev)
        # MTP
        self.mtp_e = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_xe = torch.empty((rows, c.hidden // 32), dtype=f32, device=dev)
        self.mtp_eo = torch.empty((rows, c.hidden), dtype=bf, device=dev)
        self.mtp_hn = torch.empty((rows, wide), dtype=bf, device=dev)
        self.mtp_xh = torch.empty((rows, wide // 32), dtype=f32, device=dev)
        self.mtp_hs = torch.empty((rows * c.streams, c.hidden), dtype=bf, device=dev)
        self.mtp_in = torch.empty((rows, wide), dtype=bf, device=dev)          # the MTP's input streams


def _rows(t: torch.Tensor, rows: int, keep: int) -> torch.Tensor:
    """``t`` reallocated with ``rows`` rows, its first ``keep`` copied."""

    other = torch.zeros((rows, *t.shape[1:]), dtype=t.dtype, device=t.device)
    keep = min(keep, t.shape[0], rows)
    other[:keep] = t[:keep]
    return other


class _MoECfg:
    def __init__(self, c) -> None:
        self.num_experts_per_tok = c.top_k
        self.num_experts = c.experts
        self.moe_intermediate_size = c.moe_width
        self.hidden_size = c.hidden


# -- committed state -----------------------------------------------------------------------------------
class State:
    """Committed caches of one sequence (and the MTP head's attention layer); grown by ``ensure`` up to ``limit``."""

    def __init__(self, w: Weights, capacity: int, max_rows: int, kv_dtype: str = "bf16", *,
                 limit: int | None = None) -> None:
        c = w.cfg
        dev = w.device
        self.kv_dtype = kvcache.check(kv_dtype)
        self.limit = int(capacity if limit is None else limit)     # the rows this sequence may grow to
        capacity = min(int(capacity), self.limit)
        self.capacity = capacity
        self.version = 0                 # counts reallocations: a graph's pointer table refreshes on a change
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        # text after an image prompt rotates at its cache position plus this offset (0: no images, the plain path)
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.rope_delta = 0
        self.rope_delta_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        lin = [l for l in w.layers if l.linear]
        att = [l for l in w.layers if not l.linear]
        self.lin_index = {l.index: i for i, l in enumerate(lin)}
        self.att_index = {l.index: i for i, l in enumerate(att)}
        n = len(lin)
        self.conv = torch.zeros((n, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, c.nv, c.dv, c.dk), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.scratch = [gdn_mod.GDNScratch(max_rows, dev, c.nk, c.nv) for _ in range(n)]
        self.ratio, self.index_dim = c.index_ratio, c.index_dim
        self.layers = len(att) + (w.mtp is not None)              # attention caches: the layers', the MTP head's
        self.row_bytes = kvcache.row_bytes(c.kv_heads, c.head_dim, self.kv_dtype) + c.index_dim * 2
        self.kc = [kvcache.KVCache(capacity, c.kv_heads, c.head_dim, dev, self.kv_dtype) for _ in att]
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
            self.mtp_kc = kvcache.KVCache(capacity, c.kv_heads, c.head_dim, dev, self.kv_dtype)
            self.mtp_ikc = torch.zeros((capacity, c.index_dim), dtype=torch.bfloat16, device=dev)
            self.mtp_pooled = torch.zeros((-(-capacity // c.index_ratio), c.index_dim), dtype=torch.bfloat16,
                                          device=dev)

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def cache_bytes(self, rows: int | None = None) -> int:
        """Bytes of the caches that grow with context at ``rows`` rows (now: ``capacity``), the MTP head's included."""

        return self.layers * self.layer_bytes(rows)

    def layer_bytes(self, rows: int | None = None) -> int:
        """One attention layer's share of ``cache_bytes``: what a resize holds twice at once."""

        rows = self.capacity if rows is None else int(rows)
        return rows * self.row_bytes + -(-rows // self.ratio) * self.index_dim * 2

    def ensure(self, rows: int, step: int = 8192) -> int:
        """Grow every context cache to hold ``rows`` rows (a ``step`` at a time, at most ``limit``); returns the new bytes."""

        rows = int(rows)
        if rows <= self.capacity:
            return 0
        if rows > self.limit:
            raise ValueError(f"context of {rows} rows past this sequence's {self.limit}-row window")
        return self.resize(min(self.limit, -(-rows // step) * step))

    def resize(self, rows: int) -> int:
        """Reallocate the context caches at ``rows`` rows, keeping every committed row; returns the bytes it added."""

        rows = int(rows)
        before = self.cache_bytes()
        keep = max(self.pos, self.mtp_len)
        if rows < keep:
            raise ValueError(f"a {rows}-row cache can't keep {keep} committed rows")
        blocks, kept_blocks = -(-rows // self.ratio), -(-keep // self.ratio)
        for i in range(len(self.kc)):            # a layer at a time: its old buffers go before the next one's come
            self.kc[i] = self.kc[i].resized(rows, self.pos)
            self.ikc[i] = _rows(self.ikc[i], rows, self.pos)
            self.pooled[i] = _rows(self.pooled[i], blocks, kept_blocks)
        if hasattr(self, "mtp_kc"):
            mtp_blocks = -(-self.mtp_len // self.ratio)
            self.mtp_kc = self.mtp_kc.resized(rows, self.mtp_len)
            self.mtp_ikc = _rows(self.mtp_ikc, rows, self.mtp_len)
            self.mtp_pooled = _rows(self.mtp_pooled, blocks, mtp_blocks)
        self.capacity = rows
        self.version += 1
        return self.cache_bytes() - before

    def reset(self, w: Weights) -> None:
        """An empty sequence in the same buffers (captured graphs keep pointing at them)."""

        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.ple_tail.zero_()
        self.ple_history = w.cfg.ngram(0).initial_history() if w.cfg.ple_layers else None
        self.ple_last = None
        self.set_pos(0)
        self.set_rope_delta(0)
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.mtp_drafted = 0
        self.set_mtp_len(0)

    def set_rope_delta(self, delta: int) -> None:
        self.rope_delta = int(delta)
        self.rope_delta_dev.fill_(int(delta))

    def clone(self) -> "State":
        """An independent copy (tests and A/B checks)."""

        import copy

        other = copy.copy(self)
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(other, name, value.clone())
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                setattr(other, name, [v.clone() for v in value])
            elif isinstance(value, kvcache.KVCache):
                setattr(other, name, value.clone())
            elif isinstance(value, list) and value and isinstance(value[0], kvcache.KVCache):
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
        """The committed state outside the cache rows; ``restore`` needs the cache rows below ``pos`` still in place."""

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
        self.set_rope_delta(0)                       # kept prompts are text only
        self.image_positions, self.image_rows, self.image_features = None, (), None
        self.mtp_drafted = 0
        self.set_mtp_len(snap["mtp_len"])
