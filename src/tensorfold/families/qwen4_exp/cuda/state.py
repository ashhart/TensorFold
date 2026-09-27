"""Flash Next's static per-window buffers, so a step can be graph-captured, and one sequence's committed state."""

from __future__ import annotations

import numpy as np
import torch

from . import attention as attn_mod
from . import gdn as gdn_mod
from . import moe as moe_mod
from .weights import Weights


# -- per-window buffers --------------------------------------------------------------------------------
CAND = 32      # tensor parallel: candidates a rank gathers per row for sampling (top-k 20 plus the sampler's margin 8)
ATT_ROWS = 256  # prompt attention runs in blocks of this many rows (its partials scale with rows x context)


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        wide = c.streams * c.hidden
        self.rows, self.prefill = rows, prefill
        head_rows = 1 if prefill else rows
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
        self.moe = moe_mod.MoEBuffers(rows, _MoECfg(c), dev, prefill=prefill)
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
        self.mtp_drafted = 0
        self.set_mtp_len(snap["mtp_len"])
