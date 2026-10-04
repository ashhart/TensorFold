"""One sequence's device state (caches, rings, position) and the scratch of a decode window or prompt chunk, per rank."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
from tensorfold.families.glm5_next.cuda.glue import HC_BLOCKS

from ..config import Config
from . import MOE_WINDOW, PREFILL_ROWS, SCORE_BYTES

HC_PART = 32            # glue._hc_partial's values a K block: the 24 mixing dots and the sum of squares, padded
ATTN_CHUNK = 128        # list entries one sparse-attention decode partial covers


def score_rows(entries: int) -> int:
    """Prompt rows the indexer scores at once over ``entries`` cache entries: SCORE_BYTES of fp32, 16 to 2048."""

    rows = SCORE_BYTES // (4 * max(entries, 1))
    return max(16, min(PREFILL_ROWS, rows)) // 16 * 16


def entries(ratio: int, capacity: int) -> int:
    """Compressed entries a cache of ``capacity`` slots holds: entry j pools positions [j*ratio, (j+1)*ratio)."""

    return capacity // ratio + 1 if ratio > 1 else capacity


def _device_bytes(obj, device: torch.device) -> int:
    """Bytes of the tensors ``obj`` holds on ``device``'s kind (pinned host staging excluded)."""

    total, seen, stack = 0, set(), list(vars(obj).values())
    while stack:
        v = stack.pop()
        if isinstance(v, torch.Tensor):
            if v.device.type == device.type and id(v) not in seen:
                seen.add(id(v))
                total += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (Exl3Scratch, grouped.Plan)):
            stack.extend(vars(v).values())
    return total


class State:
    """Committed state of one sequence: rings of committed rows, position-addressed compressed and index-K caches."""

    def __init__(self, cfg: Config, capacity: int, device: torch.device | str = "cuda") -> None:
        dev = torch.device(device)
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        self.ratio = {layer: cfg.roles[layer].ratio for layer in cfg.kv_source_layer_ids}
        if any(r not in (1, 2) for r in self.ratio.values()):
            raise ValueError(f"KV source ratios {self.ratio}: the engine compresses by 1 or 2")
        self.capacity, self.device = capacity, dev
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=i32, device=dev)
        # slot pos % window per layer, DSpark stages last; written only at commit
        self.rings = torch.zeros((len(cfg.roles), cfg.sliding_window, cfg.head_dim), dtype=bf, device=dev)
        self.comp = {layer: torch.zeros((entries(r, capacity), cfg.head_dim), dtype=bf, device=dev)
                     for layer, r in self.ratio.items()}
        self.index_k = {layer: torch.zeros((entries(r, capacity), cfg.index_head_dim), dtype=bf, device=dev)
                        for layer, r in self.ratio.items()}
        self.pooled = tuple(layer for layer, r in self.ratio.items() if r == 2)   # tail slots, in this order
        self.tail = torch.zeros((len(self.pooled), 2, cfg.head_dim), dtype=f32, device=dev)   # (kv, score)
        self.tail_valid = torch.zeros((len(self.pooled),), dtype=i32, device=dev)
        self.history: list[int] = []     # the last engram_max_ngram_size - 1 committed ids, oldest first

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def reset(self) -> None:
        """An empty sequence; rings and caches are read only below ``pos``, so they keep their bytes."""

        self.tail_valid.zero_()
        self.history = []
        self.set_pos(0)

    def row_views(self, n: int) -> list[torch.Tensor]:
        """The position-addressed rows the first ``n`` tokens wrote: each KV source's compressed rows, then index-K."""

        out = []
        for layer, r in self.ratio.items():
            k = n // r + 1
            out += [self.comp[layer][:k], self.index_k[layer][:k]]
        return out

    def nbytes(self) -> int:
        return _device_bytes(self, self.device)


class Buffers:
    """Scratch for a window of up to ``rows`` rows (sliced [:R]); ``prefill``: a prompt chunk's, every output kept."""

    def __init__(self, cfg: Config, world: int, rows: int, capacity: int, prefill: bool = False,
                 device: torch.device | str = "cuda") -> None:
        dev = torch.device(device)
        host = dev.type == "cuda"        # pinned staging and events exist for a real device only
        bf, f32, i32, u8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8
        D, S, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
        HL = cfg.num_attention_heads // world
        V = cfg.vocab_size // world
        NI = cfg.moe_intermediate_size // world
        self.rows, self.world, self.capacity, self.prefill, self.device = rows, world, capacity, prefill, dev

        def t(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dtype, device=dev)

        # ids and streams
        self.ids = torch.zeros((rows,), dtype=i32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=i32, pin_memory=True) if host else None
        self.staged = torch.cuda.Event() if host else None
        self.X = t((rows, S, D), bf)
        self.pre_in = t((rows, S), f32)
        # mHC, attention side then FFN side
        self.hcpart = t((rows, HC_BLOCKS, HC_PART), f32)
        self.pre_a, self.post_a, self.comb_a = t((rows, S), f32), t((rows, S), f32), t((rows, S * S), f32)
        self.pre_f, self.post_f, self.comb_f = t((rows, S), f32), t((rows, S), f32), t((rows, S * S), f32)
        self.xn = t((rows, D), bf)
        # attention
        self.qakv = t((rows, cfg.q_lora_rank + hd), bf)
        self.qr = t((rows, cfg.q_lora_rank), bf)
        self.q = t((rows, HL, hd), bf)
        self.o = t((rows, HL, hd), bf)
        self.u = t((rows, cfg.o_groups // world * cfg.o_lora_rank), bf)
        self.kvw = t((len(cfg.roles), rows, hd), bf)                 # this forward's window KV, every layer
        chunks = -(-(cfg.sliding_window + cfg.index_topk) // ATTN_CHUNK)
        self.attn_part = None if prefill else t((rows, HL, chunks, hd + 2), f32)   # (acc, max, sum)
        # compressor and indexer
        sources = [cfg.roles[layer].ratio for layer in cfg.kv_source_layer_ids]
        self.cmp = t((sources.count(2), rows, 2, hd), f32)
        self.lat = t((rows, hd), bf)
        self.kI, self.epos = t((rows, cfg.index_head_dim), bf), t((rows,), i32)   # index keys, entry positions
        self.qI = t((rows, cfg.index_n_heads, cfg.index_head_dim), bf)
        self.wI = t((rows, cfg.index_n_heads), f32)
        visible = max((entries(r, capacity) for r in sources), default=0)
        scored = (min(rows, score_rows(visible)) if prefill else rows) if visible else 0
        self.scores = t((scored, visible), f32)
        self.cand, self.cand_n = t((rows, cfg.candidate_topk_blocks), i32), t((rows,), i32)
        self.lists, self.list_n = t((rows, cfg.index_topk), i32), t((rows,), i32)   # the latest index layer's
        # MoE: routed slots only, the shared expert added to the fp32 share after them
        E, K = cfg.n_routed_experts, cfg.num_experts_per_tok
        SI = cfg.moe_intermediate_size * cfg.n_shared_experts // world
        self.mlog, self.pick, self.wts = t((rows, E), f32), t((rows, K), i32), t((rows, K), f32)
        self.exl3 = Exl3Scratch(SimpleNamespace(dims=D, width=NI, count=E), min(rows, MOE_WINDOW) if prefill else rows,
                                K, device=dev)
        self.sgu, self.sact, self.sd = t((rows, 2 * SI), bf), t((rows, SI), bf), t((rows, D), f32)
        self.part = t((rows, D), f32)
        self.gath = t((world, rows, D), f32)
        # Engram: the rank's columns of both layers as file bytes, then dequantized, gathered, in column order
        n_eng = len(cfg.engram_layer_ids)
        cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads // world
        raw = (rows, n_eng, cols, cfg.engram_head_dim + cfg.engram_head_dim // 32)   # e4m3 bytes, e8m0 per 32
        self.eraw = t(raw, u8)
        halves = 2 if prefill else 1     # a prompt reads chunk i + 1 into one half while chunk i's half copies
        self.eraw_host = [torch.zeros(raw, dtype=u8, pin_memory=True) for _ in range(halves)] if host else []
        self.eraw_done = [torch.cuda.Event() for _ in range(halves)] if host else []   # each half's last H2D
        width = cols * cfg.engram_head_dim
        self.eloc = t((rows, n_eng, width), bf)
        self.egat = t((world, rows, n_eng, width), bf)
        self.eng = t((rows, n_eng, world * width), bf)
        kv = D * (S + 1) // world
        self.ekv, self.ekv_gat = t((rows, kv), bf), t((world, rows, kv), bf)
        # DSpark taps of the last rows, final norm, head
        taps = min(rows, cfg.sliding_window)
        self.taps = t((taps, len(cfg.dspark_target_layer_ids), D), bf)
        self.hidden, self.fnormed = t((rows, D), bf), t((rows, D), bf)
        self.logits = t((1 if prefill else rows, V), f32)
        # DSpark: main_x of the absorbed rows; the block is proposed after decode windows only
        self.mx, self.mx_gat = t((taps, D), bf), t((world, taps, D // world), bf)
        if not prefill:
            B, DK = cfg.dspark_block_size, cfg.dspark_num_experts_per_tok + 1     # the shared expert last
            self.xd, self.bkv = t((B, S, D), bf), t((B, hd), bf)
            self.dmlog = t((B, cfg.dspark_n_routed_experts), f32)
            self.dpick, self.dwts = t((B, DK), i32), t((B, DK), f32)
            self.dplan = grouped.Plan(B, DK, cfg.dspark_n_routed_experts + 1, dev)
            self.dact, self.dey = t((B, DK, NI), bf), t((B, DK, D), f32)
            self.dlog = t((B, V), f32)
            self.me, self.conf = t((B, cfg.dspark_markov_rank), bf), t((B,), f32)

    def nbytes(self) -> int:
        """Device bytes of this scratch, the figure ``bytes`` gives without allocating."""

        return _device_bytes(self, self.device)


def bytes(cfg: Config, world: int, rows: int, capacity: int, prefill: bool = False) -> int:
    """Device bytes ``Buffers(cfg, world, rows, capacity, prefill)`` allocates: built on the meta device."""

    return Buffers(cfg, world, rows, capacity, prefill, device="meta").nbytes()
