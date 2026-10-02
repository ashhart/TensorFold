"""The serial two-rank DeepSeek-V4.1 forward: split EXL3 weights, position-addressed caches, rank-order reductions.

One request, R rows a call (a prompt chunk of up to 128 rows, or one decode row). The arithmetic follows the
single-GPU reference (``reference.py``, checked against vLLM); tensor parallelism splits attention heads, output
groups, expert widths and the vocabulary head, and every partial sum crosses ranks as fp32 added in rank order.
Contexts stay within the short-context regime (every compressed entry visible, no indexer) for now.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from tensorfold.cuda.capacity import gather_ints
from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.cuda.sampling import sample_rows

from .. import engram as E
from ..reference import inv_freq
from . import hc as hcf
from . import kernels as K
from .pool import ALIGN, align_up
from .weights import HCW, LayerW, Weights

BF, F32 = torch.bfloat16, torch.float32


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)
MAX_ROWS = 2048              # rows of one call (prompt chunks); every expert's weights are read once a chunk
RING = 4096                  # prompt staging rings: a 2048-row chunk plus the 127-token window
DRING = 256                  # a stream slot's decode rings: the 128-token window, a round's rows, margin
WINDOW_ROWS = 130            # ring rows a stream carries between prefill and decode (window + compressor pair)


class Comm:
    """Rank-order sums and gathers over NCCL; a single process (world 1) passes tensors through."""

    def __init__(self, nccl=None) -> None:
        self.nccl = nccl
        self.world = nccl.world if nccl is not None else 1
        self.rank = nccl.rank if nccl is not None else 0
        self.side = None

    def sum(self, partial: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return partial
        send = partial.contiguous().float()
        recv = torch.empty((self.world, *send.shape), dtype=F32, device=send.device)
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        total = recv[0].clone()
        for r in range(1, self.world):
            total += recv[r]
        return total

    def partials(self, partial: torch.Tensor, during=None) -> torch.Tensor:
        """Every rank's partial, stacked in rank order [world, ...] (the consumer adds them in order): fp32 for decode
        and verify windows, bf16 for prompt chunks (half the bytes; the prompt path has its own arithmetic)."""

        send = partial.contiguous().float() if partial.shape[0] <= PROMPT_ROWS else partial.to(BF).contiguous()
        if self.world == 1:
            return send[None]
        recv = torch.empty((self.world, *send.shape), dtype=send.dtype, device=send.device)
        if during is not None:                                  # side work while the gather waits (L2 warming)
            during()
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        return recv

    def partials_rows(self, make, R: int, during=None):
        """``partials`` of the rows ``make(r0, r1)`` computes, in PROMPT_BLOCKS row blocks for prompt chunks: each
        block's all-gather runs on a side stream while the next block computes (post() reads the block layout)."""

        if R <= PROMPT_ROWS or self.world == 1 or not PROMPT_OVERLAP:
            return self.partials(make(0, R), during if R <= PROMPT_ROWS else None)
        h = (triton_cdiv(R, PROMPT_BLOCKS) + 15) // 16 * 16
        main = torch.cuda.current_stream()
        if self.side is None:
            self.side = torch.cuda.Stream()
        buf = None
        for r0 in range(0, R, h):
            r1 = min(R, r0 + h)
            send = make(r0, r1).to(BF).contiguous()
            if buf is None:
                buf = torch.empty((self.world * R * send.shape[1],), dtype=BF, device=send.device)
            at = self.world * r0 * send.shape[1]
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.nccl.all_gather(send.view(-1), buf[at:at + self.world * send.numel()])
            send.record_stream(self.side)
        main.wait_stream(self.side)
        buf.record_stream(self.side)
        return hcf.SplitPartials(buf, self.world, h)

    def gather_last(self, part: torch.Tensor) -> torch.Tensor:
        """Concatenate each rank's slice of the last axis in rank order."""

        if self.world == 1:
            return part
        send = part.contiguous()
        recv = torch.empty((self.world, *send.shape), dtype=send.dtype, device=send.device)
        self.nccl.all_gather(send.view(-1), recv.view(-1))
        return torch.cat(list(recv), dim=-1)


@dataclass
class Caches:
    """Position-addressed state of one request (rows beyond the committed length are overwritten on reuse)."""

    cap: int
    swa: list[torch.Tensor]                     # per layer bf16 ring [RING, 512], RoPE'd window keys (= values)
    comp: dict[int, torch.Tensor]               # per kv source bf16 [cap // ratio + 1, 512], RoPE'd entries (+ spare)
    raw: dict[int, torch.Tensor]                # per ratio-2 source fp32 ring [RING, 1024]: projected kv | gate
    ik: dict[int, torch.Tensor]                 # per kv source bf16 [cap // ratio + 1, 128]: indexer keys
    ids: list[int] = field(default_factory=list)


SKIP_READS = os.environ.get("TF_SKIP_READS") == "1"   # timing experiments: stale Engram rows (wrong output)
# decode / verify: while an all-gather waits on the other rank, a side stream warms L2 with the weights read next
# (the router and shared expert before the MoE, the next layer's first projections before its attention)
L2_PREFETCH = os.environ.get("TF_L2_PREFETCH") == "1"
PF_WOA = os.environ.get("TF_PF_WOA") == "1"            # (experiment) wo_a weights into L2 during the attention core   # measured: no gain at one row (and idle gaps before the gather)
SKIP_SHARED = os.environ.get("TF_SKIP_SHARED") == "1"
PF_PROGRAMS = int(os.environ.get("TF_PF_PROGRAMS") or 4)
# decode / verify step as one graph: the graph waits on a pinned-memory flag for each table's rows (read by the host
# meanwhile) instead of three graphs launched around the reads (each graph switch left the GPU idle ~0.7 ms)
ONE_GRAPH = os.environ.get("TF_ONE_GRAPH", "1") != "0"  # timing experiments: no shared expert in decode (wrong output)
STEP_READ_THREADS = int(os.environ.get("TF_STEP_THREADS") or "16")   # decode/verify Engram row reads (few rows)
# decode/verify: each rank reads its share of a token's Engram rows and the graphs all-gather the bytes (same rows)
SPLIT_READS = os.environ.get("TF_SPLIT_READS", "1") != "0"
KV_FP8 = (os.environ.get("TF_DSV41_KV_FP8") or "1") == "1"   # compressed entries + indexer keys in fp8 (~1.8 KB a
#   token instead of 3.2; long parity unchanged: NLL 1.676 / 1.631 / 1.613 at 8K / 24K / 40K vs bf16 1.674 / 1.630 / 1.612)
BLOCKED_SELECT = True        # prompt chunks: segmented indexer top-k (no [rows, keys] fp32 matrix)
REUSE = os.environ.get("TF_DSV41_REUSE", "1") != "0"    # keep the live caches for a prompt that extends them
REUSE_MIN = 64               # shorter common prefixes start fresh
FUSE_HC = True               # prompt chunks: hc post + the next sublayer's pre in two launches (post_pre)
PAR_DECODE = os.environ.get("TF_PAR", "1") == "1"            # decode/verify rows: independent linears on side streams (Par)
PROMPT_ZB = tuple(int(v) for v in os.environ.get("TF_ZB", "121,101").split(","))       # v4 with bf16 Z (gate/up config, down config); None: fp32 Z (PROMPT_V3)
PROMPT_ROTX = True           # gate/up: rotate the token rows inside the expert kernel (no rot_in copies)
PROMPT_ZDT = torch.float16   # Z element type (fp16 stores acc / 64; bf16 also works)
PROMPT_BLOCKS = 2            # row blocks of the overlapped prompt all-gathers
PROMPT_OVERLAP = True        # prompt chunks: all-gather the first row block while the second computes
PROMPT_V3 = 103                # v3 config (experts_prompt.cu grouped_prompt3); None: v2
PROMPT_V2 = True             # smem-staged activations, K sliced (experts_prompt.cu grouped_prompt2_kernel)
PROMPT_KC = [80, 72]         # k tiles a slice for gate/up (K 5120) and down (K 1152)
PROMPT_WARPS = 8
PROMPT_MTP = [2, 2]          # 16-row member tiles sharing one weight decode
_Z2 = None
_ZB = None
PROMPT_CFG = [1, 1]          # prompt expert kernel config for gate/up and down (experts_prompt.cu)
# decode/verify rows at most (row-invariant kernels; a round of streams up to it); above: prompt chunks
PROMPT_ROWS = int(os.environ.get("TF_DSV41_DECODE_ROWS") or 32)
# decode graphs are captured at these key widths (tokens) besides the full limit; a step replays the narrowest that
# covers its rows' positions, so indexer scores, block choice and top-k run over [R, width // ratio] instead of the
# limit's (the same entries are chosen: past a row's position every score is -inf). TF_DSV41_WIDTHS=0: full only
WIDTHS = [int(v) for v in (os.environ.get("TF_DSV41_WIDTHS") or "16384,65536,262144").split(",") if int(v) > 0]
SHARED_GRAPH_POOL = os.environ.get("TF_DSV41_SHARED_GRAPHS", "1") != "0"   # all decode graphs on one memory pool


class RoundProfile:
    """TF_ROUND_PROF=1: host time between marks of a DSpark round and the GPU time of the draft and verify graphs."""

    def __init__(self) -> None:
        from collections import defaultdict

        self.host = defaultdict(list)
        self.gpu = defaultdict(list)
        self.events = {}
        self.t = 0.0

    def __bool__(self) -> bool:
        return True

    def start(self) -> None:
        self.t, self.events, self.t0 = time.perf_counter(), {}, time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.host[name].append(1e3 * (now - self.t))
        self.t = now

    def event(self, name: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events[name] = e

    def end(self, k: int) -> None:
        self.host["round"].append(1e3 * (time.perf_counter() - self.t0))
        ev = self.events
        if "d.gpu0" in ev:
            self.gpu["draft"].append(ev["d.gpu0"].elapsed_time(ev["d.gpu1"]))
        if "v.gpu0" in ev:
            self.gpu[f"verify k={k}"].append(ev["v.gpu0"].elapsed_time(ev["v.gpu1"]))

    def report(self) -> None:
        avg = lambda v: sum(v) / max(len(v), 1)
        print("[round host ms] " + ", ".join(f"{k} {avg(v):.2f}" for k, v in self.host.items()), flush=True)
        print("[round gpu ms] " + ", ".join(f"{k} {avg(v):.2f} (n={len(v)})" for k, v in self.gpu.items()), flush=True)
        self.host.clear()
        self.gpu.clear()


class Par:
    """Fork/join of independent small launches over side streams (decode and verify rows: short GEMVs leave DRAM
    idle between them; captured into the CUDA graphs as parallel branches). Arithmetic is unchanged."""

    def __init__(self, n: int = 4) -> None:
        self.streams = [torch.cuda.Stream() for _ in range(n)]

    def __call__(self, *fns):
        if not PAR_DECODE:
            return [f() for f in fns]
        main = torch.cuda.current_stream()
        outs, used = [None] * len(fns), []
        for i, f in enumerate(fns):
            if i == 0:
                continue
            st = self.streams[(i - 1) % len(self.streams)]
            if st not in used:
                st.wait_stream(main)
                used.append(st)
            with torch.cuda.stream(st):
                outs[i] = f()
        outs[0] = fns[0]()
        for st in used:
            main.wait_stream(st)
        return outs


def group_members(pick: torch.Tensor, E: int, s) -> tuple[torch.Tensor, torch.Tensor]:
    """The grouping kernel's tables built with torch (prompt chunks; its shared memory caps R * slots): expert ids
    ascending in ``ids`` (count in ``s.count``), each expert's members (row * 32 + slot, pick order) padded with -1."""

    slots = pick.shape[1]
    flat = pick.reshape(-1).long()
    counts = torch.bincount(flat, minlength=E)
    busiest = int(counts.max())
    used = torch.nonzero(counts).flatten()
    nu = used.numel()
    ids = s.ids[:nu]
    ids.copy_(used.int())
    s.count.fill_(nu)
    order = torch.sort(flat, stable=True).indices                 # entries grouped by expert, pick order within
    experts = flat[order]
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(flat.numel(), device=pick.device) - start[experts]
    slot_of = torch.full((E,), -1, dtype=torch.long, device=pick.device)
    slot_of[used] = torch.arange(nu, device=pick.device)
    members = s.members_buf[:nu * busiest].view(nu, busiest)
    members.fill_(-1)
    members[slot_of[experts], rank] = ((order // slots) * 32 + order % slots).int()
    return ids, members


def decode_slots(m, top_k: int) -> int:
    """Expert slots a decode / verify row takes: its routed picks, and the shared expert when folded in."""

    return top_k + (m.shared_id is not None)


def routed_prompt(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, ex, s, R: int, limit: float) -> torch.Tensor:
    """``ex3.routed`` for prompt chunks: the grouped kernel's grid spans member tiles up to the busiest expert's row
    count, not R (at 1,024 rows that is ~16x fewer, mostly empty, programs)."""

    ext = ex3._ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    ids, members = group_members(pick, E, s)
    if os.environ.get("TF_ROUTE_STATS") and R == MAX_ROWS:
        cnt = torch.bincount(pick.flatten().long(), minlength=E).float()
        q = torch.quantile(cnt, torch.tensor([0.1, 0.5, 0.9, 0.99], device=cnt.device)).tolist()
        print(f"[route] mean {cnt.mean():.1f} q10/50/90/99 {[round(v) for v in q]} max {cnt.max():.0f} "
              f"reads64 {(torch.ceil(cnt / 64).clamp(min=1) * (cnt > 0)).sum() / (cnt > 0).sum():.3f} "
              f"reads80 {(torch.ceil(cnt / 80).clamp(min=1) * (cnt > 0)).sum() / (cnt > 0).sum():.3f} "
              f"rows-in-64 {(cnt.sum() / (torch.ceil(cnt / 16) * 16).sum()):.2f} used {(cnt > 0).sum():.0f}", flush=True)
    rotx = PROMPT_ZB is not None and PROMPT_ROTX and x.dtype == BF and PROMPT_ZDT == torch.float16
    if not rotx:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
    from .experts_prompt import ext as prompt_ext

    pe = prompt_ext()
    if PROMPT_ZB is not None:
        global _ZB
        need = 2 * P * max(I, D)
        if _ZB is None or _ZB.numel() < need or _ZB.dtype != PROMPT_ZDT:
            _ZB = torch.empty((need,), dtype=PROMPT_ZDT, device=x.device)
        if rotx:                                    # token rows rotated per expert while staged (no xg / xu)
            xc = x.contiguous()
            pe.grouped_prompt4(xc, xc, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, _ZB, 2,
                               D, I, P, slots, ex.cb, PROMPT_ZB[0], ex.suh_g, ex.suh_u)
        else:
            pe.grouped_prompt4(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, _ZB,
                               2, D, I, P, slots, ex.cb, PROMPT_ZB[0])
        pe.gateup_epilogue_b(_ZB, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, P, I, E, float(limit))
        pe.grouped_prompt4(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, _ZB, 1,
                           I, D, P, slots, ex.cb, PROMPT_ZB[1])
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
        pe.down_combine_b(_ZB, pick, ex.svh_d, wts, out, R, D, slots, E)
        return out
    if PROMPT_V3 is not None:
        global _Z2
        need = 2 * P * max(I, D)
        if _Z2 is None or _Z2.numel() < need:
            _Z2 = torch.empty((need,), dtype=torch.float32, device=x.device)
        z = _Z2
        cfg_gu, cfg_d = PROMPT_V3 if isinstance(PROMPT_V3, tuple) else (PROMPT_V3, PROMPT_V3)
        pe.grouped_prompt3(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, z, 2, D, I,
                           P, slots, ex.cb, cfg_gu)
        ext.gateup_epilogue(z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1, slots, E, float(limit),
                            ex3.ACT_F32)
        pe.grouped_prompt3(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, z, 1, I,
                           D, P, slots, ex.cb, cfg_d)
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
        ext.down_combine(z, pick, ex.svh_d, s.y, wts, out, R, P, D, 1, slots, E)
        return out
    if PROMPT_V2:
        need = max(2 * (D // 16 // PROMPT_KC[0]) * I, (I // 16 // PROMPT_KC[1]) * D) * P
        if _Z2 is None or _Z2.numel() < need:
            _Z2 = torch.empty((need,), dtype=torch.float32, device=x.device)
        z = _Z2
        sk = pe.grouped_prompt2(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, z, 2,
                                D, I, P, slots, ex.cb, PROMPT_KC[0], PROMPT_MTP[0], PROMPT_WARPS)
        ext.gateup_epilogue(z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit),
                            ex3.ACT_F32)
        sk = pe.grouped_prompt2(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, z,
                                1, I, D, P, slots, ex.cb, PROMPT_KC[1], PROMPT_MTP[1], PROMPT_WARPS)
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
        ext.down_combine(z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E)
        return out
    z = s.z                                            # v1: one K split, the epilogues read SK = 1
    pe.grouped_prompt(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, z, 2, D, I,
                      P, slots, ex.cb, PROMPT_CFG[0])
    ext.gateup_epilogue(z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1, slots, E, float(limit),
                        ex3.ACT_F32)
    pe.grouped_prompt(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, z, 1, I,
                      D, P, slots, ex.cb, PROMPT_CFG[1])
    out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, 1, slots, E)
    return out


class _Fixed:
    """Always verify every draft (the fixed-k policy vLLM uses)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def choose(self) -> int:
        return self.n

    def update(self, k: int, accepted: int, ms: float) -> None:
        pass


class SerialEngine:
    def __init__(self, w: Weights, comm: Comm, engram_dir: str, tokenizer_json: str, *, cap: int = 4096,
                 device: str = "cuda", slots: int = 1, pool_tokens: int | None = None) -> None:
        self.w, self.c, self.comm, self.dev = w, w.cfg, comm, torch.device(device)
        # ``slots`` streams' caches side by side: prompt chunks run in one slot (``state``, views), decode graphs take
        # rows of any slots (``_sid``: each row's slot, so a row reads and writes only its own stream's caches)
        self.slots, self.slot, self._sid = int(slots), 0, None
        c = self.c
        self.freqs = {r: inv_freq(c, r, self.dev) for r in set(c.layer_ratios)}
        self.layout = E.Layout.from_config(c)
        self.tmap = E.token_map(tokenizer_json, c.engram_compressed_vocab_size)
        self.tables = E.Tables(engram_dir, c.engram_layer_ids)
        # every layer has the same shapes: one small scratch for decode / verify rows, one for prompt chunks whose
        # gate/up inputs and fp32 partials the prompt kernel no longer reads (rotated while staged, its own fp16 Z)
        small = ex3.Scratch(w.layers[0].moe.experts, PROMPT_ROWS, decode_slots(w.layers[0].moe, c.num_experts_per_tok))
        self.scratch = [small] * len(w.layers)
        self.scratch_prompt = ex3.Scratch(w.layers[0].moe.experts, MAX_ROWS, c.num_experts_per_tok)
        if PROMPT_ZB is not None and PROMPT_ROTX:
            sp = self.scratch_prompt
            sp.xg = sp.xu = sp.z = sp.y = torch.empty((0,), dtype=torch.float16, device=self.dev)
            torch.cuda.empty_cache()
        self.hcbuf = hcf.HCBuffers(MAX_ROWS, c.hidden_size, device=self.dev)
        self.par = Par()
        self._pf_stream, self._pf_moe, self._pf_attn, self._pf_woa = None, {}, {}, {}
        if PF_WOA:
            self._pf_stream = torch.cuda.Stream()
            for lw in w.layers:
                if lw.attn.wo_a_grouped is not None:
                    self._pf_woa[lw.index] = K.prefetch_table([lw.attn.wo_a_grouped.words])
        if L2_PREFETCH and comm.world > 1:
            self._pf_stream = torch.cuda.Stream()
            for i, lw in enumerate(w.layers):
                m = lw.moe
                self._pf_moe[lw.index] = K.prefetch_table([m.gate, m.shared[0].words, m.shared[1].words])
                if i + 1 < len(w.layers):
                    a = w.layers[i + 1].attn
                    self._pf_attn[lw.index] = K.prefetch_table([a.wq_a.words, a.wkv.words])
        self._rp = RoundProfile() if os.environ.get("TF_ROUND_PROF") else None
        self.pool = None                                            # tensorfold.cuda.kv_pool.PrefixPool, when kept
        self.state = None
        self.split = c.engram_layer_ids[1]
        n_cols, W = 3 * c.engram_n_heads, comm.world
        W = W if SPLIT_READS and W > 1 and n_cols % W == 0 else 1
        self.read_split = (W, n_cols // W)                  # (ranks sharing a row's reads, entries each)
        r = comm.rank if W > 1 else 0
        self.read_cols = slice(r * (n_cols // W), (r + 1) * (n_cols // W))
        self.tables_rope = {r: K.rope_tables(f, cap) for r, f in self.freqs.items()}
        self.attnbuf = K.AttnBuffers(K.FULL_ROWS, w.layers[0].attn.wq_b.n // c.head_dim, c.head_dim,   # prompt: _mqa_full
                                     c.index_topk + c.sliding_window, device=self.dev)
        self.topk: dict[int, torch.Tensor] = {}
        self.limit = cap
        self.candidates: torch.Tensor | None = None
        self.graph = None
        self.graphs: dict[int, dict] = {}                 # full-width decode graphs by rows
        self.narrow: dict[tuple[int, int], dict] = {}     # (rows, width) -> the graph at a narrower key width
        self.widths = sorted({w for w in WIDTHS if w < cap})
        self._width = cap                                 # the key width a graph being captured selects over
        # one memory pool for every decode graph (only one replays at a time): their scratch is the largest graph's,
        # not the sum over row counts and widths. A graph's outputs hold until the next decode replay.
        self._gpool = torch.cuda.graph_pool_handle() if SHARED_GRAPH_POOL else None
        self.drafter = None
        self.debug: list | None = None
        self._pinned: list = []
        self._pool = None
        self.adaptive = True
        self.taps: list[torch.Tensor] = []
        self.cap = cap
        # the per-token caches (compressed entries, indexer keys) of every slot in one arena of ``pool_tokens`` rows
        # (``pool.py``): a slot holds an extent [base, base + size) of it, ``bind`` places it; by default each slot
        # gets its own fixed extent of the per-stream limit (the layout before the shared pool)
        self.span = align_up(cap)
        self.pool_tokens = int(pool_tokens) if pool_tokens else self.slots * self.span
        if self.pool_tokens % ALIGN or self.pool_tokens < self.span:
            raise ValueError(f"a pool of {self.pool_tokens} tokens: a multiple of {ALIGN}, at least one {cap}-token "
                             "stream")
        self.reset()

    def enable_dspark(self, tokens: int = 3) -> None:
        from .dspark import DSpark

        self.drafter = DSpark(self, tokens)

    def reset(self) -> None:
        """Forget the current slot's request; caches are zeroed in place (a captured graph holds their addresses)."""

        if getattr(self, "state", None) is not None:
            # the compressed entries and indexer keys are left as they are: every reader masks the rows a stream
            # has not written (attention gathers selected visible entries, indexer scoring reads visible keys
            # only), and zeroing a long extent would write gigabytes a request; the rings are small and read whole
            st = self.state
            a, b = self.slot * DRING, (self.slot + 1) * DRING
            for t in [*self.big.swa, *self.big.raw.values()]:
                t[a:b].zero_()
            st.ids.clear()
            if self.drafter is not None:
                self.drafter.reset()
            return
        c, cap, S = self.c, self.cap, self.slots
        self.entries = {s_: cap // c.layer_ratios[s_] + 1 for s_ in c.kv_source_layer_ids}   # a stream's most
        P = self.pool_tokens
        # arena rows per source: the pool's entries, then one trash row that rows closing no compressor group write
        # (a decode graph writes every row; in a packed pool the extent's own last entry is the next one's first)
        E = {s_: P // c.layer_ratios[s_] + 1 for s_ in c.kv_source_layer_ids}
        self.trash = {s_: E[s_] - 1 for s_ in E}
        self.ebase = torch.zeros((S,), dtype=torch.long, device=self.dev)   # each slot's extent base (token rows)
        self.extents: list[tuple[int, int]] = [(0, 0)] * S
        # decode rings (DRING rows a slot) live per slot; prompt chunks run in one shared set of RING-row staging
        # rings, the slot's window copied in before and out after (``_to_stage`` / ``_from_stage``)
        self.big = Caches(
            cap,
            [torch.zeros((S * DRING, c.head_dim), dtype=BF, device=self.dev) for _ in self.w.layers],
            self._comp_pools(E),
            {s_: torch.zeros((S * DRING, 2 * c.head_dim), dtype=F32, device=self.dev)
             for s_ in c.kv_source_layer_ids if c.layer_ratios[s_] == 2},
            {s_: self._entries(E[s_], c.index_head_dim, KV_FP8, keys=True) for s_ in c.kv_source_layer_ids},
        )
        big = self.big
        self.stage_swa = [torch.zeros((RING, c.head_dim), dtype=BF, device=self.dev) for _ in self.w.layers]
        self.stage_raw = {s_: torch.zeros((RING, 2 * c.head_dim), dtype=F32, device=self.dev) for s_ in big.raw}
        self.views: list[Caches] = [None] * S        # type: ignore[list-item]
        fixed = P >= S * self.span
        for i in range(S):                               # fixed extents, or (a shared pool) all on the first rows
            self.bind(i, i * self.span if fixed else 0, self.span)   # until a stream is admitted (captures, timing)
        self.state = self.views[self.slot]

    def bind(self, slot: int, base: int, size: int, ids: list[int] | None = None) -> None:
        """Give ``slot`` the extent [base, base + size) of the arena: its views (prompt chunks, kept states) and the
        decode graphs' base table. Not during a capture. ``ids``: the tokens whose per-token caches those rows already
        hold (a resumed kept prompt), else none."""

        c = self.c
        if base % ALIGN or size % ALIGN or size <= 0 or base + size > self.pool_tokens:
            raise ValueError(f"extent [{base}, {base + size}) is not aligned inside the {self.pool_tokens}-token pool")
        if size > self.span:
            raise ValueError(f"an extent of {size} tokens: a stream needs at most {self.span}")
        self.extents[slot] = (base, size)
        self.ebase[slot] = base

        def cut(t, r):
            return t[base // r:(base + size) // r]

        self.views[slot] = Caches(self.cap, self.stage_swa,
                                  {k: cut(t, c.layer_ratios[k]) for k, t in self.big.comp.items()}, self.stage_raw,
                                  {k: cut(t, c.layer_ratios[k]) for k, t in self.big.ik.items()},
                                  list(ids or []))
        if slot == self.slot and getattr(self, "state", None) is not None:
            self.state = self.views[slot]

    # -- kept prompts inside the shared pool: their window rings in a bank, their rows copied between extents ----
    def _rings(self) -> list[torch.Tensor]:
        """Every decode ring indexed by slot * DRING + position % DRING (attention windows, compressor pairs, the
        drafter's windows)."""

        return [*self.big.swa, *self.big.raw.values(), *(self.drafter.swa_big if self.drafter is not None else [])]

    def make_bank(self, n: int) -> None:
        """Room for ``n`` kept window rings (a slot's whole DRING block of every ring each): after the drafter."""

        self.bank = [torch.empty((n * DRING, *t.shape[1:]), dtype=t.dtype, device=self.dev) for t in self._rings()]

    def bank_bytes(self, n: int) -> int:
        return n * DRING * sum(t[0].numel() * t.element_size() for t in self._rings())

    def save_window(self, slot: int, k: int) -> None:
        """Slot ``slot``'s rings into bank entry ``k``."""

        for t, b in zip(self._rings(), self.bank):
            b[k * DRING:(k + 1) * DRING].copy_(t[slot * DRING:(slot + 1) * DRING])

    def load_window(self, k: int, slot: int) -> None:
        """Bank entry ``k`` into slot ``slot``'s rings (the addresses the graphs read stay put)."""

        for t, b in zip(self._rings(), self.bank):
            t[slot * DRING:(slot + 1) * DRING].copy_(b[k * DRING:(k + 1) * DRING])

    def copy_rows(self, src: int, dst: int, n: int) -> None:
        """The per-token caches of tokens [src, src + n) of the arena to [dst, dst + n) (entries closed before
        token n, of every kv source; the ranges may overlap)."""

        from .pool import move_rows

        for s_ in self.c.kv_source_layer_ids:
            r = self.c.layer_ratios[s_]
            for t in (self.big.comp[s_], self.big.ik[s_]):
                move_rows(t, src // r, dst // r, -(-n // r))

    def _ebase(self, src: int) -> torch.Tensor:
        """Each decode row's stream's first entry of source ``src`` (in a graph: read from the base table)."""

        return self.ebase[self._sid] // self.c.layer_ratios[src]

    def _comp_pools(self, E: dict) -> dict:
        """The compressed-entry pools of every source, largest first into the display carveout when it is on
        (``TF_CARVEOUT=1``: attention gathers 512 entries a row, sparse reads that its half bandwidth suits), the
        rest (and the indexer keys, scanned whole every step) in ordinary memory."""

        from tensorfold.cuda import carveout

        c = self.c
        owner = carveout.get()
        self.carved = 0
        pools = {}
        for s_ in sorted(c.kv_source_layer_ids, key=lambda k: -E[k]):
            alloc = None
            if owner is not None:
                need = self._pool_bytes(E[s_], c.head_dim, KV_FP8)
                if need + 4096 <= owner.free:
                    alloc = owner.take
                    self.carved += need
            pools[s_] = self._entries(E[s_], c.head_dim, KV_FP8, alloc=alloc)
        return {s_: pools[s_] for s_ in c.kv_source_layer_ids}

    @staticmethod
    def _pool_bytes(n: int, dim: int, fp8: bool) -> int:
        return n * (448 + 64 * 2 + 7 * 4) if fp8 else n * dim * 2

    def _entries(self, n: int, dim: int, fp8: bool, keys: bool = False, alloc=None):
        """Per-position cache rows: bf16, or fp8 (``K.Fp8Rows``): compressed entries keep their RoPE dims bf16 and
        a scale per 64 values (DeepSeek's fp8 KV layout), indexer keys a scale per key."""

        if not fp8:
            return alloc((n, dim), BF) if alloc else torch.zeros((n, dim), dtype=BF, device=self.dev)
        if keys:
            return K.Fp8Rows(n, dim, plain=0, group=dim, device=self.dev)
        return K.Fp8Rows(n, dim, plain=self.c.qk_rope_head_dim, group=64, device=self.dev, alloc=alloc)

    def select_slot(self, slot: int) -> None:
        """Make ``slot`` the current stream (prompt chunks, single-stream decoding and kept states use it)."""

        self.slot = int(slot)
        self.state = self.views[self.slot]
        if self.drafter is not None:
            self.drafter.slot = self.slot

    # -- one forward over new rows ------------------------------------------------------------------------
    def forward(self, tokens: list[int], last_only: bool = False, raw: torch.Tensor | None = None) -> torch.Tensor:
        """Logits fp32 [R, vocab] of the new rows (``last_only``: only the last row's, [1, vocab] — what a prompt
        chunk needs); positions continue the committed ones."""

        st = self.state
        R = len(tokens)
        p0 = len(st.ids)
        if not 0 < R <= MAX_ROWS:
            raise ValueError(f"1..{MAX_ROWS} rows a call, got {R}")
        if p0 + R > self.limit:
            raise ValueError(f"context {p0 + R} beyond {self.limit} tokens")
        if p0 + R > self.extents[self.slot][1]:
            raise ValueError(f"position {p0 + R} beyond the slot's extent of {self.extents[self.slot][1]} tokens")
        if R in self.graphs:
            self.step_rows(tokens)
            return self.graph_for(R, p0 + R)["logits"]
        rows = self.engram_rows(tokens, raw)
        self._to_stage(p0)
        out = self.core(torch.tensor(tokens, device=self.dev), torch.arange(p0, p0 + R, device=self.dev), rows,
                        static=False, last_only=last_only)
        self._from_stage(p0 + R)
        return out

    def _ring_rows(self, p1: int):
        """(positions, staging rows, slot decode rows) of the window ending before position ``p1``."""

        n = min(p1, WINDOW_ROWS)
        pos = torch.arange(p1 - n, p1, device=self.dev)
        return pos % RING, self.slot * DRING + pos % DRING

    def _to_stage(self, p0: int) -> None:
        """The current slot's window (decode rings) into the staging rings, before a prompt chunk at ``p0``."""

        if p0 == 0:
            return
        st_rows, dec_rows = self._ring_rows(p0)
        for stage, dec in zip(self.stage_swa, self.big.swa):
            stage.index_copy_(0, st_rows, dec[dec_rows])
        for k, dec in self.big.raw.items():
            self.stage_raw[k].index_copy_(0, st_rows, dec[dec_rows])
        if self.drafter is not None:
            for stage, dec in zip(self.drafter.stage, self.drafter.swa_big):
                stage.index_copy_(0, st_rows, dec[dec_rows])

    def _from_stage(self, p1: int) -> None:
        """The staging rings' window back into the slot's decode rings, after a prompt chunk ending at ``p1``."""

        st_rows, dec_rows = self._ring_rows(p1)
        for stage, dec in zip(self.stage_swa, self.big.swa):
            dec.index_copy_(0, dec_rows, stage[st_rows])
        for k, dec in self.big.raw.items():
            dec.index_copy_(0, dec_rows, self.stage_raw[k][st_rows])
        if self.drafter is not None:
            for stage, dec in zip(self.drafter.stage, self.drafter.swa_big):
                dec.index_copy_(0, dec_rows, stage[st_rows])

    def read_rows(self, ids: list[int], p0: int, R: int, slot: int = 0) -> torch.Tensor:
        """Engram rows of positions p0 .. p0 + R - 1 of ``ids`` into pinned host buffer ``slot`` (uint8, host)."""

        c = self.c
        start = max(0, p0 - (c.engram_max_ngram_size - 1))        # the n-gram history of the first new row
        hashes = E.hashes(np.array(ids[start:p0 + R]), self.tmap, self.layout, c.engram_pad_token_id)[p0 - start:]
        L = len(c.engram_layer_ids)
        row = c.engram_head_dim + c.engram_head_dim // 32
        need = L * R * 3 * c.engram_n_heads
        pin = self._pinned[slot] if slot < len(self._pinned) else None
        if pin is None or pin.numel() < need * row:
            pin = torch.empty((need * row,), dtype=torch.uint8).pin_memory()
            while len(self._pinned) <= slot:
                self._pinned.append(None)
            self._pinned[slot] = pin
        out = pin[:need * row].view(L, R * 3 * c.engram_n_heads, row)
        return self.tables.gather(np.stack([hashes[:, ell, :].reshape(-1) for ell in range(L)]), out=out)

    def prefetch(self, ids: list[int], p0: int, R: int, slot: int):
        """Read a later chunk's Engram rows on a background thread (the reads release the GIL)."""

        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=1)
        return self._pool.submit(self.read_rows, list(ids), p0, R, slot)

    def engram_rows(self, tokens: list[int], raw: torch.Tensor | None = None) -> torch.Tensor:
        """Commit the tokens and read their Engram rows (or take ``raw`` read ahead): fp32 [layers, R, 24, 256]."""

        c, st = self.c, self.state
        p0 = len(st.ids)
        st.ids.extend(tokens)
        R, L = len(tokens), len(c.engram_layer_ids)
        if raw is None:
            raw = self.read_rows(st.ids, p0, R)
        raw = raw.to(self.dev, non_blocking=True).view(L, R, -1, raw.shape[-1])
        hd = c.engram_head_dim
        return E.dequant(raw[..., :hd], raw[..., hd:])

    def core(self, ids: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, *, static: bool,
             last_only: bool = False) -> torch.Tensor:
        """The device-only forward over all layers (eager prompt chunks)."""

        carry = self.part_a(ids, pos, rows[0], static=static)
        return self.part_b(carry, pos, rows[1], static=static, last_only=last_only)

    def part_a(self, ids: torch.Tensor, pos: torch.Tensor, rows1: torch.Tensor, *, static: bool) -> tuple:
        """Embedding and the layers before the second Engram layer."""

        return self.part_a1(self.part_a0(ids, pos, static=static), pos, rows1, static=static)

    def part_a0(self, ids: torch.Tensor, pos: torch.Tensor, *, static: bool) -> tuple:
        """Embedding and the layers before the first Engram layer (no table rows needed)."""

        c = self.c
        R = ids.shape[0]
        X = self.w.embed[ids][:, None, :].expand(R, c.hc_mult, c.hidden_size).contiguous()
        pre = torch.zeros((R, c.hc_mult), dtype=F32, device=self.dev)
        pre[:, 0] = 1.0
        return self.layers((X, pre, None, None, None), pos, {}, 0, c.engram_layer_ids[0], static)

    def part_a1(self, carry: tuple, pos: torch.Tensor, rows1: torch.Tensor, *, static: bool) -> tuple:
        c = self.c
        return self.layers(carry, pos, {c.engram_layer_ids[0]: rows1}, c.engram_layer_ids[0], self.split, static)

    def part_b(self, carry: tuple, pos: torch.Tensor, rows14: torch.Tensor, *, static: bool,
               last_only: bool = False) -> torch.Tensor:
        """The remaining layers and the vocabulary head."""

        c = self.c
        self.taps = []
        X, pre, f, post, comb = self.layers(carry, pos, {c.engram_layer_ids[1]: rows14}, self.split,
                                            len(self.w.layers), static)
        X = hcf.post(f, X, post, comb)
        if self.drafter is not None:
            self.drafter.context(self.taps, pos, self._sid * DRING if static else 0, static=static)
        if last_only:
            X, pre = X[-1:].contiguous(), pre[-1:].contiguous()
        h = (pre[:, :, None] * X.float()).sum(1).to(BF)
        h = K.rmsnorm(h, self.w.norm, c.rms_norm_eps)
        return self.comm.gather_last(self.w.head(h, out_dtype=F32))

    def _tap(self, X: torch.Tensor) -> torch.Tensor:
        from .dspark import VARIANT

        if "tap0" in VARIANT:
            return X[:, 0].contiguous()
        if "tapsum" in VARIANT:
            return X.float().sum(1).to(BF)
        return X.float().mean(1).to(BF)

    def layers(self, carry: tuple, pos: torch.Tensor, rows: dict, first: int, last: int, static: bool) -> tuple:
        X, pre, f, post, comb = carry
        fuse = (X.shape[0] > PROMPT_ROWS or hcf.FUSE_DECODE) and FUSE_HC   # post and the next pre in one pass
        for layer in self.w.layers[first:last]:
            fused = fuse and f is not None and layer.engram is None
            if fused:
                X, (post, comb, x, pre_a) = self.post_hc(f, X, post, comb, layer.hc_attn, pre)
            elif f is not None:
                X = hcf.post(f, X, post, comb)
            if f is not None and self.drafter is not None and layer.index in self.c.dspark_target_layer_ids:
                self.taps.append(self._tap(X))                        # V4.1 taps the entry stream of layer L
            if not fused:
                if layer.engram is not None:
                    X = self.engram(layer, X, rows[layer.index])
                post, comb, x, pre_a = self.hc(layer.hc_attn, X, pre)
            a = self.attention(layer, x, pos, static)
            if fuse:
                X, (post, comb, x, pre) = self.post_hc(a, X, post, comb, layer.hc_ffn, pre_a)
            else:
                X = hcf.post(a, X, post, comb)
                post, comb, x, pre = self.hc(layer.hc_ffn, X, pre_a)
            f = self.moe(layer, x, x.shape[0])
            if self.debug is not None:
                self.debug.append({"attn_in": x.clone(), "X": X.clone(), "f": f.clone() if torch.is_tensor(f) else f})
        self._join()                                                    # (a graph ends with every stream joined)
        return X, pre, f, post, comb

    # -- decode graphs ----------------------------------------------------------------------------------------
    def capture(self, rows: int = 1) -> None:
        """Capture the R-row decode step at the full key width and each narrower one in ``widths`` (sharing their
        input buffers). Both ranks must capture together."""

        c = self.c
        row_bytes = c.engram_head_dim + c.engram_head_dim // 32
        W, cols = self.read_split
        io = {"tok": torch.zeros((rows,), dtype=torch.long, device=self.dev),
              "pos": torch.zeros((rows,), dtype=torch.long, device=self.dev),
              "sid": torch.zeros((rows,), dtype=torch.long, device=self.dev),
              # this rank's share of each row's table entries (all of them unless the reads are split)
              "raw": [torch.zeros((rows * cols, row_bytes), dtype=torch.uint8, device=self.dev) for _ in range(2)],
              "h_raw": torch.zeros((2, rows * cols, row_bytes), dtype=torch.uint8).pin_memory(),
              "h_next": torch.zeros((rows,), dtype=torch.long).pin_memory()}
        try:
            for width in [self.cap, *self.widths]:
                self._width = width
                g = self._capture(rows, dict(io))
                g["width"] = width
                if width == self.cap:
                    self.graphs[rows] = g
                else:
                    self.narrow[(rows, width)] = g
        finally:
            self._width = self.cap
        self.graph = True

    def _graph_out(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The decode graphs' shared outputs for ``logits``' rows: fp32 [R, vocab] and their argmax (valid until the
        next decode replay), allocated outside every graph on the first (warm-up) call."""

        R, V = logits.shape
        if getattr(self, "_out", None) is None or self._out[0].shape[0] < R:
            if getattr(self, "_out", None) is not None:              # graphs captured on the old one still write it
                self._old_out = [*getattr(self, "_old_out", []), self._out]
            n = max(R, PROMPT_ROWS)
            self._out = (torch.empty((n, V), dtype=torch.float32, device=self.dev),
                         torch.empty((n,), dtype=torch.long, device=self.dev))
        return self._out[0][:R], self._out[1][:R]

    def graph_for(self, rows: int, need: int) -> dict:
        """The R-row decode graph to replay for rows reaching position ``need - 1``: the narrowest covering it."""

        for w in self.widths:
            if need <= w:
                return self.narrow.get((rows, w)) or self.graphs[rows]
        return self.graphs[rows]

    def _capture(self, rows: int, g: dict) -> dict:
        """One R-row decode step graph at the key width ``self._width``: one launch (ONE_GRAPH, waiting on pinned
        flags for each table's Engram rows) or three (embedding + layer 0 / layers 1-13 / the rest)."""

        c = self.c
        n_rows = 3 * c.engram_n_heads
        row_bytes = c.engram_head_dim + c.engram_head_dim // 32
        W, cols = self.read_split
        saved = self._save_rows(0, g["pos"])                            # capture replays write slot 0, position 0
        hd = c.engram_head_dim
        self._sid = g["sid"]                                           # the graphs read each row's slot from it

        def table(k):
            raw = g["raw"][k]
            if W > 1:                                                   # every rank's share, in rank order
                full = torch.empty((W, *raw.shape), dtype=torch.uint8, device=self.dev)
                self.comm.nccl.all_gather(raw.view(-1), full.view(-1))
                raw = full.view(W, rows, cols, row_bytes).transpose(0, 1)
            raw = raw.reshape(rows, n_rows, row_bytes)
            return E.dequant(raw[..., :hd], raw[..., hd:])

        def run_a0():
            return self.part_a0(g["tok"], g["pos"], static=True)

        def run_a1(carry):
            return self.part_a1(carry, g["pos"], table(0), static=True)

        def run_b(carry):
            logits = self.part_b(carry, g["pos"], table(1), static=True)
            out = self._graph_out(logits)
            out[0].copy_(logits)                                      # every decode graph's outputs in one place:
            out[1].copy_(logits.argmax(-1))                           # not 128 graphs' own [R, vocab] logits
            return out

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run_b(run_a1(run_a0()))
        torch.cuda.current_stream().wait_stream(side)
        if ONE_GRAPH:                                                   # the step in one launch (all paths use it)
            self._prime_flags()

            def run_one():
                carry0 = run_a0()
                K.await_rows(self._flag, self._seen, g["h_raw"][0], g["raw"][0], self._flag_err)
                carry = run_a1(carry0)
                K.await_rows(self._flag, self._seen, g["h_raw"][1], g["raw"][1], self._flag_err)
                return run_b(carry)

            g["one"] = torch.cuda.CUDAGraph()                         # (no three-graph copy: its buffers cost ~1.4 GiB
            with torch.cuda.graph(g["one"], pool=self._gpool):         # over 32 row counts)
                g["logits"], g["next"] = run_one()
        else:                                                           # three graphs, the host reads between them
            g["a0"], g["a1"], g["b"] = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g["a0"], pool=self._gpool):
                g["carry0"] = run_a0()
            pool = self._gpool if self._gpool is not None else g["a0"].pool()
            with torch.cuda.graph(g["a1"], pool=pool):
                g["carry"] = run_a1(g["carry0"])
            with torch.cuda.graph(g["b"], pool=pool):
                g["logits"], g["next"] = run_b(g["carry"])
        torch.cuda.synchronize()
        self._restore_rows(saved)
        return g

    def _prime_flags(self) -> None:
        """The one-graph step's host flag and device counter (shared by every graph), and its kernels compiled: one
        wait run eagerly against a published flag. Counters advance in step: one publish per wait."""

        if getattr(self, "_flag", None) is not None:
            return
        self._flag = torch.zeros((1,), dtype=torch.int64).pin_memory()
        self._flag_err = torch.zeros((1,), dtype=torch.int32).pin_memory()
        self._seen = torch.zeros((1,), dtype=torch.int64, device=self.dev)
        self._flag_n = 0
        scratch = torch.zeros((64,), dtype=torch.uint8).pin_memory()
        self._publish()
        K.await_rows(self._flag, self._seen, scratch, torch.empty((64,), dtype=torch.uint8, device=self.dev),
                     self._flag_err)
        torch.cuda.synchronize()

    def _publish(self) -> None:
        """Release the next wait of the one-graph step (after its rows are in the pinned buffer)."""

        from .rowread import reader

        self._flag_n += 1
        reader().publish(self._flag, self._flag_n)

    def _launch(self, g) -> None:
        """Start a decode step: the one graph (the caller then publishes each table's rows) or the first of three."""

        if "one" in g:
            g["one"].replay()
        else:
            g["a0"].replay()

    def _replay_free(self, g) -> None:
        """A whole step for timing, its waits released up front (rows: whatever the buffers hold)."""

        if "one" in g:
            self._publish()
            self._publish()
            g["one"].replay()
        else:
            g["a0"].replay(), g["a1"].replay(), g["b"].replay()

    def _check_flags(self) -> None:
        if getattr(self, "_flag_err", None) is not None and int(self._flag_err[0]):
            raise RuntimeError("decode graph waited for Engram rows that were never published")

    def agree(self, k: int) -> int:
        """Rank 0's draft count for this round, on every rank (a tiny all-gather; ranks must replay the same graphs)."""

        if self.comm.world == 1:
            return k
        return gather_ints(torch, lambda a, b: self.comm.nccl.all_gather(a, b), [k], self.comm.world)[0][0]

    def round_costs(self) -> list[float]:
        """Milliseconds of a round verifying k = 0 .. n drafts (graph replays, measured once, the slower rank's)."""

        if getattr(self, "_round_costs", None) is not None:
            return self._round_costs
        n = min(max(self.graphs), (self.drafter.N if self.drafter is not None else 0) + 1) - 1
        P0 = len(self.state.ids)
        saved = self._save_rows(self.slot, torch.arange(max(P0 - n - 1, 0), P0 + n + 1),
                                torch.arange(P0, P0 + n + 2))

        def replay_ms(fn, reps=3) -> float:
            fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return 1e3 * (time.perf_counter() - t) / reps

        ids = self.state.ids
        P = len(ids)

        def verify(R):
            g = self.graph_for(R, max(P, R))
            # distinct real tokens (the request's latest), not the capture's zeros: identical rows route to the same
            # experts and would make an R-row verify look nearly as cheap as one row
            tail = (list(ids[-R:]) if len(ids) >= R else list(ids) + list(range(1000, 1000 + R - len(ids))))
            g["tok"].copy_(torch.tensor(tail))
            g["pos"].copy_(torch.arange(max(P - R, 0), max(P - R, 0) + R))
            g["sid"].fill_(self.slot)
            return lambda: self._replay_free(g)

        draft = 0.0
        if self.drafter is not None and self.drafter.graph:
            dsp = self.drafter                                          # at this stream's position and ring
            dsp.g_anchor.fill_(ids[-1] if ids else 1000)
            dsp.g_P.fill_(P)
            dsp.g_base.fill_(self.slot * DRING)
            draft = replay_ms(dsp.graph.replay)
        mine = [replay_ms(verify(1))] + [draft + replay_ms(verify(k + 1)) for k in range(1, n + 1)]
        self._restore_rows(saved)
        both = gather_ints(torch, lambda a, b: self.comm.nccl.all_gather(a, b), [int(1e3 * c) for c in mine],
                           self.comm.world) if self.comm.world > 1 else [[int(1e3 * c) for c in mine]]
        self._round_costs = [max(row[k] for row in both) / 1e3 for k in range(n + 1)]
        return self._round_costs

    def _save_rows(self, slot: int, pos: torch.Tensor, draft: torch.Tensor | None = None) -> list:
        """The cache rows a timing replay at positions ``pos`` of ``slot`` writes (window and compressor ring rows,
        compressed entries and indexer keys with the spare entry, the drafter's rows at ``pos`` and ``draft``), so
        they can be put back: kilobytes, where cloning every slot's caches would take gigabytes."""

        c = self.c
        pos = pos.to(self.dev).long()
        ring = torch.unique(slot * DRING + pos % DRING)
        out = [(t, ring, t[ring].clone()) for t in self.big.swa]
        out += [(t, ring, t[ring].clone()) for t in self.big.raw.values()]
        base = self.extents[slot][0]
        for s_ in c.kv_source_layer_ids:
            r = c.layer_ratios[s_]
            ent = torch.unique(torch.cat([pos // r + base // r, torch.tensor([self.trash[s_]], device=self.dev)]))
            for t in (self.big.comp[s_], self.big.ik[s_]):
                out.append((t, ent, t.take(ent) if isinstance(t, K.Fp8Rows) else t[ent].clone()))
        if self.drafter is not None:
            dpos = pos if draft is None else torch.cat([pos, draft.to(self.dev).long()])
            dring = torch.unique(slot * DRING + dpos.clamp(min=0) % DRING)
            out += [(t, dring, t[dring].clone()) for t in self.drafter.swa_big]
        return out

    @staticmethod
    def _restore_rows(saved: list) -> None:
        for t, idx, vals in saved:
            if isinstance(t, K.Fp8Rows):
                t.put_rows(idx, vals)
            else:
                t.index_copy_(0, idx, vals)

    def _save_caches(self):
        st = self.big
        extra = [t.clone() for t in self.drafter.swa_big] if self.drafter is not None else []
        return ([t.clone() for t in st.swa], {k: v.clone() for k, v in st.comp.items()},
                {k: v.clone() for k, v in st.raw.items()}, extra, {k: v.clone() for k, v in st.ik.items()})

    def _restore_caches(self, saved) -> None:
        st = self.big
        for dst, src in zip(st.swa, saved[0]):
            dst.copy_(src)
        for k, v in saved[1].items():
            st.comp[k].copy_(v)
        for k, v in saved[2].items():
            st.raw[k].copy_(v)
        if self.drafter is not None:
            for dst, src in zip(self.drafter.swa_big, saved[3]):
                dst.copy_(src)
        for k, v in saved[4].items():
            st.ik[k].copy_(v)

    def step_multi(self, rows: list[tuple[int, int]]) -> tuple[torch.Tensor, list[int]]:
        """One decode step over rows of several streams: ``rows`` = (slot, token) in any order, each slot's rows at
        its next positions in order. Commits the tokens to their slots; returns the logits [R, V] (the graph's
        buffer) and each row's argmax. A row's values do not depend on the other rows (row-invariant kernels)."""

        c = self.c
        R = len(rows)
        mp = getattr(self, "_mprof", None)
        t0 = time.perf_counter()
        n = c.engram_max_ngram_size
        by_slot: dict[int, list[int]] = {}
        for slot, tok in rows:
            by_slot.setdefault(slot, []).append(tok)
        hashes: dict[int, np.ndarray] = {}
        first: dict[int, int] = {}
        for slot, toks in by_slot.items():
            ids = self.views[slot].ids
            p0 = len(ids)
            if p0 + len(toks) > self.limit:
                raise ValueError(f"stream in slot {slot}: context {p0 + len(toks)} beyond {self.limit} tokens")
            first[slot] = p0
            if p0 + len(toks) > self.extents[slot][1]:
                raise ValueError(f"stream in slot {slot}: position {p0 + len(toks)} beyond its extent's "
                                 f"{self.extents[slot][1]} tokens")
            ids.extend(toks)
            start = max(0, p0 - (n - 1))
            hashes[slot] = E.hashes(np.array(ids[start:]), self.tmap, self.layout, c.engram_pad_token_id)[-len(toks):]
        seen = {slot: 0 for slot in by_slot}
        pos, sid, h = [], [], []
        for slot, _ in rows:
            j = seen[slot]
            seen[slot] += 1
            pos.append(first[slot] + j)
            sid.append(slot)
            h.append(hashes[slot][j])
        h = np.stack(h)                                                 # [R, 2, 24]
        g = self.graph_for(R, max(pos) + 1)
        threads = min(64, max(STEP_READ_THREADS, 4 * R))               # many streams' rows: more reads in flight
        t1 = time.perf_counter()
        g["tok"].copy_(torch.tensor([t for _, t in rows]), non_blocking=True)
        g["pos"].copy_(torch.tensor(pos), non_blocking=True)
        g["sid"].copy_(torch.tensor(sid), non_blocking=True)
        one = "one" in g
        self._launch(g)
        mine = self.read_cols
        self.tables.gather(h[:, 0, mine].reshape(1, -1), out=g["h_raw"][:1], layers=[0], threads=threads)
        t2 = time.perf_counter()
        if one:
            self._publish()
        else:
            g["raw"][0].copy_(g["h_raw"][0].view_as(g["raw"][0]), non_blocking=True)
            g["a1"].replay()
        self.tables.gather(h[:, 1, mine].reshape(1, -1), out=g["h_raw"][1:], layers=[1], threads=threads)
        t3 = time.perf_counter()
        if one:
            self._publish()
        else:
            g["raw"][1].copy_(g["h_raw"][1].view_as(g["raw"][1]), non_blocking=True)
            g["b"].replay()
        g["h_next"].copy_(g["next"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        self._check_flags()
        if mp is not None:
            for k, v in (("hash", t1 - t0), ("gather0", t2 - t1), ("gather1", t3 - t2), ("wait", time.perf_counter() - t3)):
                mp[k] = mp.get(k, 0.0) + v
            mp["n"] = mp.get("n", 0) + 1
        return g["logits"], g["h_next"][:R].tolist()

    def step(self, token: int, sampling=None) -> int:
        return self.step_rows([token], sampling)[0]

    def step_rows(self, tokens: list[int], sampling=None, constraint=None, window=None) -> list[int]:
        """R rows through the captured graphs; returns the target's token at each row: the argmax, or with
        ``sampling`` the position-keyed sample (so a verify window's rows equal the serial path's)."""

        c, st = self.c, self.state
        R = len(tokens)
        p0 = len(st.ids)
        if p0 + R > self.limit:
            raise ValueError(f"context {p0 + R} beyond {self.limit} tokens")
        if p0 + R > self.extents[self.slot][1]:
            raise ValueError(f"position {p0 + R} beyond the slot's extent of {self.extents[self.slot][1]} tokens")
        g = self.graph_for(R, p0 + R)
        st.ids.extend(tokens)
        start = max(0, p0 - (c.engram_max_ngram_size - 1))
        rp = self._rp
        rp and rp.mark("v.enter")
        g["tok"].copy_(torch.tensor(tokens), non_blocking=True)
        g["pos"].copy_(torch.arange(p0, p0 + R), non_blocking=True)
        g["sid"].fill_(self.slot)
        rp and rp.event("v.gpu0")
        one = "one" in g
        self._launch(g)                                                 # layer 0 needs no table rows
        rp and rp.mark("v.a0")
        h = E.hashes(np.array(st.ids[start:]), self.tmap, self.layout, c.engram_pad_token_id)[-R:]   # [R, 2, 24]
        rp and rp.mark("v.hash")
        if not SKIP_READS:                                              # (timing experiments only)
            self.tables.gather(h[:, 0, self.read_cols].reshape(1, -1), out=g["h_raw"][:1], layers=[0],  # ~ layer 0
                               threads=STEP_READ_THREADS)
        rp and rp.mark("v.gather0")
        if one:
            self._publish()
        else:
            g["raw"][0].copy_(g["h_raw"][0].view_as(g["raw"][0]), non_blocking=True)
            g["a1"].replay()
        rp and rp.mark("v.a1")
        if not SKIP_READS:
            self.tables.gather(h[:, 1, self.read_cols].reshape(1, -1), out=g["h_raw"][1:], layers=[1],  # ~ 1-13
                               threads=STEP_READ_THREADS)
        rp and rp.mark("v.gather1")
        if one:
            self._publish()
        else:
            g["raw"][1].copy_(g["h_raw"][1].view_as(g["raw"][1]), non_blocking=True)
            g["b"].replay()
        rp and rp.mark("v.b")
        rp and rp.event("v.gpu1")
        if constraint is not None:                                      # the grammar's rows masked, then chosen
            logits = constraint.mask(g["logits"][:R].float().clone(), window)
            return sample_rows(logits, [p0 + 1 + j for j in range(R)], sampling)
        if sampling is not None and sampling.temperature > 0:
            return sample_rows(g["logits"], [p0 + 1 + j for j in range(R)], sampling)
        g["h_next"].copy_(g["next"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        self._check_flags()
        return g["h_next"].tolist()

    # -- pieces ----------------------------------------------------------------------------------------------
    def post_hc(self, b, X: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, w: HCW, pre_in: torch.Tensor):
        c = self.c
        return hcf.post_pre(b, X, post, comb, w.fn, w.base, w.scale, pre_in, w.norm, self.hcbuf, c.rms_norm_eps,
                            c.hc_eps, c.hc_sinkhorn_iters)

    def hc(self, w: HCW, X: torch.Tensor, pre_in: torch.Tensor):
        c = self.c
        return hcf.pre(X, w.fn, w.base, w.scale, pre_in, w.norm, self.hcbuf, c.rms_norm_eps, c.hc_eps,
                       c.hc_sinkhorn_iters)

    def attention(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor, static: bool) -> torch.Tensor:
        self._join()
        c, st, a = self.c, self.state, layer.attn
        L, R = layer.index, x.shape[0]
        Dh, W = c.head_dim, c.sliding_window
        cos, sin = self.tables_rope[a.ratio]
        eps = c.rms_norm_eps
        H = a.wq_b.n // Dh

        def q_branch():
            qr = K.rmsnorm(a.wq_a(x), a.q_norm, eps)
            return qr, K.rope(a.wq_b(qr).view(R, H, Dh), pos, cos, sin)          # bf16, this rank's heads

        def kv_branch():
            kv = K.rmsnorm(a.wkv(x), a.kv_norm, eps)
            if static:                                                  # each row into its own stream's ring
                self.big.swa[L].index_copy_(0, self._sid * DRING + pos % DRING, K.rope(kv, pos, cos, sin))
            else:
                st.swa[L].index_copy_(0, pos % RING, K.rope(kv, pos, cos, sin))

        def comp_branch():
            if a.ratio > 0 and a.compressor is not None:
                self.compress(layer, x, pos, static)

        if R <= PROMPT_ROWS:                                            # independent: q, window KV, compressor
            (qr, q), _, _ = self.par(q_branch, kv_branch, comp_branch)
        else:
            qr, q = q_branch()
            kv_branch()
            comp_branch()
        comp = idx = None
        if a.ratio > 0:
            if a.indexer is not None:
                self.topk[L] = self.select(layer, qr, x, pos, static)
            src = max(s for s in c.kv_source_layer_ids if s <= L)
            comp = st.comp[src]
            idx = self.topk[max(s for s in c.index_source_layer_ids if s <= L)]
            if static:                                                  # entries of the row's stream
                comp = self.big.comp[src]
                idx = torch.where(idx >= 0, idx + self._ebase(src).int()[:, None], idx)
        if R <= PROMPT_ROWS and L in self._pf_woa:
            self._prefetch(self._pf_woa[L], programs=PF_PROGRAMS)  # wo_a streams in while the attention core runs
        if static:
            o = K.mqa(q, comp, idx, self.big.swa[L], pos, a.sink, W, self.attnbuf, Dh ** -0.5, cos, sin,
                      sbase=self._sid * DRING, ring=DRING)
        else:
            o = K.mqa(q, comp, idx, st.swa[L], pos, a.sink, W, self.attnbuf, Dh ** -0.5, cos, sin)   # inverse-rotated
        groups = len(a.wo_a)
        o = o.view(R, groups, (H // groups) * Dh)
        if R <= PROMPT_ROWS and a.wo_a_grouped is not None:
            self._join()
            z = a.wo_a_grouped(o.reshape(R, -1))                        # every group in one launch
        elif R <= PROMPT_ROWS:
            z = torch.cat(self.par(*[lambda g=g, wo=wo: wo(o[:, g].contiguous()) for g, wo in enumerate(a.wo_a)]), dim=1)
        else:
            z = torch.cat([wo(o[:, g]) for g, wo in enumerate(a.wo_a)], dim=1)
        pf = self._pf_moe.get(L) if R <= PROMPT_ROWS else None        # the MoE's router + shared expert, meanwhile
        return self.comm.partials_rows(lambda r0, r1: a.wo_b(z[r0:r1], out_dtype=F32 if R <= PROMPT_ROWS else BF), R,
                                       during=(lambda: self._prefetch(pf)) if pf is not None else None)

    def select(self, layer: LayerW, qr: torch.Tensor, x: torch.Tensor, pos: torch.Tensor,
               static: bool = True) -> torch.Tensor:
        """This index source's top-k compressed entries for each row (shared by the layers after it)."""

        c, a = self.c, layer.attn
        ix = a.indexer
        R = x.shape[0]
        cos, sin = self.tables_rope[a.ratio]
        iq = K.rope(ix.wq_b(qr).view(R, c.index_n_heads, c.index_head_dim), pos, cos, sin)
        wts = K.router_logits(x, ix.weights_proj) * (c.index_head_dim ** -0.5 * c.index_n_heads ** -0.5)
        L = layer.index
        src = max(s for s in c.kv_source_layer_ids if s <= L)
        keys = self.state.ik[src]
        if static:                                                      # each row scores its own stream's keys
            scores = K.index_scores(iq, wts, self.big.ik[src], pos, a.ratio, kbase=self._ebase(src),
                                    n_keys=min(self.entries[src], self._width // a.ratio + 1))
            if L == c.candidate_source_layer_id:
                self.candidates = K.candidate_blocks(scores, pos, a.ratio, c.candidate_block_size,
                                                     c.candidate_topk_blocks)
            elif L > c.candidate_source_layer_id:
                scores = K.mask_to_blocks(scores, self.candidates, c.candidate_block_size)
            return K.top_entries(scores, c.index_topk)
        if not static:                                                  # prompt chunks: only the visible prefix
            keys = keys[:max(1, (int(pos[-1]) + 1) // a.ratio)]
        if R > PROMPT_ROWS and BLOCKED_SELECT:                          # no [rows, keys] matrix: memory O(rows)
            src = L == c.candidate_source_layer_id
            idx, cand = K.index_select_blocked(
                iq, wts, keys, pos, a.ratio, c.index_topk, block=c.candidate_block_size,
                blocks=self.candidates if L > c.candidate_source_layer_id else None,
                candidates=c.candidate_topk_blocks if src else 0)
            if src:
                self.candidates = cand
            return idx
        scores = K.index_scores(iq, wts, keys, pos, a.ratio)
        if L == c.candidate_source_layer_id:                            # publishes blocks for the later indexers
            self.candidates = K.candidate_blocks(scores, pos, a.ratio, c.candidate_block_size,
                                                 c.candidate_topk_blocks)
        elif L > c.candidate_source_layer_id:
            scores = K.mask_to_blocks(scores, self.candidates, c.candidate_block_size)
        return K.top_entries(scores, c.index_topk)

    def compress(self, layer: LayerW, x: torch.Tensor, pos: torch.Tensor, static: bool) -> None:
        c, st, a = self.c, self.state, layer.attn
        L, r = layer.index, a.ratio
        cos, sin = self.tables_rope[r]
        cw = a.compressor
        kv = cw.wkv(x, out_dtype=F32)
        if r == 1:
            latent = K.rmsnorm(kv, cw.norm, c.rms_norm_eps)
            ends, start, slot = pos, pos, pos
        else:
            gate = cw.wgate(x, out_dtype=F32)
            raw = self.big.raw[L] if static else st.raw[L]
            roff = self._sid * DRING if static else 0                   # the row's stream's ring
            rs = DRING if static else RING
            raw.index_copy_(0, roff + pos % rs, torch.cat([kv, gate], dim=1))
            if static:                                                  # write the group only when pos closes it
                ends = pos
            else:
                closing = [int(p) for p in pos.tolist() if (p + 1) % 2 == 0]
                if not closing:
                    return
                ends = torch.tensor(closing, device=self.dev)
            pair = torch.stack([raw[roff + (ends - 1).clamp(min=0) % rs], raw[roff + ends % rs]], dim=1)
            wts = torch.softmax(pair[..., c.head_dim:], dim=1)
            latent = K.rmsnorm((wts * pair[..., :c.head_dim]).sum(1), cw.norm, c.rms_norm_eps)
            start, slot = (ends // 2) * 2, ends // 2
        comp_t, ik_t = st.comp[L], st.ik[L]
        if static:                                                      # the row's stream's entries
            slot = slot + self._ebase(L)
            if a.ratio == 2:                                            # rows that close no group: the trash row
                slot = torch.where((ends + 1) % 2 == 0, slot, torch.full_like(slot, self.trash[L]))
            comp_t, ik_t = self.big.comp[L], self.big.ik[L]
        comp_t.index_copy_(0, slot, K.rope(latent, start, cos, sin))
        ix = a.indexer
        if ix is not None and ix.wk is not None:                        # this source's indexer keys
            key = K.rmsnorm(ix.wk(latent), ix.k_norm, c.rms_norm_eps)
            ik_t.index_copy_(0, slot, K.rope(key, start, cos, sin))

    def _prefetch(self, table, programs: int = 48) -> None:
        """Warm L2 with ``table``'s weights on the prefetch stream (joined by ``_join``)."""

        main, side = torch.cuda.current_stream(), self._pf_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            K.l2_prefetch(table, programs)
        self._pf_live = True

    def _join(self) -> None:
        if getattr(self, "_pf_live", False):
            torch.cuda.current_stream().wait_stream(self._pf_stream)
            self._pf_live = False

    def moe(self, layer: LayerW, x: torch.Tensor, R: int, top_k: int | None = None, scratch=None) -> torch.Tensor:
        self._join()
        c, m = self.c, layer.moe
        limit = c.swiglu_limit
        if scratch is None:
            scratch = self.scratch[layer.index] if R <= PROMPT_ROWS else self.scratch_prompt

        def route():                                                # a row's bits never depend on the row count
            return K.route(K.router_logits(x, m.gate), m.bias, top_k or c.num_experts_per_tok, c.routed_scaling_factor)

        def shared_act() -> torch.Tensor:
            g = m.shared[0](x, out_dtype=F32)
            u = m.shared[1](x, out_dtype=F32)
            return (torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)).to(BF)

        def routed_rows():
            pick, w = route()
            return ex3.routed(x.contiguous(), pick, w, m.experts, scratch, None, R, limit=limit)

        if R <= PROMPT_ROWS and m.shared_id is not None:
            # the shared expert as one more slot of every row (weight 1, added last), in the same grouped call
            pick, w = route()
            sid = m.experts.count if SKIP_SHARED else m.shared_id                 # (timing: a skipped slot)
            pick = torch.cat([pick, torch.full((R, 1), sid, dtype=pick.dtype, device=pick.device)], dim=1)
            w = torch.cat([w, torch.ones((R, 1), dtype=w.dtype, device=w.device)], dim=1)
            return self.comm.partials(ex3.routed(x.contiguous(), pick, w, m.experts, scratch, None, R, limit=limit))
        if R <= PROMPT_ROWS:                                        # the whole shared expert beside the routed ones
            if SKIP_SHARED:
                return self.comm.partials(routed_rows())
            routed, shared = self.par(routed_rows, lambda: m.shared[2](shared_act(), out_dtype=F32))
            main_layer = layer.index < len(self.scratch) and scratch is self.scratch[layer.index]   # (not drafter)
            pf = self._pf_attn.get(layer.index) if main_layer else None
            return self.comm.partials(routed + shared, during=(lambda: self._prefetch(pf)) if pf is not None else None)
        pick, w = route()
        routed = routed_prompt(x.contiguous(), pick, w, m.experts, scratch, R, limit)
        act = shared_act()

        def block(r0: int, r1: int) -> torch.Tensor:          # routed + shared, added in the GEMM, sent as bf16
            return m.shared[2].prompt(act[r0:r1], out_dtype=BF, res=routed[r0:r1])

        return self.comm.partials_rows(block, R)

    def engram(self, layer: LayerW, X: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        R = X.shape[0]
        g = layer.engram
        kv = self.comm.gather_last(g.wkv(rows.to(BF).reshape(R, -1)))    # each rank projects half the columns
        return K.engram_gate(X, kv, g.q, g.k, self.c.rms_norm_eps)

    # -- requests ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt: list[int], chunk: int = MAX_ROWS) -> torch.Tensor:
        """Chunked prompt, each chunk's Engram rows read while the previous chunk runs; the last row's logits."""

        starts = list(range(len(self.state.ids), len(self.state.ids) + len(prompt), chunk))
        base = len(self.state.ids)
        ids = list(self.state.ids) + list(prompt)
        ahead = self.prefetch(ids, starts[0], min(chunk, base + len(prompt) - starts[0]), 0)
        logits = None
        for n, p0 in enumerate(starts):
            R = min(chunk, base + len(prompt) - p0)
            raw = ahead.result()
            if n + 1 < len(starts):
                p1 = starts[n + 1]
                ahead = self.prefetch(ids, p1, min(chunk, base + len(prompt) - p1), (n + 1) % 2)
            logits = self.forward(ids[p0:p0 + R], last_only=True, raw=raw)
        return logits

    @torch.no_grad()
    # -- kept prompt states (tensorfold.cuda.kv_pool) ----------------------------------------------------------
    def _window_slots(self, n: int) -> torch.Tensor:
        """The current slot's decode-ring rows of the window ending before position ``n``."""

        win = min(n, WINDOW_ROWS)                               # the window rows and the compressor's open pair
        return self.slot * DRING + torch.arange(n - win, n, device=self.dev) % DRING

    def save_prefix(self, n: int) -> tuple[dict, int]:
        """The state after the first ``n`` committed tokens, copied out: per-position caches up to n, the rings' last
        window (attention, compressor, drafter). Restored, the next position on is rewritten before it is read."""

        st, c = self.state, self.c
        slots = self._window_slots(n)
        snap = {"n": n,
                "comp": {s_: t[:n // c.layer_ratios[s_] + 1].clone() for s_, t in st.comp.items()},
                "ik": {s_: t[:n // c.layer_ratios[s_] + 1].clone() for s_, t in st.ik.items()},
                "swa": [t[slots] for t in self.big.swa],
                "raw": {s_: t[slots] for s_, t in self.big.raw.items()},
                "dswa": [t[slots] for t in self.drafter.swa_big] if self.drafter is not None else []}
        tensors = [*snap["comp"].values(), *snap["ik"].values(), *snap["swa"], *snap["raw"].values(), *snap["dswa"]]
        return snap, sum(K.cache_nbytes(t) for t in tensors)

    def prefix_bytes(self, n: int) -> int:
        """What ``save_prefix(n)`` copies (computed without copying)."""

        c = self.c
        entries = sum((n // c.layer_ratios[s_] + 1) * (K.cache_nbytes(self.big.comp[s_][:1])
                                                         + K.cache_nbytes(self.big.ik[s_][:1]))
                      for s_ in c.kv_source_layer_ids)
        win = min(n, WINDOW_ROWS)
        row = (sum(t.shape[1] * t.element_size() for t in self.big.swa)
               + sum(t.shape[1] * t.element_size() for t in self.big.raw.values())
               + (sum(t.shape[1] * t.element_size() for t in self.drafter.swa_big) if self.drafter else 0))
        return int(entries + win * row)

    def keep_prompt(self, prompt: list[int]) -> None:
        """Keep this prompt's state in the pool (one token short, so a retry resumes too) when it would be kept:
        checked before copying, as a long prompt's state can be gigabytes."""

        ids = list(prompt[:-1])
        if self.pool is not None and self.pool.wants(ids, self.prefix_bytes(len(ids))):
            self.pool.add(ids, lambda: self.save_prefix(len(ids)))

    def load_prefix(self, snap: dict, ids: list[int]) -> None:
        """Copy a saved state into the live buffers (same addresses: the captured graphs keep reading them)."""

        st = self.state
        n = snap["n"]
        slots = self._window_slots(n)
        for s_, t in snap["comp"].items():
            st.comp[s_][:t.shape[0]].copy_(t)
        for s_, t in snap["ik"].items():
            st.ik[s_][:t.shape[0]].copy_(t)
        for live, saved in zip(self.big.swa, snap["swa"]):
            live.index_copy_(0, slots, saved)
        for s_, t in snap["raw"].items():
            self.big.raw[s_].index_copy_(0, slots, t)
        if self.drafter is not None:
            for live, saved in zip(self.drafter.swa_big, snap["dswa"]):
                live.index_copy_(0, slots, saved)
        st.ids[:] = list(ids[:n])

    def reusable(self, prompt: list[int]) -> int:
        """How many leading tokens of ``prompt`` the live caches already hold (0: start fresh). The rest is prefilled
        over them: every position from there on is rewritten before it is read, as long as the previous request
        went less than a ring past it (the window and compressor rings are addressed by position modulo RING)."""

        st = getattr(self, "state", None)
        if st is None or not REUSE:
            return 0
        ids, n = st.ids, min(len(st.ids), len(prompt) - 1)        # at least one row to prefill: its logits
        L = 0
        while L < n and ids[L] == prompt[L]:
            L += 1
        if L < REUSE_MIN or len(ids) - L > DRING - WINDOW_ROWS:     # decode went a ring past the window
            return 0
        return L

    def generate(self, prompt: list[int], max_tokens: int, *, chunk: int = MAX_ROWS, on_token=None,
                 sampling=None, on_tokens=None, draft: bool = True, stop_eos: bool = True, constraint=None,
                 reuse: bool = True) -> dict:
        """Decode after a chunked prefill (greedy, or position-keyed sampling); returns tokens and timings.

        ``on_tokens(new) -> bool`` (rank 0) gets each round's tokens; True stops after that round. Both ranks must
        call this together: rank 0's stop rides on the per-round agreement, every other stop (EOS, max_tokens, the
        context limit) follows from the tokens, which are the same on both ranks. ``draft=False``: serial decoding,
        the reference drafted replies equal."""

        t0 = time.perf_counter()
        cached = self.reusable(prompt) if reuse else 0
        kept = self.pool.match(prompt) if reuse and self.pool is not None and self.state is not None else None
        if kept is not None and len(kept.ids) > cached:            # a stored prompt state beats the live caches
            self.load_prefix(kept.snapshot, kept.ids)
            cached = len(kept.ids)
        elif cached:
            del self.state.ids[cached:]                            # positions from here on are rewritten
        else:
            self.reset()
        logits = None
        logits = self.prefill(prompt[cached:], chunk)
        if reuse:                                                  # this prompt's state, before decoding moves on
            self.keep_prompt(prompt)
        dsp = self.drafter if draft and self.drafter is not None and self.drafter.graph is not None else None
        if dsp is not None and self.adaptive:
            self.round_costs()                                     # once an engine, at a real context (not timed)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = []
        first = logits[-1:]
        if constraint is not None:                                 # the first reply token under the grammar
            first = constraint.mask(first.float().clone())
        nxt = sample_rows(first, [len(prompt)], sampling)[0]
        rounds = accepted = 0
        t_draft = t_verify = 0.0
        eos = self.c.eos_token_id
        stop = False
        if dsp is not None:
            from .dspark import DraftPolicy

            n = min(max(self.graphs), dsp.N + 1) - 1                    # verify windows: 1 .. n + 1 rows
            policy = DraftPolicy(n, self.round_costs()) if self.adaptive else _Fixed(n)
            ks = [0] * (n + 1)
            # copy drafts (TF_COPY_DRAFTS=1): a reply repeating its context drafts the continuation instead of DSpark;
            # windows up to the longest captured verify graph (both ranks compute the same proposal)
            from tensorfold.cuda.copy_drafts import CopyDrafts, CopySettings

            settings = CopySettings.from_env(max(self.graphs) - 1) if constraint is None else None
            copy = CopyDrafts(self.state.ids, settings) if settings is not None else None
            copy_rounds = copy_accepted = 0

        def emit(tokens: list[int]) -> bool:
            """Append a round's tokens (cut at EOS / max_tokens); True when the reply is complete."""

            nonlocal stop
            done = False
            room = max_tokens - len(out)
            if len(tokens) >= room:
                tokens, done = tokens[:room], True
            if stop_eos and eos in tokens:
                tokens, done = tokens[:tokens.index(eos) + 1], True
            out.extend(tokens)
            if on_token:
                for tok in tokens:
                    on_token(tok)
            if on_tokens is not None and tokens and on_tokens(tokens):
                stop = True
            return done

        while True:
            rows_left = self.limit - len(self.state.ids)
            if rows_left < 1:
                break
            if dsp is None:
                if self.agree(-1 if stop else 0) < 0:                  # rank 0's client stop, every step
                    break
                if emit([nxt]):
                    break
                if rows_left < 2:
                    break
                if constraint is not None:
                    constraint.advance([nxt])                      # (emitted above)
                    if constraint.finished:
                        break
                    nxt = self.step_rows([nxt], sampling, constraint=constraint,
                                         window=constraint.window([nxt], [-1]))[0]
                elif self.graph is not None:
                    nxt = self.step(nxt, sampling)
                else:
                    nxt = sample_rows(self.forward([nxt])[-1:], [len(self.state.ids)], sampling)[0]
                continue
            P = len(self.state.ids)
            rp = self._rp
            rp and rp.start()
            proposal: list[int] = []
            if copy is not None:
                copy.truncate(P)
                copy.extend([nxt])
                proposal = copy.propose(rows_left - 1)
            code = 100 + len(proposal) if proposal else min(policy.choose(), rows_left - 1)
            k = self.agree(-1 if stop else code)
            if k < 0:
                break
            copied = k >= 100                                       # rank 0's choice: a copy round of k - 100 rows
            if copied:
                k -= 100
            rp and rp.mark("agree")
            ta = time.perf_counter()
            window = None
            if constraint is not None:
                constraint.advance([nxt])                          # chosen last round under the grammar's mask
                if constraint.finished:                            # nxt was the grammar's stop token: emit it
                    emit([nxt])
                    break
            if k == 0:                                             # drafting does not pay here: one plain row
                drafts = []
                if constraint is not None:
                    window = constraint.window([nxt], [-1])
                    target = self.step_rows([nxt], sampling, constraint=constraint, window=window)
                else:
                    target = [self.step(nxt, sampling)]
                tb = ta
            else:
                rp and rp.event("d.gpu0")
                drafts = proposal[:k] if copied else dsp.propose(nxt, P)[:k]
                rp and rp.event("d.gpu1")
                rp and rp.mark("propose")
                if constraint is not None:                         # only the drafts the grammar can take
                    window = constraint.window([nxt, *drafts], list(range(-1, len(drafts))))
                    drafts = window.tokens[1:]
                tb = time.perf_counter()
                target = self.step_rows([nxt, *drafts], sampling, constraint=constraint, window=window)
                rp and rp.mark("verify.done")
            t_draft += tb - ta
            t_verify += time.perf_counter() - tb
            m = 0
            while m < len(drafts) and drafts[m] == target[m]:
                m += 1
            del self.state.ids[P + 1 + m:]                         # rejected rows: overwritten by later positions
            if copied:                                             # the DSpark policy learns from its own rounds
                copy_rounds += 1
                copy_accepted += m
            else:
                ks[k] += 1
                policy.update(k, m, 1e3 * (time.perf_counter() - ta))
            if copy is not None:
                copy.extend(drafts[:m])
            rp and rp.mark("tail")
            rp and rp.end(k)
            rounds += 1
            accepted += m
            if constraint is not None and m:
                constraint.advance(drafts[:m])                     # accepted drafts: chosen under the mask
            if emit([nxt, *drafts[:m]]):
                break
            nxt = target[m]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if self._rp:
            self._rp.report()
        res = {"tokens": out, "cached": cached, "prefill_s": t1 - t0, "decode_s": t2 - t1,
               "prefill_tps": len(prompt) / (t1 - t0), "decode_tps": len(out) / max(t2 - t1, 1e-9)}
        if dsp is not None:
            res.update(rounds=rounds, accepted_per_round=accepted / max(rounds, 1),
                       tokens_per_round=len(out) / max(rounds, 1), draft_ms=1e3 * t_draft / max(rounds, 1),
                       verify_ms=1e3 * t_verify / max(rounds, 1), k_histogram=ks)
            if copy is not None:
                res.update(copy_rounds=copy_rounds, copy_accepted=copy_accepted)
        return res
