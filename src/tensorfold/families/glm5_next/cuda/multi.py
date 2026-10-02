"""GLM-5.3-Flash's concurrent requests (--parallel N) on two ranks: every stream commits exactly its own rows, so each
reply is the one its request gets served alone with ``"draft": false``.

Layout: the per-token caches are one pool (``pool.py``, ``forward.Caches``); a stream, or a kept prompt, owns an
extent of it aligned to 2,048 tokens, and the single-stream code (prompt chunks, verify windows, commits, snapshots)
runs on a ``forward.State`` viewing that extent and the stream's slot (``forward.Slots``: KDA states, conv windows,
window scratch, device positions; a DFlash2 context, ``dflash2_multi.DraftContext``; host side its sampler and
grammar).

Iterations (``GlmScheduler``, one worker thread on rank 0): admissions, then either one prompt chunk of the oldest
filling stream (FIFO, at most TF_GLM_FILL_ROWS rows while others decode, the engine's prompt chunk alone) or one
decode round, alternating by TF_GLM_FILL_SHARE; a prompt chunk never shares a forward with decode rows. A round: each
decoding stream (admission order) proposes drafts under its own policy (DFlash2: every drafting stream's block in one
batched pass, ``dflash2_multi``; MTP: the stream's own chain on the MTP head), the combined window is capped at
``MAX_WINDOW`` rows (the longest tails trimmed), one verify over every stream's rows (``forward_streams``:
``verify.BatchedVerify``, one forward over every stream's rows through the segmented KDA and DSA kernels, or
``verify.SerialVerify``, the reference running each window as its own forward; TF_GLM_MULTI_VERIFY), the samplers'
candidates in one all-gather, each stream accepts and commits its own.

Rank 0 decides everything that depends on time, clients or placement; rank 1 applies it. Each iteration rank 0 sends
ONE message (``GlmEngine._share``): ops ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE, SLOT, which both ranks
apply in order through the same code (rank 0 as it decides them). Rank 1 derives drafts itself and checks the depths
rank 0 sent. An idle rank 1 waits on the doorbell (``GlmEngine._await_bell``), not in an all-gather. After an error
the two ranks cannot be trusted to agree: ``broken`` latches and every later call fails until both restart.

Drafts never change a reply: a request's policy picks DFlash2 or MTP drafts (``multi_code``) and only how many rows
each round verifies depends on it."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tensorfold.cuda.memory_gate import NoRoom as _NoRoom
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream, next_fill

from .multi_tune import MultiSettings, RoundProfile, allocate, reach_of, sample_packed, scaled_confidence
from .pool import ALIGN, Pool, align_up
from .verify import BatchedVerify, Segment, SerialVerify, Verified

ADMIT, EVICT, MOVE, GROW, FILL, ROUND, FINISH, IDLE, SLOT = range(1, 10)
OPS = {ADMIT: "ADMIT", EVICT: "EVICT", MOVE: "MOVE", GROW: "GROW", FILL: "FILL", ROUND: "ROUND", FINISH: "FINISH",
       IDLE: "IDLE", SLOT: "SLOT"}


def watchdog_seconds(value: str | None = None) -> float:
    """TF_GLM_MULTI_WATCHDOG_S: seconds a rank-0 iteration (or a rank-1 message it is owed) may take before every
    thread's Python stack is dumped to the log (faulthandler; 0: off; default 300)."""

    value = os.environ.get("TF_GLM_MULTI_WATCHDOG_S", "") if value is None else value
    try:
        s = 300.0 if value.strip() == "" else float(value)
    except ValueError:
        s = -1.0
    if s < 0:
        raise ValueError(f"TF_GLM_MULTI_WATCHDOG_S: seconds (0: off), not {value!r}")
    return s


def _watch(seconds: float, what: str) -> None:
    if seconds > 0:
        import faulthandler
        import sys

        try:                                              # the process's stderr (a test may have swapped sys.stderr)
            faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.__stderr__, exit=False)
        except (ValueError, OSError, AttributeError):     # no file descriptor to write to: no watchdog
            pass


def _unwatch(seconds: float) -> None:
    if seconds > 0:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()


LONE_LEFT = 16                            # a lone stream moves home only with at least this many tokens still to go
DONE, CANCELLED, REQUEUED = 0, 1, 2      # FINISH reasons
MAX_WINDOW = 32                          # rows of every stream's windows together in one round
MTP_GREEDY = [3, 3, 350000, 0]           # auto without a draft model: the single-stream auto's MTP rules (c3:0.35)
MTP_SAMPLED = [2, 3, 600000, 850000]     # ... and for sampled requests (a:0.6:0.85)


# -- settings ------------------------------------------------------------------------------------------------------
def fill_rows(prefill_rows: int, value: str | None = None) -> int:
    """TF_GLM_FILL_ROWS: rows of a prompt chunk while other streams decode (default 1,024 or the engine's prompt
    chunk if smaller): a multiple of 64, at most the prompt chunk (TF_GLM_PREFILL_ROWS)."""

    value = os.environ.get("TF_GLM_FILL_ROWS", "") if value is None else value
    value = value.strip()
    rows = min(1024, prefill_rows) if value == "" else int(value) if value.isdecimal() else -1
    if rows < 64 or rows % 64 or rows > prefill_rows:
        raise ValueError(f"TF_GLM_FILL_ROWS: a multiple of 64 up to the {prefill_rows}-row prompt chunk, not {value!r}")
    return rows


def fill_share(value: str | None = None) -> float:
    """TF_GLM_FILL_SHARE: the share of iterations prompt chunks take while other streams decode (default 0.5: a
    chunk, a round, a chunk, ...); 1 fills every prompt before decoding on."""

    value = os.environ.get("TF_GLM_FILL_SHARE", "") if value is None else value
    try:
        share = 0.5 if value.strip() == "" else float(value)
    except ValueError:
        share = -1.0
    if not 0 < share <= 1:
        raise ValueError(f"TF_GLM_FILL_SHARE: a share above 0 and at most 1, not {value!r}")
    return share


def multi_code(code: list[int], dflash: bool, mtp: bool, dflash_policy: list[int], greedy: bool = True) -> list[int]:
    """A request's policy code under --parallel (``engine.encode_policy``'s, after ``GlmEngine._effective``): serial
    stays serial; auto becomes ``engine.DFLASH_POLICY`` with the draft model, else the single-stream auto's MTP rule
    (``greedy``: its greedy one) with the MTP head; an MTP policy stays MTP with the head (else its DFlash2 twin), a
    DFlash2 policy stays DFlash2 with the draft model (else its MTP twin); with neither drafter, serial."""

    kind = code[0]
    if kind == 0:
        return [0, 0, 0, 0]
    if kind in (4, 5):
        if dflash:
            return list(dflash_policy)
        return list(MTP_GREEDY if greedy else MTP_SAMPLED) if mtp else [0, 0, 0, 0]
    if kind < 10:                                    # an MTP policy
        if mtp:
            return list(code)
        return [kind + 10] + list(code[1:]) if dflash else [0, 0, 0, 0]
    if dflash:                                       # a DFlash2 policy
        return list(code)
    return [kind - 10] + list(code[1:]) if mtp else [0, 0, 0, 0]


def trim(drafts: list[list[int]], cap: int = MAX_WINDOW) -> list[list[int]]:
    """Cut drafts until every window (a pending token and its drafts) fits ``cap`` rows together: one draft at a
    time off the longest window (the latest of equals)."""

    drafts = [list(d) for d in drafts]
    total = sum(1 + len(d) for d in drafts)
    while total > cap:
        k = max(range(len(drafts)), key=lambda i: (len(drafts[i]), i))
        if not drafts[k]:
            break
        drafts[k].pop()
        total -= 1
    return drafts


def _f64(x: float) -> list[int]:
    import struct

    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _unf64(lo: int, hi: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def pack_sampling(sampling) -> list[int]:
    """10 int32 words: the seed (3 words of 31, 31 and 2 bits), temperature, top_k, top_p, min_p."""

    seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
    return [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
            *_f64(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
            *_f64(sampling.top_p if sampling else 1.0), *_f64(sampling.min_p if sampling else 0.0)]


def unpack_sampling(w: Sequence[int]):
    from tensorfold.engine.exact_sampling import Sampling

    s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi = w
    temperature = _unf64(t_lo, t_hi)
    if temperature <= 0:
        return None
    return Sampling((s_top << 62) | (s_hi << 31) | s_lo, temperature, top_k, _unf64(p_lo, p_hi), _unf64(m_lo, m_hi))


# -- DFlash2 contexts per slot ---------------------------------------------------------------------------------------
def _drafter_views(ctx, n: int) -> list[torch.Tensor]:
    """A flat DFlash2 context's rows a block pass at n or past it reads (TF_GLM_DRAFT_RING=0): its sliding window's
    keys and values before n (every later query sits at n or past it, and masks older rows), per draft layer."""

    lo = max(0, n - ctx.window - 1)
    return [c[:, lo:n] for c in ctx.kc] + [c[:, lo:n] for c in ctx.vc]


def draft_window(ctx, n: int) -> list[torch.Tensor]:
    """A copy of a stream's DFlash2 window rows before n (``dflash2_multi.DraftContext``, or a solo ``Drafter``):
    what a later block pass at n or past it reads."""

    from .decode import _ring_window

    return _ring_window(ctx, n) if getattr(ctx, "ring", 0) else [v.clone() for v in _drafter_views(ctx, n)]


def put_draft_window(ctx, n: int, rows: list[torch.Tensor]) -> None:
    """Put a kept prompt's DFlash2 window back and end the context at n."""

    from .decode import _put_ring_window

    if getattr(ctx, "ring", 0):
        _put_ring_window(ctx, n, rows)
    else:
        views = _drafter_views(ctx, n)
        if len(views) != len(rows):
            raise ValueError("a kept state's DFlash2 window does not match the drafter's context")
        for dst, src in zip(views, rows):
            dst.copy_(src)
    ctx.context_end = n
    ctx.pos_dev.fill_(n)


# -- the verify seam -------------------------------------------------------------------------------------------------
def verify_kind(value: str | None = None) -> str:
    """TF_GLM_MULTI_VERIFY: how a round's windows are verified: batched (the default: one forward over every stream's
    rows, the segmented kernels) or serial (the reference: each stream's window as its own forward)."""

    value = (os.environ.get("TF_GLM_MULTI_VERIFY", "") if value is None else value).strip().lower() or "batched"
    if value not in ("batched", "serial"):
        raise ValueError(f"TF_GLM_MULTI_VERIFY: batched or serial, not {value!r}")
    return value


def forward_streams(verify, segments: Sequence[Segment]) -> Verified:
    """The seam: every segment's logits (and taps) of one round, by ``verify`` (``verify.SerialVerify`` or
    ``verify.BatchedVerify``, the same two calls)."""

    return verify.forward(segments)


def sample_streams(w, parts: Sequence[tuple]) -> list[list[int]]:
    """parts [(logits [R, V/world], positions, sampling)] -> each part's tokens, as ``decode.sample_rows`` samples
    each alone: every top-k (and greedy) part's candidates in ONE all-gather, a nucleus part (top_k off) through the
    shared nucleus rule (its own gathers), in part order on every rank."""

    from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
    from tensorfold.engine.exact_sampling import MARGIN, choose_rows

    out: list[list[int] | None] = [None] * len(parts)
    packed, layout = [], []
    for k, (logits, positions, sampling) in enumerate(parts):
        greedy = sampling is None or sampling.temperature <= 0
        if not greedy and not sampling.top_k:
            continue
        n = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
        vals, ids = torch.topk(logits.float(), n, dim=-1)
        ids = (ids + w.vocab_offset).to(torch.int32)
        packed.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
        layout.append((k, logits.shape[0], n))
    if packed:
        flat = torch.cat(packed) if len(packed) > 1 else packed[0]
        if w.comm is None:
            host = flat.view(1, -1).cpu()
        else:
            got = torch.empty((w.world * flat.numel(),), dtype=torch.float32, device=flat.device)
            w.comm.all_gather(flat.contiguous(), got)
            host = got.view(w.world, -1).cpu()
        at = 0
        for k, R, n in layout:
            g = host[:, at:at + R * 2 * n].reshape(host.shape[0], R, 2 * n)
            at += R * 2 * n
            values = torch.cat([g[r, :, :n] for r in range(g.shape[0])], dim=1).numpy().astype(np.float32)
            tokens = torch.cat([g[r, :, n:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                               dim=1).numpy().astype(np.int64)
            _, positions, sampling = parts[k]
            if sampling is None or sampling.temperature <= 0:
                order = np.lexsort((tokens, -values), axis=-1)
                out[k] = [int(tokens[i, order[i, 0]]) for i in range(R)]
            else:
                out[k] = choose_rows(values, tokens, positions, sampling)
    for k, (logits, positions, sampling) in enumerate(parts):
        if out[k] is None:
            out[k] = nucleus_rows(logits, positions, sampling, offset=w.vocab_offset,
                                  gather=one_rank if w.comm is None else comm_gather(w.comm))
    return out


# -- streams -------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class Lane:
    """An admitted stream: its request (``Stream``), slot, extent, state views and decoding state (both ranks)."""

    s: Stream
    sid: int
    slot: int
    extent: Any
    st: Any
    order: int
    code: list[int]
    spec: str = ""
    policy: Any = None                 # decode.DepthPolicy, None: one row a round
    dflash: bool = False               # the stream drafts with DFlash2 (its slot's context holds its taps)
    mtp: bool = False                  # the stream drafts with the MTP head (its extent's MTP cache holds its rows)
    depth: int = 0
    constraint: Any = None
    window: Any = None                 # the grammar's window of this round's rows
    stops: list[int] = field(default_factory=list)   # prompt positions whose states are kept (before the end)
    point: int | None = None           # where the prompt's state is kept (``keep_point``), or None
    hidden: Any = None                 # MTP: the last committed rows' final-normed hidden rows, not absorbed yet
    nexts: list[int] = field(default_factory=list)   # ... and the tokens after them
    decoding: bool = False
    paused: bool = False
    t0: float = 0.0


def keep_point(n: int, cut: int) -> int | None:
    """Where a drafted prompt of n tokens resumed at ``cut`` keeps its state: before its last token (as the
    single-stream engine's prefill keeps it, so that the same prompt again, or the next turn of its conversation,
    resumes there), or None when that is no further than where it resumed."""

    point = max(1, n - 1)
    return point if point > cut else None


class NoRoom(_NoRoom):
    """The pool cannot place a stream now (it waits for others to finish)."""


class MultiDecoder:
    """The ``GlmScheduler``'s decoder, on rank 0 (deciding) and rank 1 (``follow``, applying)."""

    def __init__(self, engine, streams: int, *, verify=None, drafts=None, draft_graphs: bool | None = None,
                 tune: MultiSettings | None = None, row_ms: Sequence[float] | None = None) -> None:
        """``drafts``: the multi-stream drafter (``dflash2_multi.MultiDrafter``; default one over the engine's
        drafter, its graphs captured unless TF_GLM_MULTI_DRAFT_GRAPHS=0); ``verify``: the verify seam (default
        ``BatchedVerify``, or ``SerialVerify`` with TF_GLM_MULTI_VERIFY=serial)."""
        from .engine import DFLASH_POLICY, encode_policy

        self.g = engine
        self.e = e = engine.e
        self.w = e.w
        self.rank = engine.rank
        self.count = streams
        if e.slots.count < streams:
            raise ValueError(f"the engine holds {e.slots.count} stream slots, not {streams}")
        self.rows_max = e.rows                           # a stream's window (decode.MAX_ROWS)
        self.pool = Pool(e.caches.rows)
        self.arena = e.caches.arena
        self.fill_rows = fill_rows(e.prefill_rows)
        self.share = fill_share()
        self.credit = 0.0
        self.mtp = self.w.mtp is not None                # MTP drafts: the head is loaded (engine.mtp_head)
        self.drafts = drafts
        if self.drafts is None and engine.drafter is not None:
            from .dflash2_multi import MultiDrafter

            self.drafts = MultiDrafter(engine.drafter, streams=streams, tap_rows=self.rows_max)
            graphs = os.environ.get("TF_GLM_MULTI_DRAFT_GRAPHS", "1") != "0" if draft_graphs is None else draft_graphs
            if graphs and torch.cuda.is_available():
                self.drafts.capture()
        if verify is None:
            if verify_kind() == "batched":
                taps = engine.drafter.tap_layers if engine.drafter is not None else ()
                verify = BatchedVerify(e, taps=taps, rows=MAX_WINDOW)
                # TF_GLM_MULTI_GRAPHS=0: eager windows (the same bits); by default a graph a window size
                if os.environ.get("TF_GLM_MULTI_GRAPHS", "1") != "0" and torch.cuda.is_available():
                    verify.capture()
            else:
                verify = SerialVerify(e, taps=self.drafts is not None)
        self.verify = verify
        # speed settings (multi_tune): the sampler's gathers, the drafts' depth, async messages, the profile
        self.tune = tune if tune is not None else MultiSettings.from_env()
        self.profiles = None
        if self.tune.profile and self.rank == 0:
            self.profiles = {False: RoundProfile(self.tune.profile, label="batched"),
                             True: RoundProfile(self.tune.profile, label="lone stream, one-stream graphs")}
        self.profile = None                              # the current round's (``profiles``), or None
        self.solo_verify = SerialVerify(e, taps=self.drafts is not None)
        graphs = getattr(e, "graphs", None)
        self.lone_rows = max((r for r, _ in getattr(graphs, "main", {})), default=0)
        self.row_ms = list(row_ms) if row_ms is not None else None
        if self.tune.depth == "joint" and self.row_ms is None:
            self.row_ms = self._time_rows()
        # both ranks: the comparison replays graphs whose all-gathers the other rank must join (TF_GLM_MULTI_PROFILE
        # is in the two-rank settings comparison, so both run it or neither)
        if self.tune.profile and isinstance(self.verify, BatchedVerify) and self.verify.graphs and self.lone_rows:
            self._compare_paths()
        self.msg = None                                  # TF_GLM_MULTI_ASYNC: rank 0's pinned message staging
        self.watchdog = watchdog_seconds()               # stack dumps of a stalled iteration (TF_GLM_MULTI_WATCHDOG_S)
        self.lanes: dict[int, Lane] = {}                 # by sid, in admission order
        self.kept: list = []                             # kept prompts (decode.Snapshot + kid, extent), oldest first
        self.next_sid = self.next_kid = self.next_order = 0
        self.outbox: list[int] = []
        self.idle = True
        self.broken: Exception | None = None
        self.requeue: list[Stream] = []                  # rank 0: streams given back to the scheduler's queue
        self.eos = tuple(self.w.cfg.eos)
        self.dflash_code = encode_policy(DFLASH_POLICY)

    # -- the scheduler's view ----------------------------------------------------------------------------------------
    @property
    def streams(self) -> dict[int, Stream]:
        """Decoding streams (``Scheduler._yield`` and /health read it)."""
        return {sid: lane.s for sid, lane in self.lanes.items() if lane.decoding}

    @property
    def filling(self) -> list[Stream]:
        return [lane.s for lane in self.lanes.values() if not lane.decoding]

    def live(self) -> int:
        return len(self.lanes)

    def health(self) -> dict:
        lanes = list(self.lanes.values())
        return {"streams": {"decoding": sum(l.decoding and not l.paused for l in lanes),
                            "filling": sum(not l.decoding for l in lanes),
                            "paused": sum(l.decoding and l.paused for l in lanes)},
                "pool_tokens": self.pool.rows, "pool_free_tokens": self.pool.free_rows(),
                "kept_prompts": len(self.kept)}

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    # -- messages --------------------------------------------------------------------------------------------------------
    def _emit(self, op: int, payload: Sequence[int]) -> list[int]:
        payload = [int(v) for v in payload]
        self.outbox += [op, len(payload), *payload]
        return payload

    def _flush(self) -> None:
        """Rank 0: this iteration's ops to rank 1, one message (after the doorbell when rank 1 idles)."""

        if self.idle:
            self.g._ring()
            self.idle = False
        msg, self.outbox = self.outbox, []
        if self.tune.async_msg:
            self._send(msg)
        else:
            self.g._share(msg)

    def _send(self, values: list[int]) -> None:
        """Rank 0, TF_GLM_MULTI_ASYNC: ``GlmEngine._share``'s two all-gathers (the length, then the values) without
        reading anything back: the values go through a pinned staging buffer (reused once its last copy has left),
        so the host does not wait for the GPU. Rank 1 receives them with its ``_share`` as ever. Each gather sends a
        device buffer of its own from its start, as rank 1's ``_share`` does: a transport may choose by the buffer's
        address, and a view past an allocation's start could split the two ranks' choices."""

        n = len(values)
        dev = self.w.device
        if self.msg is None or self.msg[0].numel() < n + 1:
            if self.msg is not None and self.msg[3] is not None:
                self.msg[3].synchronize()
            size = max(4096, 2 * (n + 1))
            pinned = torch.cuda.is_available()
            self.msg = (torch.zeros((size,), dtype=torch.int32, pin_memory=pinned),
                        torch.zeros((1,), dtype=torch.int32, device=dev),
                        torch.zeros((size,), dtype=torch.int32, device=dev),
                        torch.cuda.Event() if pinned else None)
        host, length, vals, ready = self.msg
        if ready is not None:
            ready.synchronize()                          # the last message's copies have left the staging buffer
        host[0] = n
        if n:
            host[1:1 + n].numpy()[:] = values
        length.copy_(host[:1], non_blocking=True)
        if n:
            vals[:n].copy_(host[1:1 + n], non_blocking=True)
        if ready is not None:
            ready.record()
        comm = self.g.comm
        got = torch.empty((2,), dtype=torch.int32, device=dev)
        comm.all_gather(length, got)
        allv = torch.empty((2 * n,), dtype=torch.int32, device=dev)
        comm.all_gather(vals[:n], allv)

    def _time_rows(self, reps: int = 5) -> list[float]:
        """TF_GLM_MULTI_DEPTH=joint: the batched window's ms for 1 .. MAX_WINDOW rows (its graphs, fastest of
        ``reps``), the larger of both ranks' (so both allocate alike)."""

        v = self.verify
        if not isinstance(v, BatchedVerify) or not v.graphs:
            raise ValueError("TF_GLM_MULTI_DEPTH=joint needs the batched verify window's graphs "
                             "(TF_GLM_MULTI_VERIFY=batched, TF_GLM_MULTI_GRAPHS=1)")
        mine = torch.tensor(v.time_rows(reps), dtype=torch.float32, device=self.w.device)
        got = torch.empty((2 * mine.numel(),), dtype=torch.float32, device=self.w.device)
        self.g.comm.all_gather(mine, got)
        return [float(x) for x in got.view(2, -1).max(dim=0).values.tolist()]

    def _compare_paths(self, reps: int = 5) -> None:
        """TF_GLM_MULTI_PROFILE (at startup, on both ranks: the graphs all-gather; rank 0 prints): a lone stream's
        verify window on the one-stream graphs and on the batched graphs, ms by rows up to the one-stream graphs'
        widest (fastest of ``reps``; slot 0's states cleared after)."""

        from .forward import stage
        from .verify import timing_tokens

        e, v = self.e, self.verify
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        solo = v.time_rows(reps, most=self.lone_rows)
        tokens = timing_tokens(self.w.cfg.vocab, v.rows_max)       # the rows time_rows gave the batched graphs
        out = []
        for R in range(1, self.lone_rows + 1):
            g = e.graphs.main.get((R, e.home.parity))
            if g is None:
                continue
            stage(self.w, e.home, e.buf, tokens[:R])
            best = float("inf")
            for _ in range(reps + 1):
                start.record()
                g.replay()
                stop.record()
                stop.synchronize()
                best = min(best, start.elapsed_time(stop))
            out.append(f"{R}: {best:.2f} vs {solo[R - 1]:.2f}")
        e.slots.rec[0].zero_()
        e.slots.conv[0].zero_()
        if self.rank == 0:
            print("[tensorfold] multi profile: a lone stream's verify window ms, one-stream graphs vs batched graphs, by "
                  "rows: " + ", ".join(out), flush=True)

    def _row_cost(self, rows: int) -> float:
        ms = self.row_ms
        return ms[min(max(rows, 1), len(ms)) - 1] + max(0, rows - len(ms)) * (ms[-1] - ms[-2] if len(ms) > 1 else 0)

    @staticmethod
    def parse(msg: Sequence[int]) -> list[tuple[int, list[int]]]:
        ops, i = [], 0
        while i < len(msg):
            op, n = int(msg[i]), int(msg[i + 1])
            ops.append((op, [int(v) for v in msg[i + 2:i + 2 + n]]))
            i += 2 + n
        return ops

    # -- admission (rank 0 decides) ---------------------------------------------------------------------------------------
    def _need(self, prompt: int) -> int:
        return align_up(prompt + self.rows_max)

    def fits(self, s: Stream) -> bool:
        """Whether ``s`` can be placed now (else it waits, in order, for streams to finish): the free rows and the
        kept-only extents cover its prompt, with a block of headroom for each decoding stream to grow into."""

        if not self.lanes:
            return True                  # everything can be evicted; a prompt past the pool fails in ``admit``
        need = self._need(len(s.prompt))
        room = self.pool.free_rows() + sum(x.size for x in self.pool.extents if x.owner is None)
        return room >= need + ALIGN * sum(l.decoding for l in self.lanes.values())

    def admit(self, s: Stream) -> None:
        """Rank 0: place a request (slot, extent; resumed from the longest kept prefix of its prompt when it drafts)
        and queue its ADMIT; its prompt is prefilled chunk by chunk in later iterations."""

        self._check()
        g = self.g
        prompt = [int(t) for t in s.prompt]
        if not prompt:
            raise ValueError("a request needs at least one prompt token")
        if getattr(s, "vision", None) is not None:
            raise ValueError("GLM-5.3-Flash on CUDA takes text prompts")
        if len(prompt) >= g.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {g.limit}")
        s.count = max(1, min(int(s.count), g.limit - len(prompt)))
        if self._need(len(prompt)) > self.pool.rows:
            raise ValueError(f"a prompt of {len(prompt)} tokens does not fit the {self.pool.rows}-token pool")
        info = getattr(s, "glm", None) or {}
        draft = bool(s.draft) and not g.serial_only
        greedy = s.sampling is None or s.sampling.temperature <= 0
        code = (multi_code(info.get("code") or [0, 0, 0, 0], self.drafts is not None, self.mtp, self.dflash_code,
                           greedy) if draft else [0, 0, 0, 0])
        slot = next((k for k in range(self.count) if k not in {l.slot for l in self.lanes.values()}), None)
        if slot is None:
            raise NoRoom(f"every one of the {self.count} stream slots is taken")
        hit = self._resume(prompt, *self._drafters(code)) if draft else None
        need = self._need(len(prompt))
        # placement: take a kept extent over in place, or copy its rows into a new one while a stream writes in it
        copy = 0
        base = None
        if hit is not None and hit.extent.owner is None:
            x = hit.extent
            for c in list(x.kept):            # kept prompts the stream would overwrite: gone
                if len(c.ids) > len(hit.ids) and prompt[:len(c.ids)] != c.ids:
                    self._evict(c)
            if self._grow(x, need, protect=[x]):
                base = x.base
            else:
                hit = None
        if base is None:
            base = self._room(need, [hit.extent] if hit is not None else [])
            if base is None and hit is not None:          # no room beside the kept rows: start fresh
                hit = None
                base = self._room(need)
            if base is None:
                raise NoRoom(f"no room for a {need}-token extent in the {self.pool.rows}-token pool now")
            copy = int(hit is not None)
        sid = self.next_sid
        payload = [sid, slot, base, need, hit.kid if hit is not None else -1, copy, s.count, int(s.stop_eos),
                   int(draft), *pack_sampling(s.sampling), *code, len(prompt), *prompt]
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint) if s.constraint is not None else []
        payload += [len(packed), *packed]
        self._emit(ADMIT, payload)
        try:
            self._admitted(self._parse_admit(payload), s, s.constraint, info.get("spec", ""))
        except Exception as exc:              # rank 1 applies the same ADMIT: the ranks may no longer agree
            self.broken = exc
            raise

    @staticmethod
    def _parse_admit(p: list[int]) -> dict:
        (sid, slot, base, size, kid, copy, count, stop_eos, draft) = p[:9]
        sampling = unpack_sampling(p[9:19])
        code = p[19:23]
        i = 23
        n = p[i]
        prompt = p[i + 1:i + 1 + n]
        i += 1 + n
        n = p[i]
        packed = p[i + 1:i + 1 + n]
        return dict(sid=sid, slot=slot, base=base, size=size, kid=kid, copy=copy, count=count, stop_eos=bool(stop_eos),
                    draft=bool(draft), sampling=sampling, code=code, prompt=prompt, packed=packed)

    def _drafters(self, code: list[int]) -> tuple[bool, bool]:
        """(DFlash2, MTP) drafts for a ``multi_code`` policy code."""

        return code[0] >= 10 and self.drafts is not None, 0 < code[0] < 10 and self.mtp

    def _admitted(self, a: dict, s: Stream, constraint, spec: str = "") -> None:
        """Both ranks: an ADMIT's stream in its slot and extent, resumed from its kept prompt if it names one."""

        from .decode import _absorb_rows
        from .engine import decode_policy
        from .forward import State

        e = self.e
        hit = self._kept_by_id(a["kid"]) if a["kid"] >= 0 else None
        prompt = a["prompt"]
        s.prompt = list(prompt)
        cut = len(hit.ids) if hit is not None else 0
        if hit is not None and (hit.ids != prompt[:cut] or cut >= len(prompt)):
            raise RuntimeError(f"the kept prompt {a['kid']} is not a strict prefix of stream {a['sid']}'s prompt")
        if hit is not None and not a["copy"]:
            x = hit.extent
            if (x.base, x.size) != (a["base"], a["size"]) or x.owner is not None:
                raise RuntimeError(f"stream {a['sid']} takes over extent {x.eid} at [{x.base}, {x.end}), "
                                   f"rank 0 said [{a['base']}, {a['base'] + a['size']})")
        else:
            x = self.pool.add(a["base"], a["size"])
            if hit is not None:
                self.arena.copy(hit.extent.base, x.base, cut)
        x.owner = a["sid"]
        st = State(self.w, x.size, e.rows, caches=e.caches, base=x.base, slots=e.slots, slot=a["slot"])
        code = a["code"]
        dflash, mtp = self._drafters(code)
        policy = decode_policy(code) if code[0] else None
        lane = Lane(s, a["sid"], a["slot"], x, st, self.next_order, code, spec, policy, dflash, mtp,
                    constraint=constraint)
        self.next_sid = max(self.next_sid, a["sid"] + 1)
        self.next_order += 1
        s.sid, s.cached = a["sid"], cut
        # the slot's state: the kept prompt's, or fresh
        st.reset()
        if self.drafts is not None:
            self._ctx(lane).reset()
        if hit is not None:
            st.rec[st.cur[0]].copy_(hit.rec)
            st.conv.copy_(hit.conv)
            st.set_pos(cut)
            if dflash:
                if hit.drafter_end != cut or hit.drafter_rows is None:
                    raise RuntimeError("a kept prompt without DFlash2's window was chosen for a DFlash2 stream")
                put_draft_window(self._ctx(lane), cut, hit.drafter_rows)
            if mtp:                           # the MTP cache up to the kept row's, that row with this prompt's token
                if hit.mtp_len != cut - 1 or hit.pending is None:
                    raise RuntimeError("a kept prompt without the MTP head's rows was chosen for an MTP stream")
                st.set_mtp_len(hit.mtp_len)
                prev = e.use(st)
                try:
                    _absorb_rows(e, hit.pending, [prompt[cut]])
                finally:
                    e.use(prev)
            self._touch(hit)
        # a drafted prompt keeps its state before its last token (``keep_point``)
        if a["draft"]:
            lane.point = keep_point(len(prompt), cut)
            lane.stops = [lane.point] if lane.point is not None and lane.point < len(prompt) else []
        s.started = time.perf_counter()
        lane.t0 = s.started
        self.lanes[lane.sid] = lane

    def _ctx(self, lane: Lane):
        """The lane's slot's DFlash2 context."""
        return self.drafts.contexts[lane.slot]

    # -- kept prompts ------------------------------------------------------------------------------------------------------
    def _kept_by_id(self, kid: int):
        for c in self.kept:
            if c.kid == kid:
                return c
        raise RuntimeError(f"rank {self.rank} holds no kept prompt {kid}")

    def _resume(self, prompt: list[int], dflash: bool = False, mtp: bool = False):
        """The longest kept strict prefix of ``prompt`` that holds what the stream's drafter needs (DFlash2's window,
        or the MTP head's rows)."""

        best = None
        for c in self.kept:
            n = len(c.ids)
            fits = (not dflash or (c.drafter_end == n and c.drafter_rows is not None)) and (
                not mtp or (c.mtp_len == n - 1 and c.pending is not None))
            if fits and n < len(prompt) and prompt[:n] == c.ids and (best is None or n > len(best.ids)):
                best = c
        return best

    def _touch(self, c) -> None:
        self.kept = [k for k in self.kept if k is not c] + [c]

    def _keep(self, lane: Lane, n: int) -> None:
        """Both ranks: the lane's state at prompt position n (== its pos, a prompt chunk's end) becomes a kept prompt
        in its extent; an MTP stream's keeps the last row's hidden state to absorb with the resuming prompt's next
        token (its MTP cache row n - 1 holds this prompt's), as the single-stream prefill keeps it."""

        from .decode import take_snapshot

        prev = self.e.use(lane.st)
        try:
            snap = take_snapshot(self.e, lane.s.prompt[:n], None, mtp=False, drafter=None)
        finally:
            self.e.use(prev)
        snap.drafter_end = -1
        if lane.dflash:
            snap.drafter_end = n
            snap.drafter_rows = draft_window(self._ctx(lane), n)
        if lane.mtp:
            snap.pending = self.e.last_hidden.clone()
            snap.mtp_len = n - 1
        snap.kid = self.next_kid
        self.next_kid += 1
        snap.extent = lane.extent
        for c in [c for c in self.kept if c.ids == snap.ids]:
            self._drop(c)
        lane.extent.kept.append(snap)
        self.kept.append(snap)
        while len(self.kept) > max(1, self.g.cache_entries):
            self._drop(self.kept[0])

    def _drop(self, c) -> None:
        """Forget a kept prompt; its extent goes when nothing else holds it, else shrinks to what is kept."""

        self.kept = [k for k in self.kept if k is not c]
        x = c.extent
        x.kept = [k for k in x.kept if k is not c]
        c.rec = c.conv = c.pending = None
        c.drafter_rows = None
        self._settle(x)

    def _settle(self, x) -> None:
        if x.owner is not None:
            return
        if not x.kept:
            self.pool.remove(x)
            return
        size = align_up(max(len(k.ids) for k in x.kept))
        if size < x.size:
            self.pool.resize(x, size)

    def _evict(self, c) -> None:
        """Rank 0: drop a kept prompt on both ranks."""

        self._emit(EVICT, [c.kid])
        self._drop(c)

    # -- room --------------------------------------------------------------------------------------------------------------
    def _used(self, x) -> int:
        used = max([len(k.ids) for k in x.kept], default=0)
        if x.owner is not None:
            used = max(used, self.lanes[x.owner].st.pos)
        return used

    def _move(self, x, base: int, size: int) -> None:
        """Both ranks: relocate extent x (its used rows copied) to [base, base + size)."""

        used = self._used(x)
        old = self.pool.move(x, base, size)
        self.arena.copy(old, base, used)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _resize(self, x, size: int) -> None:
        self.pool.resize(x, size)
        if x.owner is not None:
            self.lanes[x.owner].st.bind(x.base, x.size)

    def _victims(self, protect) -> list:
        keep = set(id(x) for x in protect)
        return [c for c in self.kept if c.extent.owner is None and id(c.extent) not in keep]

    def _compact(self) -> None:
        """Rank 0: slide every extent down to the lowest free rows (MOVEs), so the free rows are one range."""

        at = 0
        for x in list(self.pool.extents):
            if x.base != at:
                self._emit(MOVE, [x.eid, at, x.size])
                self._move(x, at, x.size)
            at += x.size

    def _room(self, need: int, protect=()) -> int | None:
        """Rank 0: a base with ``need`` free rows, evicting kept prompts (least recently used first, never
        ``protect``'s extents) and compacting as needed; None if even that leaves too little."""

        base = self.pool.place(need)
        while base is None:
            victims = self._victims(protect)
            if not victims:
                break
            self._evict(victims[0])
            base = self.pool.place(need)
        if base is None and self.pool.free_rows() >= need:
            self._compact()
            base = self.pool.place(need)
        return base

    def _grow(self, x, size: int, protect=()) -> bool:
        """Rank 0: extent x to at least ``size`` rows, in place or moved (evicting and compacting as needed)."""

        size = align_up(size)
        if size <= x.size:
            return True
        protect = list(protect) + [x]
        while True:
            if self.pool.room_after(x) >= size - x.size:
                self._emit(GROW, [x.eid, size])
                self._resize(x, size)
                return True
            base = self.pool.place(size, ignore=[x])
            if base is not None:
                self._emit(MOVE, [x.eid, base, size])
                self._move(x, base, size)
                return True
            victims = self._victims(protect)
            if not victims:
                break
            self._evict(victims[0])
        if self.pool.free_rows() + x.size >= size:
            self._compact()
            if self.pool.room_after(x) >= size - x.size:
                self._emit(GROW, [x.eid, size])
                self._resize(x, size)
                return True
            base = self.pool.place(size, ignore=[x])
            if base is not None:
                self._emit(MOVE, [x.eid, base, size])
                self._move(x, base, size)
                return True
        return False

    # -- iterations ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def round(self) -> list[Stream]:
        """Rank 0: one prompt chunk or one decode round; returns the streams that finished."""

        self._check()
        _watch(self.watchdog, "iteration")               # cancelled at the end of ``finish``
        try:
            return self._iterate()
        except Exception as exc:
            self.broken = exc
            raise

    def _fill_turn(self) -> Lane | None:
        filling = [l for l in self.lanes.values() if not l.decoding]
        if not filling:
            return None
        decoding = [l for l in self.lanes.values() if l.decoding and not l.paused and not l.s.done]
        if decoding:
            self.credit += self.share
            if self.credit < 1:
                return None
            self.credit -= 1
        s = next_fill([l.s for l in filling])
        return self.lanes[s.sid]

    def _iterate(self) -> list[Stream]:
        prof = self.profiles[False] if self.profiles is not None else None      # prompt chunks count there
        lane = self._fill_turn()
        if lane is not None:
            t = time.perf_counter()
            stop = self._stop(lane)
            self._emit(FILL, [lane.sid, stop])
            self._flush()
            first = self._fill(lane, stop)
            if first is not None:
                lane.s.take([first], self._ends(lane))
            if prof is not None:
                prof.fill(time.perf_counter() - t)
            return [lane.s] if lane.s.done else []
        lanes = [l for l in self.lanes.values() if l.decoding and not l.s.done]
        t_begin = time.perf_counter()
        for l in lanes:                       # room for this round's widest window, else paused
            l.paused = not self._grow(l.extent, l.st.pos + self.rows_max, protect=[l.extent])
        active = [l for l in lanes if not l.paused]
        if not active:
            done = []
            if lanes and not any(not l.decoding for l in self.lanes.values()):
                done = self._give_back()      # every stream waits for room: the youngest goes back to the queue
            if self.outbox:                   # what was decided (evictions, moves, a stream given back) goes now
                self._flush()
            return done
        lone = False
        if (self.tune.lone and len(active) == 1 and len(self.lanes) == 1 and self.lone_rows
                and active[0].s.count - len(active[0].s.out) >= LONE_LEFT):
            lone = self._home(active[0])
        prof = self.profile = self.profiles[lone] if self.profiles is not None else None
        if prof is not None:
            prof.begin(t_begin)
        plan = [(l.sid, l.depth) for l in active]
        self._emit(ROUND, [len(plan), *[v for p in plan for v in p], int(lone)])
        if prof is not None:
            prof.mark("plan")
        self._flush()
        if prof is not None:
            prof.mark("message")
        self._round(active, lone=lone)
        return [l.s for l in active if l.s.done]

    def _home(self, lane: Lane) -> bool:
        """Rank 0: a lone stream into the one-stream graphs' home, slot 0 at the pool's first rows (SLOT, then
        kept prompts in the way evicted and its extent MOVEd); True once it is there."""

        if self.e.graphed(lane.st):
            return True
        if lane.slot != 0:
            self._emit(SLOT, [lane.sid, 0])
            self._reslot(lane, 0)
        x = lane.extent
        if x.base != 0:
            for c in list(self.kept):                  # kept prompts in [0, x.size) make way
                y = c.extent
                if y is not x and y.base < x.size and c in self.kept:
                    self._evict(c)
            if any(y is not x and y.base < x.size for y in self.pool.extents):
                return False                           # something else holds the home rows (cannot happen alone)
            self._emit(MOVE, [x.eid, 0, x.size])
            self._move(x, 0, x.size)
        return self.e.graphed(lane.st)

    def _reslot(self, lane: Lane, slot: int) -> None:
        """Both ranks: the lane's slot state (KDA states, conv windows, positions, DFlash2 context) copied into
        ``slot``."""

        from .forward import State

        e, old = self.e, lane.slot
        if any(l.slot == slot for l in self.lanes.values() if l is not lane):
            raise RuntimeError(f"stream {lane.sid} cannot take slot {slot}: another stream holds it")
        e.slots.rec[slot].copy_(e.slots.rec[old])
        e.slots.conv[slot].copy_(e.slots.conv[old])
        if self.drafts is not None:
            a, b = self.drafts.contexts[old], self.drafts.contexts[slot]
            for dst, src in zip(list(b.kc) + list(b.vc), list(a.kc) + list(a.vc)):
                dst.copy_(src)
            b.context_end = a.context_end
            b.pos_dev.copy_(a.pos_dev)
        st = State(self.w, lane.extent.size, e.rows, caches=e.caches, base=lane.extent.base, slots=e.slots, slot=slot)
        st.set_pos(lane.st.pos)
        st.set_mtp_len(lane.st.mtp_len)
        st.mtp_drafted = lane.st.mtp_drafted
        st.cur = list(lane.st.cur)
        lane.st, lane.slot = st, slot

    def _stop(self, lane: Lane) -> int:
        pos, n = lane.st.pos, len(lane.s.prompt)
        stop = next((p for p in lane.stops if p > pos), n)
        busy = any(l.decoding and not l.s.done for l in self.lanes.values())
        rows = self.fill_rows if busy or lane.s.background else self.e.prefill_rows
        return min(stop, pos + rows)

    def _ends(self, lane: Lane) -> tuple[int, ...]:
        return self.eos if lane.s.stop_eos else ()

    @torch.no_grad()
    def _fill(self, lane: Lane, stop: int) -> int | None:
        """Both ranks: the lane's prompt rows pos .. stop in one chunk; its kept states; at the prompt's end its first
        token (the stream then decodes)."""

        from .decode import prefill_chunk

        e = self.e
        s = lane.s
        prompt = s.prompt
        n = len(prompt)
        start = lane.st.pos
        if not start < stop <= n or stop - start > e.prefill_rows:
            raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
        t = time.perf_counter()
        prev = e.use(lane.st)
        try:
            out = prefill_chunk(e, prompt, start, stop - start, drafter=self._ctx(lane) if lane.dflash else None,
                                mtp=lane.mtp, head=stop == n)
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop)
                return None
            if lane.point == n:
                self._keep(lane, n)
            if lane.mtp:                                # the first draft absorbs the prompt's last row
                lane.hidden = e.last_hidden.clone()
            logits = out[:1]
            if lane.constraint is not None:             # the first token's row, under the reply's grammar
                lane.constraint.mask(logits, lane.constraint.window([0], [-1]), self.w.vocab_offset)
            first = sample_streams(self.w, [(logits, [n], s.sampling)])[0][0]
            if lane.constraint is not None:
                lane.constraint.advance([first])
        finally:
            e.use(prev)
            s.prefill_s += time.perf_counter() - t
        lane.decoding = True
        lane.nexts = [first]
        s.started = time.perf_counter()
        if lane.policy is not None:
            lane.depth = min(lane.policy.next(0, 0), s.count - 1)
        return first

    def _mtp_draft(self, l: Lane) -> list[int]:
        """Both ranks: an MTP stream's drafts for this round, as ``decode.mtp_decode`` drafts alone: its last rows
        absorbed into its MTP cache, then its chain (on the head's graphs at home, else eagerly)."""

        from .decode import draft

        e = self.e
        prev = e.use(l.st)
        try:
            return draft(e, l.hidden, l.nexts, l.st.pos + 1, l.depth, l.s.sampling, l.policy.confidence)
        finally:
            e.use(prev)

    @torch.no_grad()
    def _round(self, lanes: list[Lane], depths: list[int] | None = None, lone: bool = False) -> None:
        """Both ranks: one decode round over ``lanes`` (``depths``: rank 0's, checked on rank 1); ``lone``: one
        stream at home, verified through the one-stream graphs when its window fits them."""

        from .dflash2_multi import DraftRequest

        if depths is not None and [l.depth for l in lanes] != depths:
            raise RuntimeError(f"rank {self.rank}'s draft depths {[l.depth for l in lanes]} are not rank 0's {depths}")
        prof = self.profile
        tune = self.tune
        drafts: list[list[int]] = []
        asks = []
        for l in lanes:
            drafts.append([])
            if l.depth <= 0:
                continue
            if l.mtp:
                drafts[-1] = list(self._mtp_draft(l))
            elif l.dflash:
                conf = l.policy.confidence
                if tune.depth == "scale":
                    conf = scaled_confidence(conf, len(lanes), tune.alpha)
                asks.append((len(drafts) - 1, DraftRequest(self._ctx(l), l.s.out[-1], l.depth, l.s.sampling, conf)))
        if asks and tune.depth == "joint":
            self._joint(lanes, drafts, asks)
        elif asks:
            got = self.drafts.propose([a for _, a in asks])
            for (k, _), d in zip(asks, got):
                drafts[k] = list(d)
        drafts = trim(drafts, MAX_WINDOW)
        if prof is not None:
            prof.mark("propose")
        segments = []
        for l, d in zip(lanes, drafts):
            tokens = [l.s.out[-1]] + d
            l.window = None
            if l.constraint is not None:                # the chain cut at its first draft the grammar rejects
                l.window = l.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                tokens = list(l.window.tokens)
            segments.append(Segment(l.st, tokens, hidden=l.mtp))
        verify = self.verify
        if lone:
            if not self.e.graphed(lanes[0].st):
                raise RuntimeError(f"rank {self.rank}: a lone round's stream is not at the graphs' home")
            if len(segments[0].tokens) <= self.lone_rows:
                verify = self.solo_verify
        if prof is not None:
            prof.gpu_start()
        v = forward_streams(verify, segments)
        if prof is not None:
            prof.gpu_stop()
            prof.mark("verify")
        parts = []
        for l, seg, logits in zip(lanes, segments, v.logits):
            if l.window is not None:
                l.constraint.mask(logits, l.window, self.w.vocab_offset)
            parts.append((logits, [l.st.pos + 1 + r for r in range(len(seg.tokens))], l.s.sampling))
        sampled = sample_packed(self.w, parts) if tune.sampler == "packed" else sample_streams(self.w, parts)
        if prof is not None:
            prof.mark("sample")
        keeps = []
        for l, seg, rows in zip(lanes, segments, sampled):
            keep = 1
            for i, d in enumerate(seg.tokens[1:]):
                if rows[i] != d or (l.s.stop_eos and rows[i] in self.eos):
                    break
                keep += 1
            keeps.append(keep)
        if prof is not None:
            prof.mark("accept")
        verify.commit(segments, keeps)
        if prof is not None:
            prof.mark("commit")
        if self.drafts is not None:
            items = [(self._ctx(l), v.taps[k][:keep]) for k, (l, keep) in enumerate(zip(lanes, keeps)) if l.dflash]
            if items:
                self.drafts.commit(items)
        if prof is not None:
            prof.mark("taps")
        for k, (l, seg, rows, keep) in enumerate(zip(lanes, segments, sampled, keeps)):
            s = l.s
            new = rows[:keep]
            if l.constraint is not None:
                l.constraint.advance(new)
            if l.mtp:                                   # the kept rows the next draft absorbs, and their next tokens
                l.hidden = v.hidden[k][:keep].clone()
                l.nexts = list(new)
            ndrafts = len(seg.tokens) - 1
            s.counted(len(seg.tokens))                     # Stream.take counts the accepted drafts
            room = s.count - len(s.out)
            s.take(new[:max(0, room)], self._ends(l))
            if l.policy is not None:
                l.depth = min(l.policy.next(ndrafts, keep - 1), max(0, s.count - len(s.out)))
        if prof is not None:
            prof.mark("emit")
            prof.end(len(lanes), sum(len(g.tokens) for g in segments), sum(keeps),
                     sum(len(g.tokens) - 1 for g in segments))

    def _joint(self, lanes: list[Lane], drafts: list[list[int]], asks: list) -> None:
        """TF_GLM_MULTI_DEPTH=joint: every asking stream's chain walked to its depth in one block pass, then the
        round's rows allocated across them by expected tokens per ms (``multi_tune.allocate``). The picks are each
        policy's; only where each chain stops differs (drafts never change a reply)."""

        md = self.drafts
        live = [(k, r) for k, r in asks if min(r.depth, md.block - 1) >= 1 and r.ctx.context_end > 0]
        if not live:
            return
        got = md.candidates([(r.ctx, r.pending, min(r.depth, md.block - 1)) for _, r in live])
        reach = []
        for (k, r), (tokens, values, proj) in zip(live, got):
            confs: list[float] = []
            drafts[k] = md.d.chain(tokens, values, proj, r.pending, r.ctx.context_end + 1, r.sampling, 0.0,
                                   confs=confs)
            reach.append(reach_of(confs))
        asked = {k for k, _ in live}
        fixed = sum(1 + (0 if k in asked else len(d)) for k, d in enumerate(drafts))
        take = allocate(reach, fixed, self._row_cost, self.tune.overhead_ms, MAX_WINDOW)
        for (k, _), n in zip(live, take):
            drafts[k] = drafts[k][:n]

    # -- ends ----------------------------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        """Rank 0: finished (or cancelled) streams leave on both ranks; an idle rank 1 is told to wait for the bell."""

        for s in done:
            lane = self.lanes.get(s.sid)
            if lane is None or lane.s is not s:
                continue
            reason = CANCELLED if s.error is None and len(s.out) < s.count and not (
                s.out and s.out[-1] in self._ends(lane)) else DONE
            self._emit(FINISH, [lane.sid, reason])
            self._finish(lane.sid)
        if not self.lanes and not self.idle and self.broken is None:
            self._emit(IDLE, [])
            self._flush()
            self.idle = True
        _unwatch(self.watchdog)

    def _finish(self, sid: int) -> None:
        """Both ranks: the stream's slot frees; its extent stays for its kept prompts (shrunk to them) or goes."""

        lane = self.lanes.pop(sid)
        x = lane.extent
        x.owner = None
        self._settle(x)
        lane.s.finished = lane.s.finished or time.perf_counter()

    def _give_back(self) -> list[Stream]:
        """Rank 0: the youngest stream (one without a grammar, which cannot replay) leaves its slot and extent and
        returns to the queue; it replays later, its sent tokens owed, not sent again. If every stream has a grammar,
        the youngest fails instead (returned, to be finished)."""

        lanes = sorted(self.lanes.values(), key=lambda l: l.order, reverse=True)
        lane = next((l for l in lanes if l.constraint is None), None)
        if lane is None:
            lane = lanes[0]
            lane.s.error = NoRoom("the pool is full: every stream waits for room, and the youngest cannot replay "
                                  "(a grammar)")
            lane.s.done = True
            return [lane.s]
        for c in list(lane.extent.kept):
            self._evict(c)
        self._emit(FINISH, [lane.sid, REQUEUED])
        self._finish(lane.sid)
        self.requeue.append(lane.s)
        return []

    def drop(self) -> list[Stream]:
        """After an error in an iteration: forget every stream (the ranks can no longer be trusted to agree)."""

        live = [l.s for l in self.lanes.values()]
        self.lanes = {}
        if self.broken is None:
            self.broken = RuntimeError("an iteration failed")
        return live

    def stats(self, s: Stream) -> dict:
        """A finished request's stats (as ``GlmEngine.generate`` reports them)."""

        out = s.stats()
        out["decode_s"] = round(max((s.finished or time.perf_counter()) - s.started, 0.0), 4)
        tokens = list(s.out[:s.count])
        out.update(sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
                   tokens_per_round=round((len(tokens) - 1) / max(s.rounds, 1), 3), parallel=self.count)
        return out

    # -- rank 1 --------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def follow(self, once: bool = False) -> None:
        """Rank 1: apply rank 0's messages forever (``once``: one message)."""

        while True:
            if self.idle:
                self.g._await_bell()
                self.idle = False
            _watch(self.watchdog, "rank 1 message")       # busy: rank 0's next message is due within a round
            self.apply(self.g._share(None))
            _unwatch(self.watchdog)
            if once:
                return

    def apply(self, msg: Sequence[int]) -> None:
        """Rank 1: one iteration's ops, in order."""

        from tensorfold.engine.exact_sampling import Sampling  # noqa: F401

        for op, p in self.parse(msg):
            if op == ADMIT:
                a = self._parse_admit(p)
                s = Stream(list(a["prompt"]), a["count"], a["sampling"], draft=a["draft"], stop_eos=a["stop_eos"],
                           sid=a["sid"])
                constraint = None
                if a["packed"]:
                    from tensorfold.engine import grammar

                    constraint = grammar.compiler(self.g, self.g.model_dir, self.g.eos).follow(a["packed"])
                    s.constraint = constraint
                self._admitted(a, s, constraint)
            elif op == EVICT:
                self._drop(self._kept_by_id(p[0]))
            elif op == MOVE:
                self._move(self.pool.get(p[0]), p[1], p[2])
            elif op == GROW:
                self._resize(self.pool.get(p[0]), p[1])
            elif op == FILL:
                lane = self.lanes[p[0]]
                if p[1] != self._stop_bound(lane, p[1]):
                    raise RuntimeError(f"rank 1: stream {p[0]}'s prompt chunk to {p[1]} passes a kept point")
                first = self._fill(lane, p[1])
                if first is not None:
                    lane.s.take([first], self._ends(lane))
            elif op == ROUND:
                n = p[0]
                sids, depths = p[1:1 + 2 * n:2], p[2:2 + 2 * n:2]
                lone = bool(p[1 + 2 * n]) if len(p) > 1 + 2 * n else False
                self._round([self.lanes[sid] for sid in sids], depths, lone=lone)
            elif op == SLOT:
                self._reslot(self.lanes[p[0]], p[1])
            elif op == FINISH:
                self._finish(p[0])
            elif op == IDLE:
                if self.lanes:
                    raise RuntimeError(f"rank 1 holds {len(self.lanes)} streams rank 0 has finished")
                self.idle = True
            else:
                raise RuntimeError(f"rank 1: unknown op {op}")

    def _stop_bound(self, lane: Lane, stop: int) -> int:
        """The stop rank 0 sent, if it passes no kept point (else where it should have stopped)."""

        nxt = next((q for q in lane.stops if q > lane.st.pos), len(lane.s.prompt))
        return min(stop, nxt)


# -- the scheduler ----------------------------------------------------------------------------------------------------
class GlmScheduler(Scheduler):
    """``cuda.scheduler.Scheduler`` for the GLM decoder: a request's GLM settings ride on its stream (``glm``); a
    request the pool cannot place yet waits at the head of the queue (``MultiDecoder.fits``); streams the decoder gives
    back (every stream waiting for room) return to the queue to replay."""

    def __init__(self, decoder: MultiDecoder, *, max_streams: int) -> None:
        self.held = None
        super().__init__(decoder, max_streams=max_streams)

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, glm: dict | None = None) -> dict:
        """``Scheduler.submit`` with the request's GLM settings (``glm``: its policy code and spec)."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background)
        stream.glm = dict(glm or {})
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left or a stop string: the stream ends next round
            elif kind == "error":
                raise value
            else:
                return value

    def _admit(self, first=None) -> list[Stream]:
        done: list[Stream] = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                item, first = first, None
            elif self.held is not None:
                item, self.held = self.held, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
            stream, box = item
            if not self.decoder.fits(stream):
                self.held = item                     # waits, first in line, for room
                break
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom:
                self.boxes.pop(id(stream))
                self.held = item
                break
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """``Scheduler._yield``, the stream's GLM settings carried to its replay."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(self.decoder.streams.values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        again = stream.continued()
        again.glm = getattr(stream, "glm", {})
        self.waiting.put((again, box))

    def _requeue(self) -> None:
        for stream in self.decoder.requeue:
            box = self.boxes.pop(id(stream), None)
            if box is None:
                continue
            again = stream.continued()
            again.glm = getattr(stream, "glm", {})
            self.yields += 1
            self.waiting.put((again, box))
        self.decoder.requeue = []

    def _loop(self) -> None:
        import sys
        import traceback

        while True:
            try:
                if not self._iteration():
                    return                           # close()
            except Exception as exc:                 # noqa: BLE001  (never a silent dead worker: log, fail, go on)
                print(f"[tensorfold] --parallel scheduler iteration failed: {exc!r}", file=sys.stderr, flush=True)
                traceback.print_exc()
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)

    def _iteration(self) -> bool:
        """One iteration; False once ``close`` asked the idle worker to stop."""

        self._yield()
        idle = not self.decoder.live() and self.held is None
        first = self.waiting.get() if idle else None                  # idle: wait for a request
        if idle and first is None:
            return False
        done = self._admit(first)
        try:
            done += self.decoder.round()
        except Exception as exc:                 # noqa: BLE001  (the live requests fail)
            for s in self.decoder.drop():
                self._reply(s, "error", exc)
        self._requeue()
        try:
            self.decoder.finish(done)
        except Exception as exc:                 # noqa: BLE001
            for s in self.decoder.drop():
                self._reply(s, "error", exc)
        for s in done:
            self._reply(s, *(("error", s.error) if s.error is not None else ("done", self.decoder.stats(s))))
        if self.held is not None and not self.decoder.live():
            time.sleep(0)                        # the held request retries next iteration (now alone)
        _unwatch(getattr(self.decoder, "watchdog", 0))
        return True
