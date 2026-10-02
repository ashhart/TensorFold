"""The verify seam of GLM-5.3-Flash's concurrent streams (``multi.forward_streams``): a round's windows, one per
stream (its state: extent and slot views; its tokens: the pending token, then its drafts), in, each window's logits
and DFlash2 taps out, then each stream's commit of the rows it keeps. Two implementations of the same two calls:

- ``SerialVerify``: each window as its own forward on its own state (the solo kernels: exact by construction; a CUDA
  graph where the state is the one they were captured on);
- ``BatchedVerify``: ONE forward over every stream's rows back to back: the row-independent parts (embeddings,
  hyper-connection glue, projections, MLP / MoE, head) on the whole window, KDA layers through the segmented chain
  (``kda.chain_segments``: each segment from its slot's state, into its slot's other parity), DSA layers through the
  segmented latent / sparse attention over the streams' extents (``forward.dsa_segments``, ``segments.SegRows``);
  the commit replays each segment's kept rows (``kda.replay_layers_segments``) and shifts its conv window
  (``forward.conv_shift_segments``). Every segment's rows get the bits of that stream's window alone.

The scheduler (``multi.MultiDecoder``) only sees ``forward`` / ``commit``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Segment:
    """One stream's rows of a round's verify window: its state (extent and slot views) and tokens (the pending
    token, then its drafts); ``hidden``: its rows' final-normed hidden states come back too (the MTP head's input)."""

    st: Any
    tokens: list[int]
    hidden: bool = False


@dataclass
class Verified:
    logits: list[torch.Tensor]              # per segment [rows, V / world] (this rank's vocabulary columns)
    taps: list[torch.Tensor | None]         # per segment [rows, taps * D] (DFlash2's inputs), or None
    hidden: list[torch.Tensor | None] = field(default_factory=list)   # per segment [rows, D] if it asked, else None


class SerialVerify:
    """The reference: each segment's window as its own forward on its own state. Logits and taps are copied out
    before the next segment's forward reuses the buffers; each slot keeps its own KDA window scratch, so the commits
    can come after every forward."""

    def __init__(self, e, *, taps: bool) -> None:
        self.e, self.taps = e, taps

    def forward(self, segments: Sequence[Segment]) -> Verified:
        e = self.e
        logits, taps, hidden = [], [], []
        for seg in segments:
            prev = e.use(seg.st)
            try:
                R = len(seg.tokens)
                logits.append(e.forward(seg.tokens)[:R].clone())
                taps.append(e.tap_rows(R).clone() if self.taps else None)
                hidden.append(e.main_hidden(slice(0, R)).clone() if seg.hidden else None)
            finally:
                e.use(prev)
        return Verified(logits, taps, hidden)

    def commit(self, segments: Sequence[Segment], keeps: Sequence[int]) -> None:
        from .forward import commit

        e = self.e
        for seg, keep in zip(segments, keeps):
            commit(e.w, seg.st, e.buf, len(seg.tokens), keep)


def timing_tokens(vocab: int, n: int) -> list[int]:
    """Fixed, distinct token ids for startup timings of verify windows (rows of real text route to many experts)."""

    import numpy as np

    return [int(t) for t in np.random.default_rng(1234).choice(min(vocab, 150000), size=n, replace=False)]


class BatchedVerify:
    """One forward over every segment's rows (``rows`` at most, ``segments.MAX_SEGS`` streams), on buffers of its
    own; see the module docstring. ``taps``: the drafter's tap layers (DFlash2's inputs), or ()."""

    def __init__(self, e, *, taps: Sequence[int] = (), rows: int = 32) -> None:
        from . import kda as kda_mod
        from . import latent
        from .forward import Buffers
        from .segments import MAX_SEGS, SegRows, SelectScratch

        if not latent.ENABLED:
            raise ValueError("batched verify windows need the latent cache (TF_GLM_LATENT=1)")
        w = e.w
        c = w.cfg
        self.e, self.w, self.rows_max = e, w, rows
        dev = w.device
        self.b = Buffers(w, rows, e.caches.rows)
        if taps:
            self.b.set_taps(tuple(taps), c.hidden)
        self.taps = bool(taps)
        self.seg_rows = SegRows(rows, dev, max_segs=MAX_SEGS)
        self.sel = SelectScratch(rows, e.caches.rows, dev) if e.caches.index is not None else None
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        n = len(kda_layers)
        LL = c.lin_heads // w.world
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(l for l in w.layers if l.kind == "dsa")}
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.kseg = torch.zeros((MAX_SEGS, kda_mod.SEG_COLS), dtype=torch.int32, device=dev)
        self.kseg_host = torch.zeros((MAX_SEGS, kda_mod.SEG_COLS), dtype=torch.int32,
                                     pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        self.segments: list[Segment] = []
        self.R = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.replays = {"graph": 0, "eager": 0}

    @torch.no_grad()
    def capture(self, rows: Sequence[int] | None = None) -> None:
        """CUDA graphs of the window for each row count (default 1 .. its rows), both ranks together: positions,
        extents, slots and parities live in the device tables, so one graph a row count covers every mix. Captured on
        a window of slot 0 at the pool's first rows (the warm-up runs write there; slot 0's states are cleared after,
        so capture before any stream is admitted). With the indexer the graphs always run the token selection (its
        dense rows ignore it: the same bits as a window that skips it)."""

        from . import kda as kda_mod
        from .forward import compute
        from .sparse import SPARSE_FROM

        e, w, b = self.e, self.w, self.b
        index = e.caches.index is not None
        # the KDA chain's wide scratch at its largest (prompt chunks) before any capture: a graph keeps its addresses
        kda_mod.reserve(max(self.rows_max, e.prefill_rows), w.cfg.lin_heads // w.world, w.device)
        pool = torch.cuda.graph_pool_handle()
        for R in rows or range(1, self.rows_max + 1):
            self._stage([0] * R)
            self.seg_rows.set([(0, 0, R)], sparse_from=SPARSE_FROM if index else None)
            self._write_table([(R, 0, 0, 0, R)])
            def run(R=R) -> None:
                compute(w, None, b, R, mixer=lambda layer: self._mixer(layer, R, index))

            for _ in range(2):
                run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                run()
            self.graphs[R] = g
        torch.cuda.synchronize()
        e.slots.rec[0].zero_()
        e.slots.conv[0].zero_()

    # -- the window's tables -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def time_rows(self, reps: int = 5, most: int | None = None) -> list[float]:
        """ms of the captured window of each size (1 .. its rows): the fastest of ``reps`` replays, on the capture's
        own tables (slot 0 at the pool's first rows; its states cleared after), before any stream is admitted. The
        rows are ``timing_tokens`` (distinct tokens: a window of one token repeated routes every row to the same few
        experts and times a 16-row window at about half of what 16 real rows cost)."""

        from .sparse import SPARSE_FROM

        index = self.e.caches.index is not None
        out = []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tokens = timing_tokens(self.w.cfg.vocab, self.rows_max)
        for R in range(1, min(most or self.rows_max, self.rows_max) + 1):
            g = self.graphs[R]
            self._stage(tokens[:R])
            self.seg_rows.set([(self.e.home.base, 0, R)], sparse_from=SPARSE_FROM if index else None)
            self._write_table([(R, 0, 0, 0, R)])
            best = float("inf")
            for _ in range(reps + 1):
                start.record()
                g.replay()
                stop.record()
                stop.synchronize()
                best = min(best, start.elapsed_time(stop))
            out.append(best)
        self.e.slots.rec[0].zero_()
        self.e.slots.conv[0].zero_()
        return out

    def _tables(self, segments: Sequence[Segment], keeps: Sequence[int] | None = None) -> None:
        """The KDA segment table (row0, rows, slot, parity, conv slot, keep) into the device copy, in place (a captured
        graph reads the same tensor); unused segments have 0 rows."""

        self._write_table([(len(s.tokens), s.st.slot, s.st.cur[0] if s.st.cur else 0, s.st.slot,
                            len(s.tokens) if keeps is None else keeps[k]) for k, s in enumerate(segments)])

    def _write_table(self, rows) -> None:
        from . import kda as kda_mod

        table = kda_mod.segment_table(rows, "cpu", out=torch.zeros_like(self.kseg_host))
        if self.staged is not None:
            self.staged.synchronize()
        self.kseg_host.copy_(table)
        self.kseg.copy_(self.kseg_host, non_blocking=True)
        if self.staged is not None:
            self.staged.record()

    def _stage(self, tokens: list[int]) -> None:
        b = self.b
        R = len(tokens)
        if R > b.rows:
            raise ValueError(f"a window of {R} rows, the batched buffers hold {b.rows}")
        b.staged.synchronize()
        b.ids_host[:R].numpy()[:] = tokens
        b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
        b.staged.record()

    # -- the seam --------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, segments: Sequence[Segment]) -> Verified:
        from .forward import check_room, compute
        from .sparse import SPARSE_FROM

        w, b = self.w, self.b
        index = self.e.caches.index is not None
        spans = []
        for seg in segments:
            check_room(w, seg.st, len(seg.tokens))
            spans.append((seg.st.base, seg.st.pos, len(seg.tokens)))
        tokens = [t for seg in segments for t in seg.tokens]
        R = len(tokens)
        self._stage(tokens)
        self.seg_rows.set(spans, sparse_from=SPARSE_FROM if index else None)
        self._tables(segments)
        self.segments, self.R = list(segments), R
        g = self.graphs.get(R)
        if g is not None:
            self.replays["graph"] += 1
            g.replay()
            logits = b.logits[:R]
        else:
            self.replays["eager"] += 1
            select = index and self.seg_rows.any_sparse()
            logits = compute(w, None, b, R, mixer=lambda layer: self._mixer(layer, R, select))
        out, taps, hidden, at = [], [], [], 0
        for seg in segments:
            n = len(seg.tokens)
            out.append(logits[at:at + n])
            taps.append(torch.cat([t[at:at + n] for t in b.taps], dim=1) if self.taps else None)
            hidden.append(b.fnormed[at:at + n].clone() if seg.hidden else None)
            at += n
        return Verified(out, taps, hidden)

    def _mixer(self, layer, R: int, select: bool):
        """A layer's KDA or DSA block over the whole window, each segment on its own stream's state."""

        from . import prof
        from .forward import dsa_segments, kda_segments

        e = self.e
        if layer.kind == "kda":
            with prof.timed("kda"):
                return kda_segments(layer, self.w, self.b, R, self.kseg, e.slots, self.proj, self.scratch,
                                    self.kda_index[layer.index])
        caches = e.caches
        di = self.dsa_index[layer.index]
        planes = caches.arena.planes
        lc = planes[caches.kc[di]].tensor
        index = None if caches.index is None else tuple(planes[i].tensor for i in caches.index[di])
        with prof.timed("dsa (total)"):
            return dsa_segments(layer, self.w, lc, self.b, R, self.seg_rows, self.sel, index, select=select)

    @torch.no_grad()
    def commit(self, segments: Sequence[Segment], keeps: Sequence[int]) -> None:
        """Each segment keeps its first ``keeps[k]`` rows: its KDA states replayed to them (when it keeps fewer than
        its rows), its conv window shifted, its parity flipped, its position advanced."""

        from . import kda as kda_mod
        from .forward import conv_shift_segments

        if [id(s) for s in segments] != [id(s) for s in self.segments]:
            raise ValueError("commit takes the segments of the last forward")
        for seg, keep in zip(segments, keeps):
            if not 1 <= keep <= len(seg.tokens):
                raise ValueError("keep must be in 1..rows")
        e = self.e
        if self.scratch is not None:
            self._tables(segments, keeps)
            if any(k < len(s.tokens) for s, k in zip(segments, keeps)):
                kda_mod.replay_layers_segments(self.kseg, e.slots.rec, self.scratch)
            conv_shift_segments(e.slots.conv, self.proj, self.kseg)
        for seg, keep in zip(segments, keeps):
            st = seg.st
            if st.cur:
                st.cur = [1 - st.cur[0]] * len(st.cur)
            st.set_pos(st.pos + keep)
        self.segments = []
