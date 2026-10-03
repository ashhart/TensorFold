"""A concurrent round's DeltaNet one launch a step; each tree first folds in last round's kept rows. Bit-equal."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.kernels import gdn as shared

from . import gdn_io
from .gdn import DK, DV


class Scratch:
    """Two rounds' replay inputs a DeltaNet layer (this round's rows, last round's pending ones), and the queries."""

    def __init__(self, w, rows: int) -> None:
        c, dev = w.cfg, w.device
        lin = sum(1 for layer in w.layers if layer.linear)
        self.q = torch.empty((rows, c.nk, DK), dtype=torch.float32, device=dev)
        self.k = torch.empty((2, lin, rows, c.nk, DK), dtype=torch.float32, device=dev)
        self.v = torch.empty((2, lin, rows, c.nv, DV), dtype=torch.bfloat16, device=dev)
        self.g = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.beta = torch.empty((2, lin, rows, c.nv), dtype=torch.float32, device=dev)
        self.lin, self.parity = lin, 0


class Tables:
    """A round's device tables: rows' streams and conv taps, layers' conv and state pointers, chains, pending rows."""

    def __init__(self, w, scratch: Scratch, segs: Sequence, pending: Sequence[Sequence[int]]) -> None:
        n, rows = len(segs), segs[-1][2]
        lin = scratch.lin
        taps = np.arange(4)[None, :]
        win = np.empty((rows, 4), dtype=np.int32)
        sid = np.empty((rows,), dtype=np.int32)
        for s, (_, a0, a1) in enumerate(segs):
            j = np.arange(a1 - a0)[:, None] + taps
            win[a0:a1] = np.where(j < 3, j, a0 + j)            # < 3: a conv state row, else window row tap - 3
            sid[a0:a1] = s
        entries, starts, slots, most = shared.plan_host([list(range(-1, a1 - a0 - 1)) for _, a0, a1 in segs])
        width = max([len(rows_) for rows_ in pending] + [1])
        held = np.zeros((n, width), dtype=np.int32)             # each stream's last-round kept rows, not yet folded
        for s, rows_ in enumerate(pending):
            held[s, :len(rows_)] = rows_
        counts = np.asarray([len(rows_) for rows_ in pending], dtype=np.int32)
        ints = np.concatenate([sid, win.ravel(), np.asarray(entries + starts, dtype=np.int32), held.ravel(), counts])
        ptrs = np.empty((2, lin, n), dtype=np.int64)
        for s, (st, _, _) in enumerate(segs):
            for li in range(lin):
                ptrs[0, li, s] = st.conv[li].data_ptr()
                ptrs[1, li, s] = st.rec[st.cur[li], li].data_ptr()
        dev = w.device
        i32 = shared.to_device(ints.tolist(), torch.int32, dev)
        i64 = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev)
        self.sid, self.win = i32[:rows], i32[rows:5 * rows].view(rows, 4)
        at = 8 * rows + n + 1
        self.plan = shared.Plan(i32[5 * rows:8 * rows].view(rows, 3), i32[8 * rows:at], slots, most)
        self.held, self.held_counts = i32[at:at + n * width].view(n, width), i32[at + n * width:]
        self.folds = bool(counts.any())
        self.conv, self.state = i64[:lin * n].view(lin, n), i64[lin * n:].view(lin, n)
        self.scratch, self.segs, self.cur = scratch, list(segs), scratch.parity


def block(g, li: int, t: Tables, p: torch.Tensor, eps: float, out: torch.Tensor, xs: torch.Tensor, nk: int) -> None:
    """Layer ``li``'s DeltaNet over the round's rows: projections ``p`` [R, W] -> ``out`` [R, NV*DV] and group sums."""

    sc, rows, cur = t.scratch, p.shape[0], t.cur
    q, k, v, gt, beta = (sc.q[:rows], sc.k[cur, li, :rows], sc.v[cur, li, :rows], sc.g[cur, li, :rows],
                         sc.beta[cur, li, :rows])
    gdn_io.front(p, t.conv[li], t.sid, t.win, g.conv, g.a_log, g.dt_bias, nk, out=(q, k, v, gt, beta))
    last = 1 - cur
    pending = ([sc.k[last, li], sc.v[last, li], sc.g[last, li], sc.beta[last, li], t.held, t.held_counts]
               if t.folds else None)
    y = shared.tree(q, k, v, gt, beta, t.plan, table=t.state[li], pending=pending)
    gdn_io.back(y, p, g.norm, eps, out, xs)


def keep(t: Tables, kept: Sequence[int]) -> list[list[int]]:
    """This round's kept window rows a stream, for the next round's trees to fold in (which writes the other set)."""

    t.scratch.parity = 1 - t.cur
    return [list(range(a0, a0 + n)) for (_, a0, _), n in zip(t.segs, kept)]
