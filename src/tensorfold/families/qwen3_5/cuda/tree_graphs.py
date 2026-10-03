"""One stream's tree verify, commit and draft step as CUDA graphs; padded rows never change a real row's bits."""

from __future__ import annotations

import gc
import os
import time
from typing import Callable, Sequence

import numpy as np
import torch

from tensorfold.cuda.kernels import attention as tree_attention
from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.cuda.sampling import sample_rows

from .decode import CopyIndex, DecodeResult, next_copy_rows
from .forward import GDNRecord, State, _paths, commit, grow, tree_forward

ROWS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)       # padded window sizes
SLOTS = (0, 2, 4, 8, 16, 32)                         # gdn.cu's dispatch classes (0: the chain template)
BUCKET = 8192                                        # smallest context a graph's attention items cover


def rows_bucket(n: int) -> int:
    for b in ROWS:
        if b >= n:
            return b
    raise ValueError("a verify window takes up to 128 rows")


def slot_class(slots: int) -> int:
    for c in SLOTS:
        if c >= slots:
            return c
    raise ValueError("a tree needs more than 32 live states")


def enabled() -> bool:
    return os.environ.get("TF_TREE_GRAPHS", "1") != "0"


def padded_inputs(parents: Sequence[int], width: int, keep: int) -> tuple[list, list, list, list, int]:
    """Depths, attention parents, GDN entries, flat conv windows and slot count of a tree padded to ``width`` rows."""

    n = len(parents)
    depths, _ = _paths(parents)
    entries, slots = deltanet.schedule(parents)
    for row in range(n, width):                      # after every real node, each from the state just computed
        entries += [row, -2, -1]
    windows: list[list[int]] = []
    for row, parent in enumerate(parents):
        windows.append((list(range(keep)) if parent < 0 else windows[parent][1:]) + [keep + row])
    windows += [list(range(keep)) + [keep + row] for row in range(n, width)]
    att = list(parents) + [-1] * (width - n)
    return depths + [0] * (width - n), att, entries, [i for win in windows for i in win], slots


class _Staged:
    """A graph's inputs: ids | positions | attention plan | GDN entries | conv windows, pinned and on the device."""

    def __init__(self, w, width: int, context: int, slots: int, aoffs: dict) -> None:
        c, device = w.config, w.norm.device
        self.width, self.keep = width, c.conv_kernel - 1
        flat, n_items, chunks = tree_attention.padded_host(list(range(-1, width - 1)), context, c.heads // c.kv_heads)
        self.o_att = 2 * width
        self.o_stream = self.o_att + width
        self.o_parents = self.o_stream + 4 + 3 * n_items
        self.o_gdn = self.o_att + len(flat)
        self.o_win = self.o_gdn + 3 * width
        total = self.o_win + width * (self.keep + 1)
        self.host = torch.zeros(total, dtype=torch.int32).pin_memory()
        self.h = self.host.numpy()
        self.h[self.o_att:self.o_gdn] = flat
        self.dev = self.host.to(device)
        d = self.dev
        att = d[self.o_att:self.o_gdn]
        self.ids, self.pos = d[:width], d[width:2 * width]
        self.parents = att[width + 4 + 3 * n_items:]
        paths = torch.empty((width, tree_attention.MAX_NODES), dtype=torch.int32, device=device)
        depths = torch.empty((width,), dtype=torch.int32, device=device)
        self.aplan = tree_attention.Plan(att[:width], att[width:width + 4].view(1, 4),
                                         att[width + 4:width + 4 + 3 * n_items].view(n_items, 3), self.parents, paths,
                                         depths, chunks, width)
        self.plan = deltanet.Plan(d[self.o_gdn:self.o_win].view(width, 3), d[:2], slots, width)
        self.windows = d[self.o_win:].view(width, self.keep + 1)
        self.aoffs = aoffs

    def refresh(self, tokens: Sequence[int], parents: Sequence[int], p: int, delta: int) -> None:
        n, W, h = len(tokens), self.width, self.h
        depths, att, entries, windows, _ = padded_inputs(parents, W, self.keep)
        h[:n] = tokens
        h[n:W] = 0
        h[W:2 * W] = np.asarray(depths) + (p + delta)
        h[self.o_stream + 2] = p                                 # the stream's committed keys and slots
        h[self.o_stream + 3] = tree_attention.slots(p, W)
        h[self.o_parents:self.o_parents + W] = att
        h[self.o_gdn:self.o_win] = entries
        h[self.o_win:] = windows
        self.dev.copy_(self.host, non_blocking=True)

    def paths(self) -> None:
        """Each row's path and depth from the refreshed parents (``from_packed``'s kernel)."""

        p = self.aplan
        tree_attention._paths[(self.width,)](self.parents, p.paths, p.depths, MAXD=tree_attention.MAX_NODES,
                                             num_warps=1)


class TreeGraphs:
    """The target's tree verify for one stream, captured per (rows, slot class, context bucket)."""

    def __init__(self, w, *, tp: bool = False, taps: bool = True, full_logits: bool = True,
                 log: Callable[[str], None] | None = None) -> None:
        self.w, self.tp, self.taps, self.full_logits = w, tp, taps, full_logits
        self.log = log
        self.fixed = State(w)                       # GDN states (recurrent, conv) every graph reads and commits update
        self.fixed.kv = [None] * len(self.fixed.kv)
        self.softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
        device = w.norm.device
        self.offs_host = torch.zeros((max(1, len(self.softmax)), 1, 2), dtype=torch.int64).pin_memory()
        self.offs_dev = self.offs_host.to(device)
        self.aoffs = dict(zip(self.softmax, self.offs_dev))
        self.addresses: tuple = ()
        self.graphs: dict[tuple[int, int, int], tuple] = {}
        self.commits: dict[tuple, tuple] = {}       # (verify key, cache addresses): the commit of that graph's record
        self.last: tuple | None = None              # the key and record of the latest verify
        self.pool = None
        self.captures = 0

    def load(self, st: State) -> State:
        """A view of ``st`` with its GDN states copied into the runner's buffers and its own attention caches."""

        view = object.__new__(State)
        view.pos, view.limit, view.rope_delta, view.room = st.pos, st.limit, st.rope_delta, st.room
        view.kv = st.kv
        view.rec, view.conv = list(self.fixed.rec), list(self.fixed.conv)
        dst = [t for t in view.rec + view.conv if t is not None]
        src = [t for t in st.rec + st.conv if t is not None]
        torch._foreach_copy_(dst, src)
        return view

    def _offsets(self, view: State) -> None:
        caches = [view.kv[i] for i in self.softmax]
        addresses = tuple(t.data_ptr() for kv in caches for t in kv)
        if addresses != self.addresses:             # a cache grew (or a new request): every graph reads the new offsets
            flat = tree_attention.offsets(caches, self.w.norm.device)
            self.offs_host.view(-1).numpy()[:] = flat
            self.offs_dev.copy_(self.offs_host, non_blocking=True)
            self.addresses = addresses

    def _peer(self):
        if not self.tp:
            return None
        from tensorfold.cuda import p2p

        return p2p.peer()

    def _graph(self, run: Callable, warm: bool = True):
        """``run`` captured after one eager call when ``warm``: the graph, its outputs and its P2P sum count."""

        if warm:
            run()                                   # kernels compiled and two ranks' sums met outside the capture
        torch.cuda.synchronize()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        peer = self._peer()
        if peer is not None:
            peer.begin_capture()
        g = torch.cuda.CUDAGraph()
        was = gc.isenabled()
        gc.disable()                                # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                out = run()
        finally:
            calls = peer.end_capture() if peer is not None else 0
            if was:
                gc.enable()
        return g, out, calls

    def _capture(self, key: tuple[int, int, int], staged: _Staged):
        width = key[0]
        chain = list(range(-1, width - 1))          # host-side shape only: the staged buffers drive every kernel

        def run():
            staged.paths()
            return tree_forward(self.w, staged.ids, chain, self.fixed, tp=self.tp, full_logits=self.full_logits,
                                capture_taps=self.taps, staged=staged)

        g, out, calls = self._graph(run)
        self.captures += 1
        if self.log is not None:
            self.log(f"[tensorfold] tree verify captured as a CUDA graph: {width} rows, slot class {key[1]}, "
                     f"context {key[2]}")
        return g, out, calls

    def _commit_graph(self, key: tuple, view: State, record) -> tuple:
        """``commit(..., in_place=True)`` of ``key``'s record as a graph; path, position and conv picks are inputs."""

        width, device = key[0], self.w.norm.device
        linear = [(i, item) for i, item in enumerate(record) if isinstance(item, GDNRecord)]
        att = [(i, item) for i, item in enumerate(record) if not isinstance(item, GDNRecord)]
        items = [item for _, item in linear]
        keep = self.fixed.conv[linear[0][0]].shape[0] if linear else 0
        host = torch.zeros(3 * width + 1 + keep, dtype=torch.int64).pin_memory()
        dev = host.to(device)
        rows, count = torch.zeros((1, width), dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32,
                                                                                             device=device)
        table = None
        if linear:
            table = torch.tensor(deltanet.replay_table([t.k for t in items], [t.v for t in items],
                                                       [t.g for t in items], [t.beta for t in items],
                                                       [[self.fixed.rec[i] for i, _ in linear]]),
                                 dtype=torch.int64, device=device)
        caches = [(view.kv[i][0], view.kv[i][1], item) for i, item in att]
        olds = [self.fixed.conv[i] for i, _ in linear]

        def run():
            take, dest = dev[width + 1:2 * width + 1], dev[2 * width + 1:3 * width + 1]
            if linear:
                rows.copy_(dev[:width].view(1, width))
                count.copy_(dev[width:width + 1])
                deltanet.replay(table, len(items), 1, rows, count, items[0].k, items[0].v, in_place=True)
                pick = dev[3 * width + 1:]
                if width <= 32:                     # as ``_commit``: a few rows, every layer in one gather
                    news = torch.cat([torch.stack(olds), torch.stack([t.qkv for t in items])], dim=1).index_select(
                        1, pick).unbind(0)
                else:
                    news = [torch.cat([old, t.qkv]).index_select(0, pick) for old, t in zip(olds, items)]
                torch._foreach_copy_(olds, list(news))
            for k, v, item in caches:               # pad entries repeat the last accepted row into its own slot
                k.index_copy_(0, dest, item.k.index_select(0, take))
                v.index_copy_(0, dest, item.v.index_select(0, take))

        g, _, _ = self._graph(run, warm=False)
        return g, host, host.numpy(), dev, (rows, count, table, caches)   # everything the graph reads stays alive

    @torch.no_grad()
    def commit(self, view: State, record, path: Sequence[int]) -> None:
        """``commit(view, record, path, in_place=True)`` from a graph when ``record`` is the latest verify's."""

        if self.last is None or self.last[1] is not record:
            commit(view, record, path, in_place=True)
            return
        key, n = self.last[0], len(path)
        for i in self.softmax:                      # the eager commit's growth, then the graph for these addresses
            grow(view, i, view.pos + n)
        addresses = tuple(t.data_ptr() for i in self.softmax for t in view.kv[i])
        entry = self.commits.get((key, addresses))
        if entry is None:
            if any(k[0] == key and k[1] != addresses for k in self.commits):     # the caches moved: drop the old
                self.commits = {k: v for k, v in self.commits.items() if k[1] == addresses}
            entry = self.commits[(key, addresses)] = self._commit_graph(key, view, record)
        g, host, h, dev, _ = entry
        width, keep = key[0], len(h) - 3 * key[0] - 1
        p = view.pos
        h[:n] = path
        h[n:width] = 0
        h[width] = n
        h[width + 1:width + 1 + n] = path
        h[width + 1 + n:2 * width + 1] = path[-1]
        h[2 * width + 1:2 * width + 1 + n] = np.arange(p, p + n)
        h[2 * width + 1 + n:3 * width + 1] = p + n - 1
        h[3 * width + 1:] = [j for j in range(n, keep)] + [keep + r for r in path[max(0, n - keep):]]
        dev.copy_(host, non_blocking=True)
        g.replay()
        view.pos += n

    def _context(self, end: int) -> int:
        return max(BUCKET, 1 << (end - 1).bit_length())

    @torch.no_grad()
    def verify(self, view: State, tokens: Sequence[int], parents: Sequence[int]):
        """The eager forward's (logits, record[, taps]) on the real rows: logits sliced, record and taps padded."""

        n = len(tokens)
        width = rows_bucket(n)
        _, slots = deltanet.schedule(parents)
        key = (width, slot_class(slots), self._context(view.pos + width))
        self._offsets(view)
        entry = self.graphs.get(key)
        if entry is None:
            staged = _Staged(self.w, width, key[2], key[1], self.aoffs)
            staged.refresh(tokens, parents, view.pos, view.rope_delta)
            g, out, calls = self._capture(key, staged)
            entry = self.graphs[key] = (g, staged, out, calls)
        g, staged, out, calls = entry
        staged.refresh(tokens, parents, view.pos, view.rope_delta)
        if calls:
            self._peer().before_replay(calls)
        g.replay()
        self.last = (key, out[1])
        return (out[0][:n],) + tuple(out[1:])


@torch.no_grad()
def tree_decode(w, st: State, prompt: Sequence[int], pending: int, count: int, sampling, draft, runner: TreeGraphs,
                *, max_rows: int = 128, tree_rows: int | None = None, allow_copy: bool = True, stop_eos: bool = True,
                on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult:
    """``draft_decode``'s rounds with each verify, commit and draft step replayed from graphs."""

    tree_rows = max_rows if tree_rows is None else tree_rows
    view = runner.load(st)
    out = [pending]
    context = list(prompt) + out
    copies = CopyIndex() if allow_copy else None
    copy_rows = next_copy_rows(tree_rows, False, tree_rows, max_rows)
    rounds = drafted = accepted = 0
    widths: list[int] = []
    eos = w.config.eos if stop_eos else ()
    start = time.perf_counter()
    while len(out) < count and out[-1] not in eos:
        copied = copies.propose(context, copy_rows - 1) if copies is not None else []
        if copied:
            guesses, gparents = copied, list(range(-1, len(copied) - 1))
        else:
            guesses, gparents = draft.propose_tree(out[-1], len(context), tree_rows - 1, sampling)
        tokens = [out[-1]] + list(guesses)
        parents = [-1] + [0 if p < 0 else p + 1 for p in gparents]
        logits, record, taps = runner.verify(view, tokens, parents)
        depths, _ = _paths(parents)
        sampled = sample_rows(logits, [view.pos + d + 1 for d in depths], sampling)
        children: dict[tuple[int, int], int] = {}
        for row in range(1, len(tokens)):
            children.setdefault((parents[row], tokens[row]), row)
        path, terminal = [0], sampled[0]
        while len(out) + len(path) < count:
            if terminal in eos:
                break
            child = children.get((path[-1], terminal))
            if child is None:
                break
            path.append(child)
            terminal = sampled[child]
        runner.commit(view, record, path)
        draft.add_taps(taps[path])
        new = [tokens[r] for r in path[1:]] + [terminal]
        out.extend(new)
        context.extend(new)
        if copied:
            copy_rows = next_copy_rows(copy_rows, len(path) == len(tokens), tree_rows, max_rows)
        rounds, drafted, accepted = rounds + 1, drafted + len(tokens) - 1, accepted + len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None and on_tokens(new):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, rounds, drafted, accepted, widths=widths)


# ---- several streams in one graph -----------------------------------------------------------------------------


class _MultiStaged:
    """Inputs of a graph over S streams' windows, each padded to R rows, pinned and on the device."""

    def __init__(self, w, S: int, R: int, context: int, slots: int, shared: dict) -> None:
        c, device = w.config, w.norm.device
        self.S, self.R, self.keep = S, R, c.conv_kernel - 1
        W = self.width = S * R
        group = c.heads // c.kv_heads
        codes = list(range(context // tree_attention.SPAN)) + [-1 - j for j in range(context // tree_attention.CHUNK)]
        items = [x for s in range(S) for code in codes
                 for first in range(0, R * group, tree_attention.QUERY_TILE) for x in (s, first, code)]
        rows = [s for s in range(S) for _ in range(R)]
        streams = [x for s in range(S) for x in (s * R, R, 0, 0)]
        att = rows + streams + items + [0] * W
        n_items = len(items) // 3
        self.o_att = 3 * W
        self.o_stream = self.o_att + W
        self.o_parents = self.o_stream + 4 * S + 3 * n_items
        self.o_gdn = self.o_att + len(att)
        self.o_starts = self.o_gdn + 3 * W
        self.o_win = self.o_starts + S + 1
        self.o_real = self.o_win + W * (self.keep + 1)
        self.host = torch.zeros(self.o_real + W, dtype=torch.int32).pin_memory()
        h = self.h = self.host.numpy()
        h[2 * W:3 * W] = rows
        h[self.o_att:self.o_gdn] = att
        h[self.o_starts:self.o_win] = [s * R for s in range(S + 1)]
        self.dev = self.host.to(device)
        d = self.dev
        a = d[self.o_att:self.o_gdn]
        self.ids, self.pos, self.sids = d[:W], d[W:2 * W], d[2 * W:3 * W]
        self.parents = a[W + 4 * S + 3 * n_items:]
        chunks = -(-(context + R) // tree_attention.CHUNK)
        self.aplan = tree_attention.Plan(a[:W], a[W:W + 4 * S].view(S, 4), a[W + 4 * S:W + 4 * S + 3 * n_items].view(
            n_items, 3), self.parents, torch.empty((W, tree_attention.MAX_NODES), dtype=torch.int32, device=device),
            torch.empty((W,), dtype=torch.int32, device=device), chunks, W)
        self.plan = deltanet.Plan(d[self.o_gdn:self.o_starts].view(W, 3), d[self.o_starts:self.o_win], slots, R)
        self.windows = d[self.o_win:self.o_real].view(W, self.keep + 1)
        self.real = d[self.o_real:]
        self.starts = [s * R for s in range(S + 1)]
        self.tables, self.aoffs, self.conv = shared["tables"], shared["aoffs"], shared["conv"]

    def refresh(self, wins, states) -> int:
        """This round's windows over the streams' committed positions; returns the real row count."""

        S, R, h, keep = self.S, self.R, self.h, self.keep
        W = self.width
        real = []
        for s, ((tokens, parents), st) in enumerate(zip(wins, states)):
            n, base = len(tokens), s * R
            depths, att, entries, windows, _ = padded_inputs(parents, R, keep)
            h[base:base + n] = tokens
            h[base + n:base + R] = 0
            h[W + base:W + base + R] = np.asarray(depths) + (st.pos + st.rope_delta)
            h[self.o_stream + 4 * s + 2] = st.pos
            h[self.o_stream + 4 * s + 3] = tree_attention.slots(st.pos, R)
            h[self.o_parents + base:self.o_parents + base + R] = [x if x < 0 else x + base for x in att]
            e = np.asarray(entries).reshape(R, 3)
            e[:, 0] += base
            h[self.o_gdn + 3 * base:self.o_gdn + 3 * (base + R)] = e.reshape(-1)
            win = np.asarray(windows).reshape(R, keep + 1)
            win[win >= keep] += base                     # window rows, not committed ones, shift by the base
            h[self.o_win + base * (keep + 1):self.o_win + (base + R) * (keep + 1)] = win.reshape(-1)
            real.extend(range(base, base + n))
        h[self.o_real:self.o_real + len(real)] = real
        self.dev.copy_(self.host, non_blocking=True)
        return len(real)

    def paths(self) -> None:
        p = self.aplan
        tree_attention._paths[(self.width,)](self.parents, p.paths, p.depths, MAXD=tree_attention.MAX_NODES,
                                             num_warps=1)


class MultiGraphs(TreeGraphs):
    """``multi_tree_forward`` captured per (streams, rows a stream, slot class, context bucket)."""

    def __init__(self, w, *, tp: bool = False, taps: bool = True, full_logits: bool = True,
                 log: Callable[[str], None] | None = None) -> None:
        super().__init__(w, tp=tp, taps=taps, full_logits=full_logits, log=log)
        self.linear = [i for i, layer in enumerate(w.layers) if layer.linear]
        self.per: dict[int, dict] = {}              # stream count -> state tables, cache offsets, conv staging
        self.mgraphs: dict[tuple[int, int, int, int], tuple] = {}

    def _shared(self, S: int) -> dict:
        got = self.per.get(S)
        if got is None:
            device = self.w.norm.device
            th = torch.zeros((max(1, len(self.linear)), S), dtype=torch.int64).pin_memory()
            oh = torch.zeros((max(1, len(self.softmax)), S, 2), dtype=torch.int64).pin_memory()
            td, od = th.to(device), oh.to(device)
            conv = {i: torch.empty((S * self.fixed.conv[i].shape[0], self.fixed.conv[i].shape[1]),
                                   dtype=self.fixed.conv[i].dtype, device=device) for i in self.linear}
            got = self.per[S] = {"th": th, "oh": oh, "tables": dict(zip(self.linear, td)),
                                 "aoffs": dict(zip(self.softmax, od)), "td": td, "od": od, "conv": conv,
                                 "rec": (), "kv": ()}
        return got

    def _states(self, shared: dict, states) -> None:
        rec = tuple(st.rec[i].data_ptr() for st in states for i in self.linear)
        if rec != shared["rec"]:
            shared["th"].numpy()[:] = np.asarray(rec, dtype=np.int64).reshape(len(states), -1).T
            shared["td"].copy_(shared["th"], non_blocking=True)
            shared["rec"] = rec
        caches = [st.kv[i] for i in self.softmax for st in states]
        kv = tuple(t.data_ptr() for c in caches for t in c)
        if kv != shared["kv"]:
            shared["oh"].view(-1).numpy()[:] = tree_attention.offsets(caches, self.w.norm.device)
            shared["od"].copy_(shared["oh"], non_blocking=True)
            shared["kv"] = kv
        conv = shared["conv"]
        keep = self.fixed.conv[self.linear[0]].shape[0] if self.linear else 0
        dst = [conv[i][s * keep:(s + 1) * keep] for i in self.linear for s in range(len(states))]
        src = [st.conv[i] for i in self.linear for st in states]
        if dst:
            torch._foreach_copy_(dst, src)          # the streams' conv rows into the graphs' staging

    @torch.no_grad()
    def verify_streams(self, wins, states):
        """Compact logits, their starts, padded record and taps, and the padded starts the commit takes."""

        S = len(states)
        R = rows_bucket(max(len(t) for t, _ in wins))
        slots = max(deltanet.schedule(p)[1] for _, p in wins)
        key = (S, R, slot_class(slots), self._context(max(st.pos for st in states) + R))
        shared = self._shared(S)
        self._states(shared, states)
        entry = self.mgraphs.get(key)
        if entry is None:
            staged = _MultiStaged(self.w, S, R, key[3], key[2], shared)
            staged.refresh(wins, states)
            g, out, calls = self._capture_multi(key, staged)
            entry = self.mgraphs[key] = (g, staged, out, calls)
        g, staged, out, calls = entry
        n = staged.refresh(wins, states)
        if calls:
            self._peer().before_replay(calls)
        g.replay()
        logits, record, taps = out[0], out[1], out[2] if self.taps else None
        compact = logits.index_select(0, staged.real[:n])
        lengths = [len(t) for t, _ in wins]
        starts = [0]
        for k in lengths:
            starts.append(starts[-1] + k)
        return compact, starts, record, taps, staged.starts

    def _capture_multi(self, key, staged: _MultiStaged):
        from .forward import multi_tree_forward

        def run():
            staged.paths()
            return multi_tree_forward(self.w, (), full_logits=self.full_logits, tp=self.tp, capture_taps=self.taps,
                                      staged=staged)

        g, out, calls = self._graph(run)
        self.captures += 1
        if self.log is not None:
            self.log(f"[tensorfold] verify of {key[0]} streams captured as a CUDA graph: {key[1]} rows each, slot "
                     f"class {key[2]}, context {key[3]}")
        return g, out, calls
