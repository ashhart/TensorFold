"""Verify a Qwen3.8 draft tree with state read-only until commit, using the same forward and one-row commit for serial decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.kernels import attention as tree_attention
from tensorfold.cuda.kernels import gdn as deltanet

from . import glue
from .qmm_fast import matmul
from .weights import QLinear, Weights


def _mm(x: torch.Tensor, w: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(w, QLinear):
        return w(x)                                        # an EXL3 pack's weights run their own row-invariant kernels
    return matmul(x, w, xs)


def _row_mm(x: torch.Tensor, w: QLinear, tp: bool,
            xs: torch.Tensor | None = None) -> torch.Tensor:
    if not tp:
        return _mm(x, w, xs)
    from .distributed import gather_rank_partials, row_partial

    return gather_rank_partials(row_partial(x, w, xs=xs if w.layout == "tiled" else None))


def _mlp(layer, h: torch.Tensor, xs: torch.Tensor, tp: bool) -> torch.Tensor:
    """A layer's MLP on its normed rows: routed experts with the shared expert, or the dense SwiGLU."""

    if layer.moe is not None:
        if tp:
            raise ValueError("routed experts run on one GPU")
        return moe.run(h, layer.moe)
    act, act_xs = glue.swiglu(_mm(h, layer.gate, xs), _mm(h, layer.up, xs))
    return _row_mm(act, layer.down, tp, act_xs)


def _paths(parents: Sequence[int]) -> tuple[list[int], bool]:
    if not parents or parents[0] != -1:
        raise ValueError("a verify window needs a root at row zero")
    depths: list[int] = []
    for row, parent in enumerate(parents):
        if parent == -1:
            depths.append(0)
        elif 0 <= parent < row:
            depths.append(depths[parent] + 1)
        else:
            raise ValueError(f"invalid parent {parent} for row {row}")
    chain = all(p == row - 1 for row, p in enumerate(parents))
    if len(parents) > 128:
        raise ValueError("a verify window takes up to 128 nodes")
    return depths, chain


def _cache_offsets(states: Sequence["State"], layers: Sequence[int], device) -> dict[int, torch.Tensor]:
    """Each attention layer's (streams, 2) key and value cache offsets, from one pinned copy."""

    flat = [o for i in layers for o in tree_attention.offsets([st.kv[i] for st in states], device)]
    dev = deltanet.to_device(flat, torch.int64, device).view(len(layers), len(states), 2)
    return dict(zip(layers, dev))


def _conv_windows(parents: Sequence[int], keep: int) -> torch.Tensor:
    """Last ``keep`` inputs along each path, then the node's own QKV row."""

    windows: list[list[int]] = []
    for row, parent in enumerate(parents):
        tail = list(range(keep)) if parent < 0 else windows[parent][1:]
        windows.append(tail + [keep + row])
    return torch.tensor(windows, dtype=torch.int32)


@dataclass
class GDNRecord:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    qkv: torch.Tensor


@dataclass
class AttentionRecord:
    k: torch.Tensor
    v: torch.Tensor


Record = GDNRecord | AttentionRecord


class State:
    """Cloned states share growable KV buffers whose rows below ``pos`` remain committed, so writes preserve shorter clones but invalidate longer cached extensions."""

    rope_delta = 0                      # an image prompt's rotary shift past its tokens; text has none

    def __init__(self, w: Weights):
        c = w.config
        device = w.norm.device
        self.pos = 0
        self.limit = 0                  # attention caches grow to at most this many rows (0: as a commit needs)
        self.conv: list[torch.Tensor | None] = []
        self.rec: list[torch.Tensor | None] = []
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for layer in w.layers:
            if layer.linear:
                cd = 2 * c.k_heads * c.dk + c.v_heads * c.dv
                self.conv.append(torch.zeros((c.conv_kernel - 1, cd), device=device, dtype=torch.bfloat16))
                self.rec.append(torch.zeros((c.v_heads, c.dv, c.dk), device=device, dtype=torch.float32))
                self.kv.append(None)
            else:
                self.conv.append(None)
                self.rec.append(None)
                self.kv.append((torch.empty((0, c.kv_heads, c.head_dim), device=device, dtype=torch.bfloat16),
                                torch.empty((0, c.kv_heads, c.head_dim), device=device, dtype=torch.bfloat16)))


@dataclass
class Staged:
    """A chain window's device inputs for graph replays: static plans, and a pinned mirror copied in before each replay."""

    width: int
    ids: torch.Tensor
    pos: torch.Tensor
    plan: deltanet.Plan
    aplan: tree_attention.Plan
    aoffs: dict[int, torch.Tensor]
    windows: torch.Tensor
    host: torch.Tensor
    dev: torch.Tensor

    def refresh(self, tokens: Sequence[int], p: int) -> None:
        """This round's tokens at positions [p, p + width) (the host mirror, then one copy)."""

        w, h = self.width, self.host.numpy()
        h[:w] = tokens
        h[w:2 * w] = np.arange(p, p + w)
        h[2 * w + w + 2] = p                                 # the attention stream's committed keys and chunks
        h[2 * w + w + 3] = -(-(p + w) // tree_attention.CHUNK)
        self.dev.copy_(self.host, non_blocking=True)


def stage(w: Weights, st: State, width: int, context: int) -> Staged:
    """Static inputs for chains of ``width`` rows over at most ``context`` committed keys (``st``'s buffers fixed)."""

    c, device = w.config, w.norm.device
    parents = list(range(-1, width - 1))
    flat, items, chunks = tree_attention.padded_host(parents, context, c.heads // c.kv_heads)
    host = torch.tensor([0] * (2 * width) + flat, dtype=torch.int32).pin_memory()
    dev = host.to(device)
    softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
    return Staged(width, dev[:width], dev[width:2 * width], deltanet.plan([parents], device),
                  tree_attention.from_packed(dev[2 * width:], 1, width, items, chunks),
                  _cache_offsets([st], softmax, device), _conv_windows(parents, c.conv_kernel - 1).to(device), host, dev)


def reserve(st: State, rows: int) -> None:
    """Grow every attention cache to ``rows`` now, so later commits never move a buffer (graphs keep addresses)."""

    for i, kv in enumerate(st.kv):
        if kv is not None and kv[0].shape[0] < rows:
            k, v = kv[0].new_empty((rows, *kv[0].shape[1:])), kv[1].new_empty((rows, *kv[1].shape[1:]))
            k[:st.pos], v[:st.pos] = kv[0][:st.pos], kv[1][:st.pos]
            st.kv[i] = (k, v)
    st.limit = rows


@torch.no_grad()
def tree_forward(w: Weights, tokens: torch.Tensor, parents: Sequence[int], st: State,
                 *, full_logits: bool = True, tp: bool = False,
                 initial: tuple[torch.Tensor, torch.Tensor] | None = None,
                 finish: bool = True, capture_taps: bool = False, hidden: bool = False,
                 staged: Staged | None = None):
    """Return uncommitted node logits and layer data for topologically sorted parents, with each node seeing only its ancestors and committed prefix."""

    c = w.config
    parents = [int(p) for p in parents]
    W = len(parents)
    depths, chain = _paths(parents)
    if tokens.shape != (W,) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be a 1-D int tensor matching parents")
    if tokens.device != w.norm.device:
        raise ValueError("tokens and weights must share a device")
    if staged is not None:                          # a graph replay's inputs, refreshed in place
        ids, pos, plan, aplan = staged.ids, staged.pos, staged.plan, staged.aplan
        aoffs, windows = staged.aoffs, staged.windows
    else:
        ids = tokens.to(torch.int32)
        pos = torch.tensor([st.pos + st.rope_delta + d for d in depths], device=tokens.device,
                           dtype=torch.int32)
        plan = deltanet.plan([parents], tokens.device)
        softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
        aplan = tree_attention.plan([parents], [st.pos], c.heads // c.kv_heads, tokens.device)
        aoffs = _cache_offsets([st], softmax, tokens.device)
        windows = _conv_windows(parents, c.conv_kernel - 1).to(tokens.device)
    if initial is None:
        x = glue.embedding(ids, w.embed)
        pending: torch.Tensor | None = None
    else:
        x, pending = initial
        if x.shape != (W, c.hidden) or pending.shape != x.shape:
            raise ValueError("pipeline activation shape must match window and hidden width")
    record: list[Record] = []
    taps: list[torch.Tensor] = []
    for i, layer in enumerate(w.layers):
        x, h, xs = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv, xs)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba, xs)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z, xs).reshape(W, c.v_heads, c.dv)
                b = _mm(h, gdn.b, xs)
                a = _mm(h, gdn.a, xs)
            q, k, v, g, beta = glue.gdn_pre(qkv, st.conv[i], gdn.conv, windows, a, b,
                                              gdn.A_log, gdn.dt_bias, kh=c.k_heads,
                                              vh=c.v_heads, dk=c.dk)
            yr = deltanet.tree(q, k, v, g, beta, plan, state=st.rec[i])
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _row_mm(out, gdn.out, tp, out_xs)
            record.append(GDNRecord(q, k, v, g, beta, qkv))
        else:
            attn = layer.attn
            qg = _mm(h, attn.q, xs)
            if attn.kv is not None:
                kv = _mm(h, attn.kv, xs)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k, xs)
                value = _mm(h, attn.v, xs).reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos,
                                    w.inv_freq, c.eps, heads=c.heads, kv_heads=c.kv_heads,
                                    head_dim=c.head_dim)
            out = tree_attention.attention(q, key, value, aoffs[i], aplan, scale=c.head_dim ** -0.5)
            gated, out_xs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _row_mm(gated, attn.o, tp, out_xs)
            record.append(AttentionRecord(key, value))
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(layer, h, xs, tp)
        if capture_taps and i in (5, 19, 33, 47, 61):
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    if not finish:
        if pending is None:
            raise ValueError("pipeline stage must contain at least one layer")
        return (x, pending), record
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs) if full_logits else h
    if capture_taps:
        if len(taps) != 5:
            raise ValueError("DFlash2 taps require the complete 64-layer target")
        return logits, record, torch.cat(taps, dim=-1)
    if hidden:                                     # the rows' final normed states (what an MTP head reads)
        return logits, record, h
    return logits, record


@torch.no_grad()
def multi_tree_forward(w: Weights, streams: Sequence[tuple[Sequence[int], Sequence[int], State]], *,
                       full_logits: bool = True, tp: bool = False, capture_taps: bool = False):
    """Several streams' windows in one forward, each row with the bits of its stream's own ``tree_forward``."""

    c = w.config
    device = w.norm.device
    keep = c.conv_kernel - 1
    starts = [0]
    ids, positions, windows, sids, local, states = [], [], [], [], [], []
    for s, (tokens, parents, st) in enumerate(streams):
        parents = [int(p) for p in parents]
        depths, _ = _paths(parents)
        if len(tokens) != len(parents):
            raise ValueError("each stream needs one token id a parent")
        base = starts[-1]
        rows: list[list[int]] = []
        for row, parent in enumerate(parents):
            tail = list(range(keep)) if parent < 0 else rows[parent][1:]
            rows.append(tail + [keep + base + row])
        windows.extend(rows)
        ids.extend(int(t) for t in tokens)
        positions.extend(st.pos + st.rope_delta + d for d in depths)
        sids.extend([s] * len(parents))
        local.append(parents)
        states.append(st)
        starts.append(base + len(parents))
    W = starts[-1]
    entries, _, slots, most = deltanet.plan_host(local)
    linear = [i for i, layer in enumerate(w.layers) if layer.linear]
    softmax = [i for i, layer in enumerate(w.layers) if not layer.linear]
    ptrs = [p for i in linear for p in deltanet.pointers([st.rec[i] for st in states])]
    tables = dict(zip(linear, deltanet.to_device(ptrs, torch.int64, device).view(len(linear), len(states))))
    aoffs = _cache_offsets(states, softmax, device)
    S = len(states)
    attn_flat, attn_items, attn_chunks = tree_attention.plan_host(local, [st.pos for st in states],
                                                                  c.heads // c.kv_heads)
    host = torch.tensor(positions + sids + entries + starts + ids + [i for win in windows for i in win] + attn_flat,
                        dtype=torch.int32).pin_memory()
    dev = host.to(device, non_blocking=True)            # one copy, and the host runs on
    pos, sid_t = dev[:W], dev[W:2 * W]
    plan = deltanet.Plan(dev[2 * W:5 * W].view(W, 3), dev[5 * W:5 * W + S + 1], slots, most)
    ids_t = dev[5 * W + S + 1:6 * W + S + 1]
    windows_t = dev[6 * W + S + 1:6 * W + S + 1 + W * (keep + 1)].view(W, keep + 1)
    aplan = tree_attention.from_packed(dev[6 * W + S + 1 + W * (keep + 1):], S, W, attn_items, attn_chunks)
    x = glue.embedding(ids_t, w.embed)
    pending: torch.Tensor | None = None
    record: list[Record] = []
    taps: list[torch.Tensor] = []
    for i, layer in enumerate(w.layers):
        x, h, xs = glue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            qkv = _mm(h, gdn.qkv, xs)
            if gdn.zba is not None:
                zba = _mm(h, gdn.zba, xs)
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                z = _mm(h, gdn.z, xs).reshape(W, c.v_heads, c.dv)
                b = _mm(h, gdn.b, xs)
                a = _mm(h, gdn.a, xs)
            conv = states[0].conv[i] if S == 1 else torch.cat([st.conv[i] for st in states])
            q, k, v, g, beta = glue.gdn_pre(qkv, conv, gdn.conv, windows_t, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk, stream_ids=sid_t, nkeep=keep)
            yr = deltanet.tree(q, k, v, g, beta, plan, table=tables[i])
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _row_mm(out, gdn.out, tp, out_xs)
            record.append(GDNRecord(q, k, v, g, beta, qkv))
        else:
            attn = layer.attn
            qg = _mm(h, attn.q, xs)
            if attn.kv is not None:
                kv = _mm(h, attn.kv, xs)
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                key = _mm(h, attn.k, xs)
                value = _mm(h, attn.v, xs).reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos,
                                    w.inv_freq, c.eps, heads=c.heads, kv_heads=c.kv_heads,
                                    head_dim=c.head_dim)
            out = tree_attention.attention(q, key, value, aoffs[i], aplan, scale=c.head_dim ** -0.5)
            gated, out_xs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _row_mm(gated, attn.o, tp, out_xs)
            record.append(AttentionRecord(key, value))
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(layer, h, xs, tp)
        if capture_taps and i in (5, 19, 33, 47, 61):
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    _, h, xs = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head, xs) if full_logits else h
    if capture_taps:
        if len(taps) != 5:
            raise ValueError("DFlash2 taps require the complete 64-layer target")
        return logits, record, torch.cat(taps, dim=-1), starts
    return logits, record, None, starts


@torch.no_grad()
def path_indices(record: Sequence[Record], paths: Sequence[Sequence[int]]) -> list[tuple[torch.Tensor, ...]]:
    """Each path's device indices for ``commit`` (padded rows, count, int64 rows), all from one pinned copy."""

    width = record[0].k.shape[0]
    flat: list[int] = []
    for path in paths:
        flat += list(path) + [0] * (width - len(path)) + [len(path)]
    dev = torch.tensor(flat, dtype=torch.int32).pin_memory().to(record[0].k.device, non_blocking=True)
    out = []
    for j, path in enumerate(paths):
        base = j * (width + 1)
        rows = dev[base:base + width]
        out.append((rows, dev[base + width:base + width + 1], rows[:len(path)].long()))
    return out


def commit(st: State, record: Sequence[Record], path: Sequence[int], indices: tuple | None = None, *,
           in_place: bool = False) -> None:
    """Replay only an accepted root-to-leaf path into the committed state (``in_place``: states no clone shares)."""

    rows, count, take = indices if indices is not None else path_indices(record, [path])[0]
    _commit([st], record, [path], rows.view(1, -1), count, [take], in_place)


def commit_streams(states: Sequence[State], record: Sequence[Record], paths: Sequence[Sequence[int]],
                   indices: list | None = None, *, in_place: bool = False) -> None:
    """Commit each stream's accepted path into its own state with one GDN replay launch for every stream (``in_place``: states nothing else holds)."""

    indices = indices if indices is not None else path_indices(record, paths)
    width = record[0].k.shape[0]
    packed = indices[0][0].as_strided((len(paths), width + 1), (width + 1, 1))    # path_indices' one copy
    _commit(states, record, paths, packed[:, :width], packed[:, width], [t for _, _, t in indices], in_place)


def _commit(states: Sequence[State], record: Sequence[Record], paths: Sequence[Sequence[int]], rows: torch.Tensor,
            counts: torch.Tensor, takes: Sequence[torch.Tensor], in_place: bool = False) -> None:
    if any(not p for p in paths) or any(len(record) != len(st.rec) for st in states):
        raise ValueError("record and nonempty paths required")
    # one GDN replay launch covers every layer and stream
    linear = [(i, item) for i, item in enumerate(record) if isinstance(item, GDNRecord)]
    att = [(i, item) for i, item in enumerate(record) if not isinstance(item, GDNRecord)]
    replayed = None
    if linear:
        items = [item for _, item in linear]
        ptrs = deltanet.replay_table([t.k for t in items], [t.v for t in items], [t.g for t in items],
                                     [t.beta for t in items], [[st.rec[i] for i, _ in linear] for st in states])
        table = deltanet.to_device(ptrs, torch.int64, rows.device)
        replayed = deltanet.replay(table, len(items), len(states), rows, counts, items[0].k, items[0].v,
                                   in_place=in_place)
    device = rows.device
    if linear:
        # a stream's new conv rows are the last ``keep`` of [its committed rows | its accepted rows]
        keep = states[0].conv[linear[0][0]].shape[0]
        width = len(states) * keep
        pick: list[int] = []
        for s, path in enumerate(paths):
            n = len(path)
            pick += [s * keep + j for j in range(n, keep)] + [width + r for r in path[max(0, n - keep):]]
        pick_t = deltanet.to_device(pick, torch.int64, device)
        olds = [states[0].conv[i] if len(states) == 1 else torch.cat([st.conv[i] for st in states]) for i, _ in linear]
        if items[0].qkv.shape[0] <= 32:             # a few rows: every layer in one gather
            src = torch.cat([torch.stack(olds), torch.stack([t.qkv for t in items])], dim=1)
            news = src.index_select(1, pick_t).unbind(0)
        else:                                       # many rows: gather a layer at a time, the record uncopied
            news = [torch.cat([old, t.qkv]).index_select(0, pick_t) for old, t in zip(olds, items)]
        for j, ((i, _), new) in enumerate(zip(linear, news)):
            for s, st in enumerate(states):
                if replayed is not None:
                    st.rec[i] = replayed[s, j]
                if in_place:
                    st.conv[i].copy_(new[s * keep:(s + 1) * keep])
                else:
                    st.conv[i] = new[s * keep:(s + 1) * keep]
    if att:
        for st, path in zip(states, paths):
            need = st.pos + len(path)
            for i, _ in att:
                kbuf, vbuf = st.kv[i]
                if kbuf.shape[0] < need:
                    cap = max(need, 2 * kbuf.shape[0], 1024)
                    if st.limit:
                        cap = max(need, min(cap, st.limit))
                    grown_k = kbuf.new_empty((cap, *kbuf.shape[1:]))
                    grown_v = vbuf.new_empty((cap, *vbuf.shape[1:]))
                    grown_k[:st.pos] = kbuf[:st.pos]
                    grown_v[:st.pos] = vbuf[:st.pos]
                    st.kv[i] = (grown_k, grown_v)
        # every stream's accepted key/value rows: one gather a layer, one multi-tensor copy into every cache
        take = takes[0] if len(takes) == 1 else torch.cat(list(takes))
        dst, src = [], []
        for i, item in att:
            keys, values = item.k.index_select(0, take), item.v.index_select(0, take)
            a0 = 0
            for st, path in zip(states, paths):
                n = len(path)
                dst += [st.kv[i][0][st.pos:st.pos + n], st.kv[i][1][st.pos:st.pos + n]]
                src += [keys[a0:a0 + n], values[a0:a0 + n]]
                a0 += n
        torch._foreach_copy_(dst, src)
    for st, path in zip(states, paths):
        st.pos += len(path)
