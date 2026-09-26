"""Qwen3.8 CUDA verify forward over one exact draft tree or chain.

The input state is read-only until ``commit``. A serial decode is the same
forward with one root node, followed by a one-row commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from . import gdn_tree, glue
from .qmm_fast import matmul
from .weights import QLinear, Weights


def _mm(x: torch.Tensor, w: QLinear, xs: torch.Tensor | None = None) -> torch.Tensor:
    return matmul(x, w, xs)


def _row_mm(x: torch.Tensor, w: QLinear, tp: bool,
            xs: torch.Tensor | None = None) -> torch.Tensor:
    if not tp:
        return _mm(x, w, xs)
    from .distributed import gather_rank_partials, row_partial

    return gather_rank_partials(row_partial(x, w, xs=xs if w.layout == "tiled" else None))


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
    if len(parents) > 128 or (not chain and max(depths) >= 32):
        raise ValueError("CUDA GDN supports up to 128 nodes and branching depth below 32")
    return depths, chain


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
    """Committed cache state, with one local sequence and no per-node copies.

    Attention keys and values live in growable buffers: rows [0, pos) are committed and a commit
    writes the accepted rows in place (a copy of the whole cache per round cost ~13 ms at 20k
    keys). Cloned states share buffers; a clone only ever writes rows at or past its own ``pos``,
    so an older clone's rows stay intact, while a longer one that shares the buffer is overwritten
    (``cuda_server`` drops such cache entries when it resumes from a shorter one).
    """

    def __init__(self, w: Weights):
        c = w.config
        device = w.norm.device
        self.pos = 0
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


@torch.no_grad()
def tree_forward(w: Weights, tokens: torch.Tensor, parents: Sequence[int], st: State,
                 *, full_logits: bool = True, tp: bool = False,
                 initial: tuple[torch.Tensor, torch.Tensor] | None = None,
                 finish: bool = True, capture_taps: bool = False):
    """Return node logits and uncommitted per-layer data for a window.

    ``parents`` are topologically sorted; a node sees only its ancestors and
    the committed prefix. The input cache is untouched until ``commit``.
    """

    c = w.config
    parents = [int(p) for p in parents]
    W = len(parents)
    depths, chain = _paths(parents)
    if tokens.shape != (W,) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be a 1-D int tensor matching parents")
    if tokens.device != w.norm.device:
        raise ValueError("tokens and weights must share a device")
    ids = tokens.to(torch.int32)
    pos = torch.tensor([st.pos + d for d in depths], device=tokens.device, dtype=torch.int32)
    parent_buf = torch.tensor(parents, device=tokens.device, dtype=torch.int32)
    windows = _conv_windows(parents, c.conv_kernel - 1).to(tokens.device)
    if initial is None:
        x = glue.embed(ids, w.embed.weight, w.embed.scales, w.embed.biases, c.hidden)
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
            yr = gdn_tree.tree(q, k, v, g, beta, st.rec[i], parent_buf, chain=chain)
            out, out_xs = glue.gated_norm(yr, z, gdn.norm, c.eps)
            r = _row_mm(out, gdn.out, tp, out_xs)
            record.append(GDNRecord(q, k, v, g, beta, qkv))
        else:
            from .attention import attention

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
            old_k, old_v = st.kv[i]
            old_k, old_v = old_k[:st.pos], old_v[:st.pos]
            out = attention(q, key, value, old_k, old_v, parent_buf,
                            scale=c.head_dim ** -0.5)
            gated, out_xs = glue.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim)
            r = _row_mm(gated, attn.o, tp, out_xs)
            record.append(AttentionRecord(key, value))
        x, h, xs = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        gate = _mm(h, layer.gate, xs)
        up = _mm(h, layer.up, xs)
        act, act_xs = glue.swiglu(gate, up)
        pending = _row_mm(act, layer.down, tp, act_xs)
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
    return logits, record


@torch.no_grad()
def commit(st: State, record: Sequence[Record], path: Sequence[int]) -> None:
    """Replay only an accepted root-to-leaf path into the committed state."""

    if not path or len(record) != len(st.rec):
        raise ValueError("record and nonempty path required")
    # All layers at once: one GDN replay launch for every layer, the conv rows of every layer in one
    # gather, the new key/value rows in one multi-tensor copy. Per layer this was ~130 small launches
    # a round (~2.3 ms of a ~62 ms round on two Sparks); the arithmetic is unchanged.
    device = record[0].k.device
    n = len(path)
    take = torch.tensor(list(path), dtype=torch.int64, device=device)
    count = torch.tensor([n], dtype=torch.int32, device=device)
    gdn = [(i, item) for i, item in enumerate(record) if isinstance(item, GDNRecord)]
    att = [(i, item) for i, item in enumerate(record) if not isinstance(item, GDNRecord)]
    if gdn:
        items = [item for _, item in gdn]
        width = items[0].q.shape[0]
        rows = torch.tensor(list(path) + [0] * (width - n), dtype=torch.int32, device=device)
        states = gdn_tree.replay_many([t.q for t in items], [t.k for t in items], [t.v for t in items],
                                      [t.g for t in items], [t.beta for t in items],
                                      [st.rec[i] for i, _ in gdn], rows, count)
        # the last ``keep`` rows of [old state | accepted rows], for every layer
        keep = st.conv[gdn[0][0]].shape[0]
        qkv = torch.stack([t.qkv for t in items])
        if n >= keep:
            conv = qkv[:, take[n - keep:]]
        else:
            conv = torch.cat([torch.stack([st.conv[i] for i, _ in gdn])[:, n:], qkv[:, take]], dim=1)
        for j, (i, _) in enumerate(gdn):
            st.rec[i] = states[j]
            st.conv[i] = conv[j]
    if att:
        need = st.pos + n
        for i, _ in att:
            kbuf, vbuf = st.kv[i]
            if kbuf.shape[0] < need:
                cap = max(need, 2 * kbuf.shape[0], 1024)
                grown_k = kbuf.new_empty((cap, *kbuf.shape[1:]))
                grown_v = vbuf.new_empty((cap, *vbuf.shape[1:]))
                grown_k[:st.pos] = kbuf[:st.pos]
                grown_v[:st.pos] = vbuf[:st.pos]
                st.kv[i] = (grown_k, grown_v)
        keys = torch.stack([item.k for _, item in att])[:, take]
        values = torch.stack([item.v for _, item in att])[:, take]
        torch._foreach_copy_([st.kv[i][0][st.pos:need] for i, _ in att] + [st.kv[i][1][st.pos:need] for i, _ in att],
                             list(keys.unbind(0)) + list(values.unbind(0)))
    st.pos += len(path)
