"""One attention sublayer (M:765-789); an index layer always scores its own KV source's index keys."""

from __future__ import annotations

import torch

from . import PREFILL_ROWS, attn_kernel, compressor, indexer, mx8, norms, quant, rope
from .buffers import Buffers, State
from .weights import LayerW, Weights

_ROWS: dict[torch.device, torch.Tensor] = {}


def _anchors(pos: torch.Tensor, rows: int) -> torch.Tensor:
    """int32 [rows]: pos + r, each row's last window position, computed on the device so a graph replays it."""

    ar = _ROWS.get(pos.device)
    if ar is None or ar.numel() < rows:
        ar = _ROWS[pos.device] = torch.arange(max(rows, PREFILL_ROWS), dtype=torch.int32, device=pos.device)
    return pos + ar[:rows]


def _index(w: Weights, lw: LayerW, xa: torch.Tensor, qr: torch.Tensor, state: State, buf: Buffers, rows: int,
           table: torch.Tensor, prompt: bool) -> None:
    """M:550-580: the rows' lists into ``buf.lists``; the candidate source keeps its blocks, reindex layers score in them."""

    role, idx = lw.role, lw.attn.idx
    qI, wI = buf.qI[:rows], buf.wI[:rows]
    indexer.index_q(idx, qr, table, state.pos_dev, qI, prompt=prompt)
    indexer.index_weights(idx, xa, wI)
    cand = (buf.cand[:rows], buf.cand_n[:rows])
    indexer.select(w.cfg, qI, wI, state.index_k[role.kv_src], role.ratio, state.pos_dev, buf.scores,
                   buf.lists[:rows], buf.list_n[:rows], pos=state.pos if prompt else None,
                   source=cand if role.candidate_source else None, within=cand if role.uses_candidates else None)


def attention(lw: LayerW, w: Weights, state: State, buf: Buffers, rows: int, prompt: bool) -> torch.Tensor:
    """``buf.X[:rows]`` collapsed with ``buf.pre_in`` -> the rank's fp32 share ``buf.part[:rows]`` after ``wo_b``."""

    cfg, a, role, L = w.cfg, lw.attn, lw.role, lw.index
    if role.mode == "dspark" or not 0 < rows <= buf.rows:
        raise ValueError(f"attention: layer {L} ({role.mode}) on {rows} rows of a {buf.rows}-row buffer")
    eps, table, pos = cfg.rms_norm_eps, w.rope[role.rope], state.pos_dev
    xa, qr, qakv, q, kv = buf.xn[:rows], buf.qr[:rows], buf.qakv[:rows], buf.q[:rows], buf.kvw[L, :rows]
    norms.collapse_norm(buf.X[:rows].view(rows, -1), buf.pre_in[:rows], lw.attn_norm, xa, eps)
    mx8.mm(a.wqa_kv, xa, qakv, prompt=prompt)
    norms.rmsnorm(qakv[:, :cfg.q_lora_rank], a.q_norm, eps, qr)
    norms.rmsnorm(qakv[:, cfg.q_lora_rank:], a.kv_norm, eps, kv)
    mx8.mm(a.wq_b, qr, q.view(rows, -1), prompt=prompt)
    rope.apply(q, pos, table)
    quant.fp8_qdq_1x32(rope.apply(kv, pos, table))
    extra = lists = counts = None
    if role.ratio:
        if role.kv_src == L:
            compressor.compress(lw, xa, state, buf, table, eps)
        if role.idx_src == L:
            _index(w, lw, xa, qr, state, buf, rows, table, prompt)
        extra, lists, counts = state.comp[role.kv_src], buf.lists[:rows], buf.list_n[:rows]
    o = attn_kernel.attention(q, state.rings[L], buf.kvw[L], pos, _anchors(pos, rows), extra, lists, counts,
                              a.sink, buf.o[:rows], prompt=prompt, part=buf.attn_part)
    rope.apply(o, pos, table, inverse=True)
    u = mx8.grouped(a.wo_a, o.view(rows, -1), buf.u[:rows], prompt=prompt)
    return mx8.mm(a.wo_b, u, buf.part[:rows], f32=True, prompt=prompt)
