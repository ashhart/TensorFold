"""The target forward on one rank: embedding, Engram, the mHC blocks, the final norm and head rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.cuda.comm import fast_gather
from tensorfold.families.glm5_next.cuda import glue

from ..engram_hash import rank_columns
from . import compressor, engram, moe, mx8, norms
from .attention import attention
from .buffers import Buffers, State
from .weights import LayerW, Weights


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int], hasher=None, reader=None, half: int = 0) -> int:
    """Host work before a forward of ``tokens``: ids and this rank's Engram rows through pinned halves -> R."""

    R = len(tokens)
    if not 0 < R <= b.rows or st.pos + R > st.capacity:
        raise ValueError(f"{R} rows at {st.pos}: buffers hold {b.rows} rows, the state {st.capacity} positions")
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = tokens
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    if w.engram:
        ids = hasher.ids(np.asarray(st.history, dtype=np.int64), np.asarray(tokens, dtype=np.int64))
        engram.stage_rows(rank_columns(ids, w.rank, w.world), reader, b.eraw_host[half], b.eraw_done[half], b.eraw)
    return R


def _gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Every rank's fp32 share ``b.part[:R]`` -> [world, R, D] in rank order (one rank: the share itself)."""

    part = b.part[:R]
    if w.world == 1:
        return part[None]
    recv = b.gath.view(-1)[:w.world * part.numel()].view(w.world, R, -1)
    fast_gather(w.comm, part, recv)
    return recv


def _tap(b: Buffers, R: int, slot: int) -> None:
    """bf16(mean of the 4 copies) of the forward's last min(R, taps) rows into DSpark input slot ``slot``."""

    n = min(R, b.taps.shape[0])
    glue.stream_mean(b.X[R - n:R].view(n, -1), b.hidden[:n])
    b.taps[:n, slot].copy_(b.hidden[:n])


def layer(lw: LayerW, w: Weights, st: State, b: Buffers, R: int, e: torch.Tensor | None, prompt: bool) -> None:
    """One block (M:968-994) on ``b.X[:R]`` in place: Engram, tap, attention, MoE; ``b.pre_in`` is its FFN pre."""

    cfg, L = w.cfg, lw.index
    eps = cfg.rms_norm_eps
    X = b.X[:R]
    flat = X.view(R, -1)
    if L in w.engram:
        eg = w.engram[L]
        kv = engram.kv(e[:, cfg.engram_layer_ids.index(L)], eg.wkv, w.comm, b.ekv, b.ekv_gat, prompt=prompt)
        engram.inject(X, kv, eg.wqk, eps)
    if lw.role.tap is not None:
        _tap(b, R, lw.role.tap)
    h = lw.hc_attn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:R], b.pre_a[:R], b.post_a[:R], b.comb_a[:R], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    attention(lw, w, st, b, R, prompt)
    glue.hc_post(flat, flat, _gather(w, b, R), b.post_a[:R], b.comb_a[:R])
    h = lw.hc_ffn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:R], b.pre_f[:R], b.post_f[:R], b.comb_f[:R], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xf = norms.collapse_norm(flat, b.pre_a[:R], lw.ffn_norm, b.xn[:R], eps)    # delayed mHC: the attention's pre
    moe.backbone(cfg, lw.moe, xf, b, prompt=prompt, comm=w.comm)
    glue.hc_post(flat, flat, _gather(w, b, R), b.post_f[:R], b.comb_f[:R])
    b.pre_in[:R].copy_(b.pre_f[:R])


def head(w: Weights, b: Buffers, a: int, n: int, out: torch.Tensor, *, prompt: bool) -> torch.Tensor:
    """Rows a..a+n of the last forward: final collapse and norm, then the rank's vocabulary rows, fp32."""

    norms.collapse_norm(b.X[a:a + n].view(n, -1), b.pre_in[a:a + n], w.norm, b.fnormed[:n], w.cfg.rms_norm_eps)
    return mx8.mm(w.head, b.fnormed[:n], out, f32=True, prompt=prompt)


def compute(w: Weights, st: State, b: Buffers, R: int, *, prompt: bool, head_rows: int) -> torch.Tensor | None:
    """Staged rows through every loaded layer -> the last ``head_rows`` rows' fp32 logits; commits nothing."""

    if not 0 < R <= b.rows or not 0 <= head_rows <= min(R, b.logits.shape[0]):
        raise ValueError(f"compute: {R} rows, head on {head_rows}, for {b.rows}-row buffers and "
                         f"{b.logits.shape[0]} logit rows")
    cfg = w.cfg
    glue.embed(b.ids[:R], w.embed, cfg.hidden_size, cfg.hc_mult, b.X[:R])
    b.pre_in[:R].zero_()
    b.pre_in[:R, 0].fill_(1.0)                  # the first block collapses to copy 0, the embedding itself
    e = None
    if w.engram:
        e = engram.exchange(engram.dequant_rows(b.eraw[:R], b.eloc[:R]), w.comm, b.egat, b.eng)
    for lw in w.layers:
        layer(lw, w, st, b, R, e, prompt)
    if not head_rows:
        return None
    return head(w, b, R - head_rows, head_rows, b.logits[:head_rows], prompt=prompt)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows: window KV into the rings, the tails, then ``pos += keep``."""

    if not 0 < keep <= R <= b.rows or (b.prefill and keep != R):
        raise ValueError(f"commit: keep {keep} of {R} rows ({'a prompt chunk keeps all' if b.prefill else 'decode'})")
    win, layers = st.rings.shape[1], w.cfg.num_hidden_layers
    n = min(keep, win)
    slots = torch.arange(keep - n, keep, device=st.rings.device).add_(st.pos).remainder_(win)
    st.rings[:layers].index_copy_(1, slots, b.kvw[:layers, keep - n:keep])
    compressor.commit_tail(st, b, st.pos, keep)
    back = w.cfg.engram_max_ngram_size - 1
    st.history = (st.history + b.ids_host[:keep].tolist())[-back:] if back > 0 else []
    st.set_pos(st.pos + keep)
