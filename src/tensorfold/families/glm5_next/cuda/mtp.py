"""MTP reads final-normed main rows, chains its own shared_head.norm output, zeros the position-0 embedding, and trims draft cache entries on absorb."""

from __future__ import annotations

from typing import Sequence

import torch

from . import glue, qmm
from .forward import Buffers, State, check_room, dsa_block, mm, moe_block
from .weights import Weights


def mtp_stage(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor) -> int:
    """Host work before an MTP step: the next tokens and the input hidden rows into the static buffers."""

    n = len(next_tokens)
    b.zero_first = st.mtp_len == 0
    check_room(w, st, n, pos=st.mtp_len)
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = list(next_tokens)
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    if hidden.data_ptr() != b.hin.data_ptr():
        b.hin[:n].copy_(hidden)
    b.staged.record()
    return n


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, *, last_only: bool = True,
                nch: int | None = None, host_pos: int | None = None, sparse_np: int | None = None) -> torch.Tensor:
    """The MTP head's GPU work on staged rows (capturable)."""

    c = w.cfg
    m = w.mtp
    D = c.hidden
    glue.embed(b.ids[:n], w.embed, D, 1, b.me[:n])
    if b.zero_first:
        b.me[0].zero_()
    glue.rmsnorm(b.me[:n], m.enorm, c.eps, b.mcat[:n, :D])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.eps, b.mcat[:n, D:])
    mm(b, b.mcat[:n], m.eh, None if b.prefill else qmm.group_sums(b.mcat[:n], b.mxs[:n]), b.mx[:n])
    layer = m.layer
    glue.rmsnorm(b.mx[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    g = dsa_block(layer, w, st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, b, n, nch,
                  st.index[-1] if st.index is not None else None, host_pos, sparse_np)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    glue.rmsnorm(b.mx[:n], layer.post_norm, c.eps, b.normed[:n], b.xs[:n])
    g = moe_block(layer, w, b, n)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    lo = n - 1 if last_only else 0
    k = n - lo
    glue.rmsnorm(b.mx[lo:n], m.norm, c.eps, b.fnormed[:k], b.fxs[:k])
    head = w.draft_head if w.draft_head is not None else w.head        # the head's logits only draft
    return mm(b, b.fnormed[:k], head, b.fxs[:k], b.logits[:k, :w.head.n])


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Write hidden/token rows into cache slots mtp_len onward and expose logits and b.mx; the caller advances st.mtp_len."""

    n = mtp_stage(w, st, b, next_tokens, hidden)
    from .attention import CHUNK

    return mtp_compute(w, st, b, n, last_only=last_only, nch=-(-(st.mtp_len + n) // CHUNK), host_pos=st.mtp_len)
