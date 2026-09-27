"""The MTP head reads final residual streams and next-token embeddings, appending its own attention entries that the next absorb trims after chained drafts."""

from __future__ import annotations

from typing import Sequence

import torch

from . import glue
from .forward import _mm, candidates, finish, layer_forward
from .state import Buffers, State
from .weights import Weights


def mtp_stage(w: Weights, b: Buffers, windows: Sequence[tuple[State, Sequence[int], torch.Tensor]]) -> list:
    """Host work before an MTP step (next tokens, input streams); returns each stream's segment."""

    segs: list = []
    for st, next_tokens, _ in windows:
        a0 = segs[-1][2] if segs else 0
        if st.mtp_len + len(next_tokens) > st.capacity:
            raise ValueError("MTP context past the cache capacity")
        segs.append((st, a0, a0 + len(next_tokens)))
    n = segs[-1][2]
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = [int(t) for _, next_tokens, _ in windows for t in next_tokens]
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    b.last_host[:len(segs)].numpy()[:] = [a1 - 1 for _, _, a1 in segs]
    b.last[:len(segs)].copy_(b.last_host[:len(segs)], non_blocking=True)
    for (_, _, streams), (_, a0, a1) in zip(windows, segs):
        if streams.data_ptr() != b.mtp_in[a0:a1].data_ptr():
            b.mtp_in[a0:a1].copy_(streams, non_blocking=True)
    b.staged.record()
    return segs


def mtp_compute(w: Weights, segs: Sequence, b: Buffers, *, last_only: bool = True,
                context: int | None = None) -> torch.Tensor:
    """The MTP head's GPU work (capturable); ``last_only``: the draft head's logits of each stream's last row."""

    c = w.cfg
    m = w.mtp
    n = segs[-1][2]
    glue.embed(b.ids[:n], *w.embed, c.hidden, copies=1, out=b.mtp_e[:n])
    en, xe = glue.rmsnorm(b.mtp_e[:n], m.norm_e, c.eps, out=b.mixed[:n], xs=b.xs_mixed[:n])
    _mm(en, m.fc_e, xe, b.mtp_eo[:n], b)
    hn, xh = glue.rmsnorm(b.mtp_in[:n], m.norm_h, c.eps, out=b.mtp_hn[:n], xs=b.mtp_xh[:n])
    _mm(hn.view(n * c.streams, c.hidden), m.fc_h, xh.view(n * c.streams, c.hidden // 32), b.mtp_hs[:n * c.streams], b)
    glue.add_streams(b.mtp_eo[:n], b.mtp_hs[:n * c.streams], b.h[:n], c.streams)
    pending = layer_forward(m.layer, w, segs, b, n, None, mtp=True, context=context)
    finish(w, m.mixer, b, n, pending, logits=False)
    if not last_only:
        return _mm(b.mixed[:n], w.head, b.xs_mixed[:n], b.logits[:n], b)
    head = w.head if w.draft_head is None else w.draft_head
    k = len(segs)
    if b.prefill:                                # prefill buffers mix the last row into row 0
        rows, xs = b.mixed[:1], b.xs_mixed[:1]
    elif k == 1:
        rows, xs = b.mixed[n - 1:n], b.xs_mixed[n - 1:n]
    else:
        rows = b.mixed.index_select(0, b.last[:k])
        xs = b.xs_mixed.index_select(0, b.last[:k])
    out = _mm(rows, head, xs, b.logits.view(-1)[:k * head.n].view(k, head.n), b)     # contiguous at any k
    if w.comm is not None:
        candidates(w, b, out, k, id_map=w.draft_ids, offset=int(w.meta["vocab_offset"]))
    return out


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], streams: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Process bf16 main-model streams and next tokens into last-row or all-row logits plus b.streams[:n], appending n cache entries while the caller advances ``st.mtp_len``."""

    return mtp_compute(w, mtp_stage(w, b, [(st, next_tokens, streams)]), b, last_only=last_only)
