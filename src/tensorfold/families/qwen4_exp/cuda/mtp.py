"""Flash Next's MTP head on CUDA, through the same row-invariant kernels as the main model.

At position i it reads the main model's residual streams after the last layer (before the final mixer)
and the embedding of token i + 1:

    x = fc_embedding(norm_e(embed(t_{i+1})))  added to each stream of  fc_hidden(norm_h(h_i)) per stream
    x = decoder layer (attention over the head's own cache, MoE), four streams
    logits = lm_head(mixer(x))

Its output streams feed the next draft the same way. The head's attention cache holds one entry per
absorbed position (``State.mtp_len``); chained drafts append entries that the next absorb trims.
"""

from __future__ import annotations

from typing import Sequence

import torch

from . import glue
from .forward import Buffers, State, _mm, candidates, finish, layer_forward
from .weights import Weights


def mtp_stage(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], streams: torch.Tensor) -> int:
    """Host work before an MTP step: the next tokens and the input streams into the static buffers."""

    n = len(next_tokens)
    if st.mtp_len + n > st.capacity:
        raise ValueError("MTP context past the cache capacity")
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = [int(t) for t in next_tokens]
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    if streams.data_ptr() != b.mtp_in.data_ptr():
        b.mtp_in[:n].copy_(streams, non_blocking=True)
    b.staged.record()
    return n


def mtp_compute(w: Weights, st: State, b: Buffers, n: int, *, last_only: bool = True) -> torch.Tensor:
    """The MTP head's GPU work on staged rows (capturable)."""

    c = w.cfg
    m = w.mtp
    glue.embed(b.ids[:n], *w.embed, c.hidden, copies=1, out=b.mtp_e[:n])
    en, xe = glue.rmsnorm(b.mtp_e[:n], m.norm_e, c.eps, out=b.mixed[:n], xs=b.xs_mixed[:n])
    _mm(en, m.fc_e, xe, b.mtp_eo[:n], b)
    hn, xh = glue.rmsnorm(b.mtp_in[:n], m.norm_h, c.eps, out=b.mtp_hn[:n], xs=b.mtp_xh[:n])
    _mm(hn.view(n * c.streams, c.hidden), m.fc_h, xh.view(n * c.streams, c.hidden // 32), b.mtp_hs[:n * c.streams], b)
    glue.add_streams(b.mtp_eo[:n], b.mtp_hs[:n * c.streams], b.h[:n], c.streams)
    pending = layer_forward(m.layer, w, st, b, n, None, mtp=True)
    finish(w, m.mixer, b, n, pending, logits=False)
    if last_only:
        head = w.head if w.draft_head is None else w.draft_head
        out = _mm(b.mixed[n - 1:n], head, b.xs_mixed[n - 1:n], b.logits[:1, :head.n], b)
        if w.comm is not None:
            candidates(w, b, out, 1, id_map=w.draft_ids, offset=int(w.meta["vocab_offset"]))
        return out
    return _mm(b.mixed[:n], w.head, b.xs_mixed[:n], b.logits[:n], b)


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], streams: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Rows (main-model streams [n, S*D] bf16, the tokens after them): logits of the last row [1, V] (or all
    rows), with the head's output streams in b.streams[:n]. Appends n entries to the MTP cache (the caller
    advances ``st.mtp_len``)."""

    n = mtp_stage(w, st, b, next_tokens, streams)
    return mtp_compute(w, st, b, n, last_only=last_only)
