"""GLM-5.3-Flash's MTP head (the nextn layer) on CUDA, through the same row-invariant kernels as the model.

At position i it reads the main model's hidden row h_i and the embedding of token i + 1, the way vLLM's GLM-5.3
MTP reads them: h is the final-normed row, chained drafts recycle the head's shared_head.norm output, and the
embedding at position 0 is zeroed (the final-normed row predicted the next token 0.739 of the time against 0.696
for the mean of the streams before the final norm):

    x = eh_proj([enorm(embed(t_{i+1})) | hnorm(h_i)])
    x = x + attn(input_layernorm(x))            DSA over the head's own cache, plain residual (no hyper-connections)
    x = x + moe(post_attention_layernorm(x))
    logits = lm_head(shared_head.norm(x))

Its output x feeds the next chained draft as h. The head's attention cache holds one entry per absorbed
position (``State.mtp_len``); chained drafts append entries that the next absorb trims.
"""

from __future__ import annotations

from typing import Sequence

import torch

from . import glue, qmm
from .forward import Buffers, State, check_room, dsa_block, moe_block
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
                nch: int | None = None, host_pos: int | None = None) -> torch.Tensor:
    """The MTP head's GPU work on staged rows (capturable)."""

    c = w.cfg
    m = w.mtp
    D = c.hidden
    glue.embed(b.ids[:n], w.embed, D, 1, b.me[:n])
    if b.zero_first:
        b.me[0].zero_()
    glue.rmsnorm(b.me[:n], m.enorm, c.eps, b.mcat[:n, :D])
    glue.rmsnorm(b.hin[:n], m.hnorm, c.eps, b.mcat[:n, D:])
    qmm.matmul(b.mcat[:n], m.eh, qmm.group_sums(b.mcat[:n], b.mxs[:n]), out=b.mx[:n], part=b.sk)
    layer = m.layer
    glue.rmsnorm(b.mx[:n], layer.in_norm, c.eps, b.normed[:n], b.xs[:n])
    g = dsa_block(layer, w, st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, b, n, nch,
                  st.index[-1] if st.index is not None else None, host_pos)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    glue.rmsnorm(b.mx[:n], layer.post_norm, c.eps, b.normed[:n], b.xs[:n])
    g = moe_block(layer, w, b, n)
    glue.residual_add(b.mx[:n], b.mx[:n], g)
    lo = n - 1 if last_only else 0
    k = n - lo
    glue.rmsnorm(b.mx[lo:n], m.norm, c.eps, b.fnormed[:k], b.fxs[:k])
    return qmm.matmul(b.fnormed[:k], w.head, b.fxs[:k], out=b.logits[:k, :w.head.n], part=b.sk)


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], hidden: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Rows (hidden [n, D] bf16, the tokens after them): logits of the last row [1, V/world] (or all rows), with
    the head's output rows in b.mx[:n]. Writes n entries at the head's cache slots mtp_len.. (the caller
    advances ``st.mtp_len``)."""

    n = mtp_stage(w, st, b, next_tokens, hidden)
    from .attention import CHUNK

    return mtp_compute(w, st, b, n, last_only=last_only, nch=-(-(st.mtp_len + n) // CHUNK), host_pos=st.mtp_len)
