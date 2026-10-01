"""Last-row logits of a GLM prompt, one vocabulary shard, with no sample."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .forward import chunks_for, commit, compute, stage


@torch.no_grad()
def prompt_logits(e: Any, prompt: Sequence[int]) -> list[float]:
    """Last-row logits of a prompt, as one vocabulary shard. No sample, draft, or grammar step."""

    if not prompt:
        raise ValueError("empty prompt")
    w, st, b = e.w, e.st, e.pbuf
    e.reset()
    last = None
    try:
        for start in range(0, len(prompt), e.prefill_rows):
            chunk = list(prompt[start:start + e.prefill_rows])
            rows = len(chunk)
            last = compute(
                w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, rows), host_pos=st.pos).clone()
            commit(w, st, b, rows, rows)
        values = [float(item) for item in last.reshape(-1).float().cpu().tolist()]
    finally:
        e.reset()
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("label scoring produced a non-finite logit")
    return values
