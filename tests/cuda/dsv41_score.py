"""Every row's logits of a DeepSeek-V4.1 prompt through the prompt path, for the model-level tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from tensorfold.families.deepseek_v41.cuda.forward import commit, compute, head, stage

ROWS = 256              # head rows a block


@torch.no_grad()
def prompt_logits(e: Any, ids: Sequence[int], rows: int = ROWS) -> torch.Tensor:
    """fp32 [len(ids), V / world] on the host: this rank's vocabulary shard of each row's logits, the rows read in
    prompt chunks of ``e.pbuf`` from position 0 and the head run in blocks of ``rows``.

    ``e`` holds ``w``, ``st``, ``pbuf`` (prompt-chunk buffers), ``hasher`` and ``reader``; the state is reset before
    and after. Every rank calls it in lockstep. No sample, draft or snapshot.
    """

    if not ids:
        raise ValueError("empty prompt")
    if rows < 1:
        raise ValueError(f"head blocks of {rows} rows")
    w, st, b = e.w, e.st, e.pbuf
    rows = min(rows, b.rows)
    out = torch.empty((len(ids), b.logits.shape[1]), dtype=torch.float32, pin_memory=True)
    block = torch.empty((rows, b.logits.shape[1]), dtype=torch.float32, device=b.logits.device)
    st.reset()
    try:
        for a in range(0, len(ids), b.rows):
            R = stage(w, st, b, list(ids[a:a + b.rows]), e.hasher, e.reader)
            compute(w, st, b, R, prompt=True, head_rows=0)
            for r in range(0, R, rows):
                n = min(rows, R - r)
                out[a + r:a + r + n].copy_(head(w, b, r, n, block[:n], prompt=True))
            commit(w, st, b, R, R)
    finally:
        st.reset()
    return out
