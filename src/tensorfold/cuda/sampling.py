"""Position-keyed CUDA target sampling with the Metal engine's host-side rule, for every CUDA family."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Sample each row from its logits and absolute position; serial and verify-window rows share this one path."""

    if logits.ndim != 2 or not logits.is_cuda or len(positions) != logits.shape[0]:
        raise ValueError("expected CUDA logits [rows, vocab] and one position per row")
    if sampling is None or sampling.temperature <= 0:
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]
    width = logits.shape[1]
    count = min(width, int(sampling.top_k) + MARGIN) if sampling.top_k else width
    if count < width:
        values, ids = torch.topk(logits.float(), count, dim=-1, sorted=False)
        values_np = values.cpu().numpy()
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False)
    else:
        values_np = logits.float().cpu().numpy()
        ids_np = np.broadcast_to(np.arange(width, dtype=np.int64), values_np.shape)
    return choose_rows(values_np, ids_np, positions, sampling)


def sample_streams(logits: torch.Tensor, starts: Sequence[int], positions: Sequence[Sequence[int]],
                   samplings: Sequence[Sampling | None]) -> list[list[int]]:
    """``sample_rows`` for several streams, grouped by candidate count so each row gets its own stream's call."""

    groups: dict[int, list[int]] = {}
    width = logits.shape[1]
    for s, smp in enumerate(samplings):
        greedy = smp is None or smp.temperature <= 0
        count = 0 if greedy else (min(width, int(smp.top_k) + MARGIN) if smp.top_k else width)
        groups.setdefault(count, []).append(s)
    launched = []
    for count, members in groups.items():
        rows = torch.cat([logits[starts[s]:starts[s + 1]] for s in members]) if len(members) > 1 \
            else logits[starts[members[0]]:starts[members[0] + 1]]
        if count == 0:
            launched.append((count, members, rows.argmax(dim=-1), None))
        elif count < width:
            values, ids = torch.topk(rows.float(), count, dim=-1, sorted=False)
            launched.append((count, members, ids, values))
        else:
            launched.append((count, members, None, rows.float()))
    out: list[list[int]] = [[] for _ in samplings]
    for count, members, ids, values in launched:
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False) if ids is not None else None
        values_np = values.cpu().numpy() if values is not None else None
        row = 0
        for s in members:
            n = starts[s + 1] - starts[s]
            if count == 0:
                out[s] = [int(x) for x in ids_np[row:row + n]]
            else:
                v = values_np[row:row + n]
                i = ids_np[row:row + n] if ids_np is not None else np.broadcast_to(np.arange(width, dtype=np.int64),
                                                                                   v.shape)
                out[s] = choose_rows(v, i, positions[s], samplings[s])
            row += n
    return out
