"""Position-keyed CUDA target sampling with the Metal engine's host-side rule, for every CUDA family."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import math

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, uniform

MASS = 2.0 ** 40        # a token's share of the mass in fixed point: shard sums are exact, so ranks agree bit for bit
NUCLEUS = 1024          # candidates a rank reads for a top_k-off draw; a row they don't cover reads whole shards


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Sample each row from its logits and absolute position; serial and verify-window rows share this one path."""

    if logits.ndim != 2 or not logits.is_cuda or len(positions) != logits.shape[0]:
        raise ValueError("expected CUDA logits [rows, vocab] and one position per row")
    if sampling is None or sampling.temperature <= 0:
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]
    if not sampling.top_k:
        return nucleus_rows(logits, positions, sampling)
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
    out: list[list[int]] = [[] for _ in samplings]
    for s, smp in enumerate(samplings):
        greedy = smp is None or smp.temperature <= 0
        if not greedy and not smp.top_k:              # top_k off: the nucleus rule, stream by stream
            out[s] = nucleus_rows(logits[starts[s]:starts[s + 1]], positions[s], smp)
            continue
        groups.setdefault(0 if greedy else min(width, int(smp.top_k) + MARGIN), []).append(s)
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


def _stacked(gather: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """Every rank's copy of ``t`` [world, *t.shape], this rank's included, through ``gather`` of its float32 words."""

    words = t.contiguous().view(torch.float32).view(-1)
    return gather(words).reshape(-1, words.numel()).view(t.dtype).reshape(-1, *t.shape)


def one_rank(words: torch.Tensor) -> torch.Tensor:
    return words[None]


def comm_gather(comm) -> Callable[[torch.Tensor], torch.Tensor]:
    """``nucleus_rows``'s gather over a family's NCCL ``comm`` (``tensorfold.cuda.comm``)."""

    def gather(words: torch.Tensor) -> torch.Tensor:
        got = torch.empty((comm.world * words.numel(),), dtype=words.dtype, device=words.device)
        comm.all_gather(words, got)
        return got.view(comm.world, -1)

    return gather


def dist_gather(words: torch.Tensor) -> torch.Tensor:
    """``nucleus_rows``'s gather over torch.distributed (the 27B's two ranks)."""

    import torch.distributed as dist

    got = torch.empty((dist.get_world_size(), words.numel()), dtype=words.dtype, device=words.device)
    dist.all_gather_into_tensor(got, words)
    return got


def nucleus_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling, *, offset: int = 0,
                 id_map: torch.Tensor | None = None, gather: Callable = one_rank,
                 probs: list[float] | None = None) -> list[int]:
    """top_k off: the keyed draw over the top_p nucleus, then min_p, cut by fixed-point mass (``MASS``), so one
    rank's whole rows and several ranks' vocabulary shards (``gather`` stacks every rank's words, rank 0 first;
    ``offset`` or ``id_map`` gives a column's token id) draw the same tokens. Each rank reads its ``NUCLEUS`` top
    candidates and its shard's total mass; a row they don't cover reads every rank's whole shard. ``probs`` takes
    each drawn token's share of the mass."""

    scaled = logits.float().double() / max(float(sampling.temperature), 1e-6)
    top = _stacked(gather, scaled.max(dim=-1).values).max(dim=0).values           # every rank's maxima
    mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
    got = _shares(gather, scaled, mass, NUCLEUS, offset, id_map)
    drawn = _draw(got, positions, sampling)
    if drawn is None:                           # some row's nucleus runs past the candidates: every whole shard
        drawn = _draw(_shares(gather, scaled, mass, int(got[4].max()), offset, id_map), positions, sampling)
    if probs is not None:
        probs.extend(share for _, share in drawn)
    return [token for token, _ in drawn]


def _shares(gather, scaled, mass, count, offset, id_map):
    """Every rank's best (value, id, mass) a row, padded to ``count`` (value -inf, mass 0), its shard's mass and
    width: numpy [world, R, count] thrice, then [world, R] twice."""

    rows, width = scaled.shape
    vals, cols = torch.topk(scaled, min(count, width), dim=-1)
    ids = id_map[cols].to(torch.int64) if id_map is not None else cols + int(offset)
    pad = count - vals.shape[1]
    if pad:
        vals = torch.cat([vals, vals.new_full((rows, pad), float("-inf"))], dim=1)
        ids = torch.cat([ids, ids.new_full((rows, pad), -1)], dim=1)
    kept = mass.gather(1, cols)
    kept = torch.cat([kept, kept.new_zeros((rows, pad))], dim=1) if pad else kept
    shard = torch.tensor([[width]], dtype=torch.int64, device=scaled.device).expand(rows, 1)
    packed = torch.cat([vals.view(torch.int64), ids, kept, mass.sum(dim=-1, keepdim=True), shard], dim=1)
    both = _stacked(gather, packed).cpu().numpy()
    return (np.ascontiguousarray(both[:, :, :count]).view(np.float64), both[:, :, count:2 * count],
            both[:, :, 2 * count:3 * count], both[:, :, 3 * count], both[:, :, 3 * count + 1])


def _draw(got, positions, s: Sampling) -> list[tuple[int, float]] | None:
    """Each row's (token, its share of the mass) from every rank's candidates; None if a row needs whole shards."""

    vals, ids, mass, sums, widths = got
    cut = 0.0 < s.top_p < 1.0
    drawn = []
    for r, position in enumerate(positions):
        total = int(sums[:, r].sum())
        need = math.ceil(Fraction(s.top_p) * total) if cut else None
        floor = vals[:, r, :].max() + s.min_log                   # the min_p cut (-inf when off)
        for k in range(vals.shape[0]):                            # a rank's share must end inside its candidates
            real = int((ids[k, r] >= 0).sum())
            if real == widths[k, r]:                              # its whole shard
                continue
            order = np.lexsort((ids[k, r], -vals[k, r]))
            order = order[ids[k, r][order] >= 0]                  # the padding goes
            v = vals[k, r][order]
            if cut:                                               # its mass reaches the need above its last one
                at = np.nonzero(np.cumsum(mass[k, r][order]) >= need)[0]
                covered = len(at) > 0 and v[int(at[0])] > v[-1]
            else:                                                 # min_p alone: a candidate below its cut
                covered = s.min_p > 0.0 and bool((v < floor).any())
            if not covered:
                return None
        v, i, m = vals[:, r].reshape(-1), ids[:, r].reshape(-1), mass[:, r].reshape(-1)
        order = np.lexsort((i, -v))
        order = order[i[order] >= 0]
        v, i, m = v[order], i[order], m[order]
        keep = len(v)
        if cut:
            keep = int((np.cumsum(m) < need).sum()) + 1
        if s.min_p > 0.0:
            keep = min(keep, int((v >= floor).sum()))
        score = v[:keep] - np.log(-np.log(uniform(s.seed, int(position), i[:keep])))
        best = int(np.argmax(score))
        drawn.append((int(i[best]), float(m[best]) / total))
    return drawn
