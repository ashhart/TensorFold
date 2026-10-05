"""Position-keyed CUDA target sampling with the Metal engine's host-side rule, for every CUDA family."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
import math

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, _mix, choose_rows, uniform, uniform_rows

MASS = 2.0 ** 40        # a token's share of the mass in fixed point: shard sums are exact, so ranks agree bit for bit
NUCLEUS = 1024          # candidates a rank reads for a top_k-off draw; a row they don't cover reads whole shards
SLACK = 2.0 ** -30      # relative bound far above any gap between device and host float64 logs (a few ulps)
DIGITS = (13, 13, 13, 13, 12)   # bits of the 64-bit (value, id) order each radix pass of the top_p cut resolves
TMAX = 2.0 ** 800       # up to this temperature float32 logits order and tie exactly as their float64 scaled values


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
    """top_k off: the keyed draw over the top_p nucleus then min_p, cut by fixed-point mass, the same on each shape."""

    t = max(float(sampling.temperature), 1e-6)
    scaled = logits.float().double() / t
    peak = logits.float().amax(dim=-1).double() / t             # scaled's row maxima: dividing by t is monotone
    top = _stacked(gather, peak).max(dim=0).values             # every rank's maxima
    mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
    drawn = _keyed(gather, logits, scaled, top, mass, positions, sampling, offset, id_map)
    if drawn is None:
        got = _shares(gather, scaled, mass, NUCLEUS, offset, id_map)
        drawn = _draw(got, positions, sampling)
    if drawn is None:                           # some row's nucleus runs past the candidates: every whole shard
        drawn = _draw(_shares(gather, scaled, mass, int(got[4].max()), offset, id_map), positions, sampling)
    if probs is not None:
        probs.extend(share for _, share in drawn)
    return [token for token, _ in drawn]


def _shares(gather, scaled, mass, count, offset, id_map):
    """Every rank's padded top (value, id, mass) per row, plus the shard's mass and width sums."""

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


def _keyed(gather, logits, scaled, top, mass, positions, s: Sampling, offset, id_map) -> list[tuple[int, float]] | None:
    """``_draw``'s keyed argmax over the top_p nucleus and {v >= floor}, each rank's best device score rescored.

    A rank's other tokens score on the host at most its runner-up's device score plus the device-host gap, so a host
    best above each such bound plus ``SLACK`` is the whole-shard draw; otherwise None (``_draw`` decides)."""

    cut = None
    if 0.0 < s.top_p < 1.0:
        t = max(float(s.temperature), 1e-6)
        if t > TMAX:
            return None
        cut = _cut(gather, logits, t, mass, s.top_p, offset, id_map)
    floor = top + s.min_log
    if scaled.is_cuda:
        from .keyed_draw import best_two

        with np.errstate(over="ignore"):
            key = _mix(np.uint64(s.seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
            keys = _mix(key ^ (np.asarray(positions).astype(np.uint64) * np.uint64(0xD1B54A32D192ED03)))
        keys = torch.from_numpy(keys.view(np.int64)).to(scaled.device)
        f, i = best_two(scaled, keys, floor, int(offset), id_map, cut)
    else:
        f, i = _best_two(scaled, positions, s, floor, offset, id_map, cut)
    packed = torch.cat([f.view(torch.int64), i[:, 1:], mass.gather(1, i[:, :1]), mass.sum(dim=-1, keepdim=True)], 1)
    both = _stacked(gather, packed).cpu().numpy()
    first_np, second_np, vals_np = (np.ascontiguousarray(both[:, :, c]).view(np.float64) for c in range(3))
    ids_np, mass_np, sums = both[:, :, 3], both[:, :, 4], both[:, :, 5]
    drawn = []
    for r, position in enumerate(positions):
        live = first_np[:, r] > -math.inf                     # a rank with no token at or above the min_p floor
        v, i, m = vals_np[live, r], ids_np[live, r], mass_np[live, r]
        if not len(v):
            return None
        order = np.lexsort((i, -v))
        v, i, m = v[order], i[order], m[order]
        score = v - np.log(-np.log(uniform(s.seed, int(position), i)))
        best = int(np.argmax(score))
        if not math.isfinite(score[best]):
            return None
        for last in second_np[:, r].tolist():                 # a rank's other tokens cannot reach the best
            if last > -math.inf and not score[best] > last + SLACK * (1.0 + abs(last)):
                return None
        drawn.append((int(i[best]), float(m[best]) / int(sums[:, r].sum())))
    return drawn


def _cut(gather, logits, t, mass, top_p, offset, id_map) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's last nucleus token (its scaled value, its id): a radix select over every rank's exact int64 mass."""

    rows, width = logits.shape
    ids = id_map[:width].to(torch.int64) if id_map is not None else \
        torch.arange(offset, offset + width, device=logits.device)
    count, pick = _count, _pick
    if logits.is_cuda:
        from . import keyed_draw

        count, pick = keyed_draw.count, keyed_draw.pick
    part = torch.zeros(rows, dtype=torch.int64, device=logits.device)     # the boundary key's bits so far
    below = torch.zeros_like(part)                                        # every rank's mass before them
    need, shift = None, 64
    for digit in DIGITS:
        shift -= digit
        hist = _stacked(gather, count(logits, ids, mass, part, shift, digit))
        if need is None:                                                  # the first pass counts every token
            need = torch.tensor([math.ceil(Fraction(top_p) * total) for total in hist.sum(dim=(0, 2)).tolist()],
                                dtype=torch.int64, device=logits.device)
        part, below = pick(hist, need, below, part, shift, digit)
    value = -1 - (part >> 32)
    value = (value ^ ((value >> 31) & 0x7FFFFFFF)).to(torch.int32).view(torch.float32).double() / t
    return value, part & 0xFFFFFFFF


def _order(logits, ids) -> torch.Tensor:
    """Each token's int64 key, ascending in the order (-v, id): float32 logits order and tie as their scaled values."""

    x = logits.float()
    bits = torch.where(x == 0, 0.0, x).view(torch.int32).to(torch.int64)  # -0 ties +0, as in scaled
    return (-1 - (bits ^ ((bits >> 31) & 0x7FFFFFFF))) * 2 ** 32 + ids


def _count(logits, ids, mass, part, shift, digit) -> torch.Tensor:
    """int64 [rows, 2**digit]: the mass of the keys matching ``part`` above ``shift + digit``, by their next digit."""

    key = _order(logits, ids)
    if shift + digit == 64:                                               # the top digit is signed
        return torch.zeros((key.shape[0], 1 << digit), dtype=torch.int64).scatter_add_(
            1, (key >> shift) + (1 << digit - 1), mass)
    weight = torch.where((key >> shift + digit) == (part >> shift + digit)[:, None], mass, 0)
    return torch.zeros((key.shape[0], 1 << digit), dtype=torch.int64).scatter_add_(
        1, (key >> shift) & (1 << digit) - 1, weight)


def _pick(hist, need, below, part, shift, digit) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank's ``_count`` summed: the first digit whose running mass reaches ``need``, appended to ``part``."""

    hist = hist.sum(dim=0)
    run = below[:, None] + hist.cumsum(dim=1)
    pick = (run < need[:, None]).sum(dim=1, keepdim=True)
    below = (run - hist).gather(1, pick)[:, 0]
    return part + (pick[:, 0] - (1 << digit - 1 if shift + digit == 64 else 0)) * 2 ** shift, below


def _best_two(scaled, positions, s: Sampling, floor, offset, id_map, cut=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``keyed_draw.best_two`` for host tensors, from the host's own uniforms."""

    rows, width = scaled.shape
    ids = id_map[:width].to(torch.int64) if id_map is not None else torch.arange(offset, offset + width)
    u = uniform_rows(s.seed, np.asarray(positions), np.broadcast_to(ids.numpy(), (rows, width)))
    score = scaled - torch.log(-torch.log(torch.from_numpy(u)))
    score.masked_fill_(scaled < floor[:, None], -math.inf)
    if cut is not None:                                       # the order (-v, id) up to each row's last kept token
        last, at = cut[0][:, None], cut[1][:, None]
        score.masked_fill_(~((scaled > last) | ((scaled == last) & (ids <= at))), -math.inf)
    best, cols = torch.topk(score, min(2, width), dim=-1)
    if width == 1:                                            # no runner-up
        best = torch.cat([best, torch.full_like(best, -math.inf)], dim=1)
    col = cols[:, :1]
    return torch.cat([best, scaled.gather(1, col)], dim=1), torch.cat([col, ids[col]], dim=1)
