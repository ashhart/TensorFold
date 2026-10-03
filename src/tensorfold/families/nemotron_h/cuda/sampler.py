"""Keyed sampler for CUDA graphs: a token depends only on (seed, position, logits), so serial and drafted rows agree."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import triton
import triton.language as tl

from tensorfold.engine.exact_sampling import MARGIN, Sampling

C1 = 0x9E3779B97F4A7C15
C2 = 0xD1B54A32D192ED03
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


@triton.jit
def _mix(x, m1, m2):
    x = x ^ (x >> 30)
    x = x * m1
    x = x ^ (x >> 27)
    x = x * m2
    return x ^ (x >> 31)


@triton.jit
def _keyed(VALS, IDS, META, OUT, SEED, FP, PROB, c1, c2, m1, m2, offset,
           C: tl.constexpr, CP: tl.constexpr, K: tl.constexpr, CUT: tl.constexpr, GREEDY: tl.constexpr = False,
           WRITE_PROB: tl.constexpr = False, MINP: tl.constexpr = False, CONF_T: tl.constexpr = False):
    r = tl.program_id(0)
    seed = tl.load(SEED)
    temp = tl.load(FP)
    top_p = tl.load(FP + 1)
    i = tl.arange(0, CP)
    ok = i < C
    v = tl.load(VALS + r * C + i, mask=ok, other=float("-inf")).to(tl.float64)
    ids = tl.load(IDS + r * C + i, mask=ok, other=2**40)
    # rank by (value desc, id asc), as the host's lexsort
    better = (v[None, :] > v[:, None]) | ((v[None, :] == v[:, None]) & (ids[None, :] < ids[:, None]))
    rank = tl.sum(tl.where(better & ok[None, :], 1, 0), axis=1)
    kept = (rank < K) & ok
    if GREEDY:
        tl.store(OUT + r, tl.sum(tl.where(rank == 0, ids, 0), axis=0).to(tl.int32))
        if WRITE_PROB:                     # the argmax's share of the top-K mass (temperature 1)
            top0 = tl.max(tl.where(kept, v, float("-inf")), axis=0)
            e0 = tl.where(kept, tl.exp(v - top0), 0.0)
            tl.store(PROB + r, (1.0 / tl.sum(e0, axis=0)).to(tl.float32))
        return
    scaled = v / temp
    top = tl.max(tl.where(kept, scaled, float("-inf")), axis=0)
    p = tl.where(kept, tl.exp(scaled - top), 0.0)
    total = tl.sum(tl.where(rank == 0, p, 0.0), axis=0)
    for k in tl.static_range(1, K):
        total += tl.sum(tl.where(rank == k, p, 0.0), axis=0)
    limit = K
    if CUT:
        run = tl.sum(tl.where(rank == 0, p, 0.0), axis=0) / total
        below = tl.where(run < top_p, 1, 0)
        for k in tl.static_range(1, K):
            run += tl.sum(tl.where(rank == k, p, 0.0), axis=0) / total
            below += tl.where(run < top_p, 1, 0)
        limit = below + 1
    if MINP:                               # the tokens within ln(min_p) of the top: a prefix of the rank order
        floor = top + tl.load(FP + 2)
        limit = tl.minimum(limit, tl.sum(tl.where(kept & (scaled >= floor), 1, 0), axis=0))
    pos = (tl.load(META) + r + 1 + offset).to(tl.uint64)
    x = _mix(seed.to(tl.uint64) + c1, m1, m2)
    x = _mix(x ^ (pos * c2), m1, m2)
    x = _mix(x ^ ids.to(tl.uint64), m1, m2)
    u = (x >> 11).to(tl.float64) * (2.0 ** -53) + 2.0 ** -54
    score = scaled - tl.log(-tl.log(u))
    score = tl.where(kept & (rank < limit), score, float("-inf"))
    best = tl.max(score, axis=0)
    first = tl.min(tl.where(score == best, rank, CP), axis=0)
    tok = tl.sum(tl.where(rank == first, ids, 0), axis=0)
    tl.store(OUT + r, tok.to(tl.int32))
    if WRITE_PROB and CONF_T:              # a draft drawn sharper: its share at the request's temperature FP[3]
        sc = v / tl.load(FP + 3)
        pc = tl.where(kept, tl.exp(sc - tl.max(tl.where(kept, sc, float("-inf")), axis=0)), 0.0)
        tl.store(PROB + r, (tl.sum(tl.where(rank == first, pc, 0.0), axis=0) / tl.sum(pc, axis=0)).to(tl.float32))
    elif WRITE_PROB:                       # the drawn token's share of the top-k mass
        tl.store(PROB + r, (tl.sum(tl.where(rank == first, p, 0.0), axis=0) / total).to(tl.float32))


class Params:
    """Sampling parameters live on the device so captured graphs serve any request; top_k is compiled in."""

    def __init__(self, device):
        self.seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.fp = torch.zeros(4, dtype=torch.float64, device=device)
        self.sampling: Sampling | None = None

    def set(self, sampling: Sampling | None) -> None:
        self.sampling = sampling
        if sampling is not None:
            self.seed.fill_(int(sampling.seed) & ((1 << 63) - 1))
            temp = max(float(sampling.temperature), 1e-6)
            self.fp.copy_(torch.tensor([temp, float(sampling.top_p), sampling.min_log, temp], dtype=torch.float64))


class DraftParams:
    """A draft's keyed rule: the request's top_k at tau x T, no top-p or min-p; the noise is the target's."""

    def __init__(self, params: Params, tau: float):
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"the draft temperature factor is in (0, 1], not {tau}")
        self.seed, self.tau = params.seed, float(tau)
        self.fp = torch.zeros(4, dtype=torch.float64, device=params.seed.device)
        self.sampling: Sampling | None = None
        self.source: Sampling | None = None             # the request's rule these follow

    def set(self, sampling: Sampling | None) -> None:
        self.sampling, self.source = None, sampling
        if sampling is not None and sampling.temperature > 0:
            temp = max(float(sampling.temperature), 1e-6)
            self.sampling = replace(sampling, temperature=self.tau * temp, top_p=1.0, min_p=0.0)
            self.fp.copy_(torch.tensor([max(self.tau * temp, 1e-6), 1.0, -math.inf, temp], dtype=torch.float64))


def keyed(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
          prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None, conf_t: bool = False):
    """Row r samples position meta[0] + r + 1 + offset; ``prob`` takes each draw's top-k share, a draft's confidence."""

    s = params.sampling
    greedy_mode = s is None or s.temperature <= 0
    rows, vocab = logits.shape
    top_k = 20 if greedy_mode else int(s.top_k)
    if not top_k:                                  # top_k off: the nucleus over the whole vocabulary
        return nucleus(logits, meta, params, out, offset=offset, prob=prob, id_map=id_map, conf_t=conf_t)
    count = min(vocab, top_k + MARGIN) if top_k else vocab
    if count > 256:
        raise ValueError("the GPU sampler takes top_k + margin <= 256 candidates")
    vals, ids = torch.topk(logits.float(), count, dim=-1, sorted=False)
    if id_map is not None:
        ids = id_map[ids]
    k = max(1, min(top_k if top_k else count, count))
    cut = (not greedy_mode) and 0.0 < float(s.top_p) < 1.0
    _keyed[(rows,)](vals, ids, meta, out, params.seed, params.fp, prob if prob is not None else out, C1, C2, M1, M2,
                    offset, C=count, CP=triton.next_power_of_2(count), K=k, CUT=cut, GREEDY=greedy_mode,
                    WRITE_PROB=prob is not None, MINP=(not greedy_mode) and float(s.min_p) > 0.0,
                    CONF_T=conf_t and not greedy_mode, num_warps=1)
    return out


def _signed(c: int) -> int:
    return c - (1 << 64) if c >= 1 << 63 else c


def _shr(x: torch.Tensor, k: int) -> torch.Tensor:
    return (x >> k) & ((1 << (64 - k)) - 1)       # a logical shift on int64 (uint64 bits)


def _mix_t(x: torch.Tensor) -> torch.Tensor:
    x = x ^ _shr(x, 30)
    x = x * _signed(M1)
    x = x ^ _shr(x, 27)
    x = x * _signed(M2)
    return x ^ _shr(x, 31)


def nucleus(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
            prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None, conf_t: bool = False):
    """``_keyed``'s rule with top_k off: rank the whole vocabulary by (value desc, id asc), cut at top_p, draw."""

    rows, vocab = logits.shape
    v = logits.double()
    ids = torch.arange(vocab, dtype=torch.int64, device=logits.device)
    ids = id_map.to(torch.int64) if id_map is not None else ids
    temp, top_p = params.fp[0], params.fp[1]
    scaled = v / temp
    order = torch.argsort(scaled, dim=-1, descending=True, stable=True)       # ties keep the lower id first
    ranked = torch.gather(scaled, 1, order)
    p = torch.exp(ranked - ranked[:, :1])
    run = torch.cumsum(p, dim=-1) / p.sum(dim=-1, keepdim=True)
    limit = (run < top_p).sum(dim=-1, keepdim=True) + 1
    if params.sampling is not None and params.sampling.min_p > 0.0:        # the kernel's min-p prefix
        limit = torch.minimum(limit, (ranked >= ranked[:, :1] + params.fp[2]).sum(dim=-1, keepdim=True))
    pos = (meta[0].to(torch.int64) + torch.arange(rows, device=logits.device) + 1 + offset)[:, None]
    x = _mix_t(params.seed.to(torch.int64) + _signed(C1))
    x = _mix_t(x ^ (pos * _signed(C2)))
    x = _mix_t(x ^ ids[order])
    u = _shr(x, 11).double() * 2.0 ** -53 + 2.0 ** -54
    score = ranked - torch.log(-torch.log(u))
    rank = torch.arange(vocab, device=logits.device)[None, :]
    score = torch.where(rank < limit, score, torch.full_like(score, float("-inf")))
    first = torch.argmax(score, dim=-1)                                        # the lowest rank among equal scores
    out.copy_(ids[order.gather(1, first[:, None])[:, 0]].to(out.dtype))
    if prob is not None:
        if conf_t:                                                             # the draw's share at temperature fp[3]
            p = torch.exp((ranked - ranked[:, :1]) * (temp / params.fp[3]))
        prob.copy_((p.gather(1, first[:, None])[:, 0] / p.sum(dim=-1)).to(prob.dtype))
    return out


def sample_candidates(vals: torch.Tensor, ids: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor,
                      *, offset: int = 0, prob: torch.Tensor | None = None, conf_t: bool = False):
    """As ``sample`` over candidate lists (R, C) holding each row's top top_k + MARGIN, such as the ranks' union."""

    s = params.sampling
    rows, count = vals.shape
    greedy_mode = s is None or s.temperature <= 0
    k = (20 if prob is not None else 1) if greedy_mode else max(1, min(int(s.top_k) if s.top_k else count, count))
    k = min(k, count)
    cut = (not greedy_mode) and 0.0 < float(s.top_p) < 1.0
    _keyed[(rows,)](vals, ids, meta, out, params.seed, params.fp, prob if prob is not None else out, C1, C2, M1, M2,
                    offset, C=count, CP=triton.next_power_of_2(count), K=k, CUT=cut, GREEDY=greedy_mode,
                    WRITE_PROB=prob is not None, MINP=(not greedy_mode) and float(s.min_p) > 0.0,
                    CONF_T=conf_t and not greedy_mode, num_warps=1)
    return out


def greedy(logits: torch.Tensor, out: torch.Tensor):
    out.copy_(torch.argmax(logits, dim=-1).to(torch.int32))
    return out


def sample(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
           prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None, conf_t: bool = False):
    """Greedy is the first argmax; with ``prob`` or ``id_map`` the keyed kernel picks the same token."""

    s = params.sampling
    if (s is None or s.temperature <= 0) and prob is None and id_map is None:
        return greedy(logits, out)
    return keyed(logits, meta, params, out, offset=offset, prob=prob, id_map=id_map, conf_t=conf_t)
