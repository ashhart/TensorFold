"""Speed settings of GLM-5.3-Flash's concurrent rounds (``multi``), none of which changes a reply's bits:

- TF_GLM_MULTI_SAMPLER=streams|packed: ``streams`` (the default) samples as ``multi.sample_streams`` did: every
  top-k stream's candidates in one all-gather, each nucleus (top_k off) stream through ``nucleus_rows``' own two
  gathers and host syncs. ``packed``: every stream's rows in two all-gathers whatever the samplings: the top-k
  candidates and the nucleus rows' maxima together, then every nucleus row's candidates and shard masses together
  (``sample_packed``: row for row the arithmetic ``nucleus_rows`` does; a stream whose nucleus runs past the
  candidates falls back to whole shards on its own, as it does alone).
- TF_GLM_MULTI_DEPTH=policy|joint|scale:A: how many drafts each stream verifies. ``policy`` (the default): each its
  own DFlash2 policy, as alone. ``joint``: every drafting stream's chain walked to its policy's most, then the rows
  of the round allocated across the streams by expected tokens per ms (``allocate``) on the startup-timed curve of
  the batched window's ms by rows plus TF_GLM_MULTI_OVERHEAD_MS (default 12) for the rest of a round.
  ``scale:A``: each stream's chain stop threshold raised to min(0.95, p * S**A) with S streams in the round.
- TF_GLM_MULTI_ASYNC=1: rank 0 sends each iteration's message without waiting for the GPU (a pinned staging copy;
  the same two all-gathers rank 1 reads), so its host runs ahead into the next round's host work.
- TF_GLM_MULTI_LONE=1 (default 0: off): a stream decoding alone (no other stream decoding or filling) moves once
  into the one-stream CUDA graphs' home (slot 0 at the pool's first rows) and verifies its windows through them
  (``verify.SerialVerify``, the single-stream forward) while they fit the graphs; a second stream joins it on the
  batched path at the next round boundary. Kept prompts in the home rows make way for it, so conversations taking
  turns re-read more of their prompts: off by default.
- TF_GLM_MULTI_PROFILE=N: rank 0 logs a round's time breakdown every N decode rounds, lone-stream rounds (one-stream
  graphs) and batched ones apart (``RoundProfile``), and at startup both paths' window ms for one stream.

All but the profile change the collectives or the drafts, so both ranks must agree: ``MultiSettings.code`` joins the
engine's two-rank settings comparison."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MultiSettings:
    sampler: str = "streams"
    depth: str = "policy"
    alpha: float = 0.0
    overhead_ms: float = 12.0
    async_msg: bool = False
    profile: int = 0
    lone: bool = False

    @classmethod
    def from_env(cls, env=None) -> MultiSettings:
        env = os.environ if env is None else env
        s = cls()
        s.sampler = (env.get("TF_GLM_MULTI_SAMPLER", "") or "streams").strip().lower()
        if s.sampler not in ("streams", "packed"):
            raise ValueError(f"TF_GLM_MULTI_SAMPLER: streams or packed, not {s.sampler!r}")
        depth = (env.get("TF_GLM_MULTI_DEPTH", "") or "policy").strip().lower()
        if depth.startswith("scale:"):
            try:
                s.alpha = float(depth[6:])
            except ValueError:
                s.alpha = -1.0
            if not 0 < s.alpha <= 4:
                raise ValueError(f"TF_GLM_MULTI_DEPTH=scale:A: A above 0 and at most 4, not {depth[6:]!r}")
            depth = "scale"
        if depth not in ("policy", "joint", "scale"):
            raise ValueError(f"TF_GLM_MULTI_DEPTH: policy, joint or scale:A, not {depth!r}")
        s.depth = depth
        try:
            s.overhead_ms = float(env.get("TF_GLM_MULTI_OVERHEAD_MS", "") or 12.0)
        except ValueError:
            s.overhead_ms = -1.0
        if not 0 <= s.overhead_ms <= 1000:
            raise ValueError("TF_GLM_MULTI_OVERHEAD_MS: 0 to 1,000 ms")
        a = (env.get("TF_GLM_MULTI_ASYNC", "") or "0").strip()
        if a not in ("0", "1"):
            raise ValueError(f"TF_GLM_MULTI_ASYNC: 0 or 1, not {a!r}")
        s.async_msg = a == "1"
        lone = (env.get("TF_GLM_MULTI_LONE", "") or "0").strip()
        if lone not in ("0", "1"):
            raise ValueError(f"TF_GLM_MULTI_LONE: 0 or 1, not {lone!r}")
        s.lone = lone == "1"
        p = (env.get("TF_GLM_MULTI_PROFILE", "") or "0").strip()
        if not p.isdecimal():
            raise ValueError(f"TF_GLM_MULTI_PROFILE: 0 (off) or the rounds between reports, not {p!r}")
        s.profile = int(p)
        return s

    def code(self) -> list[int]:
        """What both ranks must agree on: every setting that changes collectives or drafts, and whether the profile
        is on (its startup timing replays graphs on both ranks; only rank 0 logs)."""

        return [("streams", "packed").index(self.sampler), ("policy", "joint", "scale").index(self.depth),
                int(round(self.alpha * 1000)), int(round(self.overhead_ms * 1000)), int(self.async_msg),
                int(self.lone), int(self.profile > 0)]

    def describe(self) -> str:
        depth = f"scale:{self.alpha:g}" if self.depth == "scale" else self.depth
        return (f"sampler {self.sampler}, depth {depth}"
                f"{f' (overhead {self.overhead_ms:g} ms)' if self.depth == 'joint' else ''}, async messages "
                f"{'on' if self.async_msg else 'off'}, lone stream on the one-stream graphs "
                f"{'on' if self.lone else 'off'}{f', profile every {self.profile} rounds' if self.profile else ''}")


def scaled_confidence(p: float, streams: int, alpha: float) -> float:
    """TF_GLM_MULTI_DEPTH=scale:A: a stream's chain stop threshold with ``streams`` streams in the round."""

    if p <= 0 or alpha <= 0 or streams <= 1:
        return p
    return min(0.95, p * streams ** alpha)


def allocate(reach: Sequence[Sequence[float]], fixed: int, cost: Callable[[int], float], overhead: float,
             cap: int) -> list[int]:
    """TF_GLM_MULTI_DEPTH=joint: drafts to verify per stream. ``reach[s][k]``: the chance stream s's draft k is kept
    (the product of its chain's confidences up to k), non-increasing in k; ``fixed``: the round's rows that are
    decided (every pending token, copied drafts); ``cost(rows)``: the verify window's ms. Rows are added best reach
    first while the expected tokens per ms (``fixed`` streams' pending tokens count 1 each, a draft its reach) grows,
    at most ``cap`` rows in all. Deterministic in its inputs, so both ranks allocate alike."""

    take = [0] * len(reach)
    rows = fixed
    tokens = float(len(reach))
    best = tokens / (cost(rows) + overhead)
    while rows < cap:
        k = max(range(len(reach)), key=lambda s: (reach[s][take[s]] if take[s] < len(reach[s]) else -1.0, -s),
                default=None)
        if k is None or take[k] >= len(reach[k]):
            break
        gain = reach[k][take[k]]
        rate = (tokens + gain) / (cost(rows + 1) + overhead)
        if rate <= best:
            break
        take[k] += 1
        rows += 1
        tokens += gain
        best = rate
    return take


def reach_of(confs: Sequence[float]) -> list[float]:
    return [float(x) for x in np.cumprod(np.asarray(confs, dtype=np.float64))] if len(confs) else []


# -- sampling --------------------------------------------------------------------------------------------------------
def sample_packed(w, parts: Sequence[tuple]) -> list[list[int]]:
    """parts [(logits [R, V/world], positions, sampling)] -> each part's tokens, bit for bit what ``decode.sample_rows``
    gives each part alone, in two all-gathers for every part: (1) every top-k / greedy part's top candidates (values,
    ids as fp32 bits) and every nucleus part's row maxima (fp64 as two fp32 words), (2) every nucleus row's
    ``NUCLEUS`` best (value, id, mass), its shard's mass and width. Row by row the arithmetic is ``nucleus_rows``' and
    ``sample_rows``'; a nucleus part some row of which runs past the candidates reads whole shards on its own."""

    from tensorfold.cuda.sampling import MASS, NUCLEUS, _draw, _shares, comm_gather, one_rank
    from tensorfold.engine.exact_sampling import MARGIN, choose_rows

    gather = one_rank if w.comm is None else comm_gather(w.comm)
    world = 1 if w.comm is None else w.world
    out: list[list[int] | None] = [None] * len(parts)
    words, topk, nucl = [], [], []
    at = 0                                                         # each part's first word in the first gather
    for k, (logits, positions, sampling) in enumerate(parts):
        greedy = sampling is None or sampling.temperature <= 0
        if greedy or sampling.top_k:
            n = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
            vals, ids = torch.topk(logits.float(), n, dim=-1)
            ids = (ids + w.vocab_offset).to(torch.int32)
            words.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
            topk.append((k, logits.shape[0], n, at))
            at += logits.shape[0] * 2 * n
        else:
            scaled = logits.float().double() / max(float(sampling.temperature), 1e-6)
            words.append(scaled.max(dim=-1).values.contiguous().view(torch.float32).view(-1))
            nucl.append((k, scaled, at))
            at += 2 * logits.shape[0]
    if not words:
        return []
    flat = torch.cat(words) if len(words) > 1 else words[0]
    got1 = gather(flat.contiguous())                              # [world, words]
    host2 = None
    if nucl:
        # every rank's maxima of each nucleus row, then the rows' masses and candidates (nucleus_rows' arithmetic)
        scaled_all, tops = [], []
        for k, scaled, off in nucl:
            R = scaled.shape[0]
            m = got1[:, off:off + 2 * R].contiguous().view(torch.float64)        # [world, R]
            tops.append(m.reshape(world, R).max(dim=0).values)
            scaled_all.append(scaled)
        scaled = torch.cat(scaled_all) if len(scaled_all) > 1 else scaled_all[0]
        top = torch.cat(tops) if len(tops) > 1 else tops[0]
        mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
        rows, width = scaled.shape
        count = NUCLEUS
        vals, cols = torch.topk(scaled, min(count, width), dim=-1)
        ids = cols + int(w.vocab_offset)
        pad = count - vals.shape[1]
        if pad:
            vals = torch.cat([vals, vals.new_full((rows, pad), float("-inf"))], dim=1)
            ids = torch.cat([ids, ids.new_full((rows, pad), -1)], dim=1)
        kept = mass.gather(1, cols)
        kept = torch.cat([kept, kept.new_zeros((rows, pad))], dim=1) if pad else kept
        shard = torch.tensor([[width]], dtype=torch.int64, device=scaled.device).expand(rows, 1)
        packed = torch.cat([vals.view(torch.int64), ids, kept, mass.sum(dim=-1, keepdim=True), shard], dim=1)
        words2 = packed.contiguous().view(torch.float32).view(-1)
        host2 = gather(words2).reshape(-1, words2.numel()).view(torch.int64).reshape(-1, rows, 3 * count + 2)
        host2 = host2.cpu().numpy()
    host1 = got1.cpu()
    for k, R, n, at in topk:
        g = host1[:, at:at + R * 2 * n].reshape(host1.shape[0], R, 2 * n)
        values = torch.cat([g[r, :, :n] for r in range(g.shape[0])], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, n:].contiguous().view(torch.int32) for r in range(g.shape[0])],
                           dim=1).numpy().astype(np.int64)
        _, positions, sampling = parts[k]
        if sampling is None or sampling.temperature <= 0:
            order = np.lexsort((tokens, -values), axis=-1)
            out[k] = [int(tokens[i, order[i, 0]]) for i in range(R)]
        else:
            out[k] = choose_rows(values, tokens, positions, sampling)
    if nucl:
        r0 = 0
        for k, scaled_k, _ in nucl:
            R = scaled_k.shape[0]
            both = host2[:, r0:r0 + R]
            got = (np.ascontiguousarray(both[:, :, :count]).view(np.float64), both[:, :, count:2 * count],
                   both[:, :, 2 * count:3 * count], both[:, :, 3 * count], both[:, :, 3 * count + 1])
            _, positions, sampling = parts[k]
            drawn = _draw(got, positions, sampling)
            m_k = mass[r0:r0 + R]
            if drawn is None:              # a row's nucleus runs past the candidates: this part reads whole shards
                drawn = _draw(_shares(gather, scaled_k, m_k, int(got[4].max()), int(w.vocab_offset), None),
                              positions, sampling)
            out[k] = [token for token, _ in drawn]
            r0 += R
    return out


# -- the profile -------------------------------------------------------------------------------------------------------
class RoundProfile:
    """TF_GLM_MULTI_PROFILE=N (rank 0): where a decode round's wall time goes, averaged over the last N rounds, and
    the prompt chunks' share of the wall time between reports. Stages are host wall time between marks; ``verify``
    also has the window's GPU time from CUDA events, read after the sampler has waited for it anyway."""

    STAGES = ("plan", "message", "propose", "verify", "sample", "accept", "commit", "taps", "emit")

    def __init__(self, every: int, log=print, label: str = "") -> None:
        self.every, self.log, self.label = every, log, label
        self.cuda = torch.cuda.is_available()
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if self.cuda else None
        self._reset()
        self.t_report = time.perf_counter()

    def _reset(self) -> None:
        self.rounds = 0
        self.sums = dict.fromkeys(self.STAGES + ("wall", "gpu"), 0.0)
        self.rows = self.streams = self.tokens = self.drafted = 0
        self.fill_s = 0.0
        self.fills = 0

    def begin(self, t0: float | None = None) -> None:
        self.t0 = time.perf_counter() if t0 is None else t0
        self.t = self.t0

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.sums[stage] += now - self.t
        self.t = now

    def gpu_start(self) -> None:
        if self.cuda:
            self.ev[0].record()

    def gpu_stop(self) -> None:
        if self.cuda:
            self.ev[1].record()

    def fill(self, seconds: float) -> None:
        self.fills += 1
        self.fill_s += seconds

    def end(self, streams: int, rows: int, tokens: int, drafted: int) -> None:
        self.sums["wall"] += time.perf_counter() - self.t0
        if self.cuda:
            self.sums["gpu"] += self.ev[0].elapsed_time(self.ev[1]) / 1e3
        self.rounds += 1
        self.streams += streams
        self.rows += rows
        self.tokens += tokens
        self.drafted += drafted
        if self.rounds >= self.every:
            self.report()

    def report(self) -> None:
        n = max(self.rounds, 1)
        ms = {k: 1e3 * v / n for k, v in self.sums.items()}
        staged = sum(ms[k] for k in self.STAGES)
        span = time.perf_counter() - self.t_report
        self.log(f"[tensorfold] multi profile{' (' + self.label + ')' if self.label else ''}, {self.rounds} rounds: {self.streams / n:.2f} streams, "
                 f"{self.rows / n:.1f} rows ({self.drafted / n:.1f} drafted), {self.tokens / n:.2f} tokens a round; "
                 f"ms a round: wall {ms['wall']:.1f} = " +
                 " + ".join(f"{k} {ms[k]:.1f}" for k in self.STAGES) +
                 f" + other {ms['wall'] - staged:.1f}; verify GPU {ms['gpu']:.1f}; "
                 f"{self.tokens / max(self.sums['wall'], 1e-9):.1f} tok/s in rounds; prompt chunks {self.fills} "
                 f"({1e3 * self.fill_s:.0f} ms, {100 * self.fill_s / max(span, 1e-9):.1f}% of {span:.1f} s)")
        self._reset()
        self.t_report = time.perf_counter()
