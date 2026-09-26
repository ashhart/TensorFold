"""GLM-5.3-Flash decode on CUDA: prefill, serial decoding, and MTP-drafted decoding byte-identical to it.

Every emitted token is the keyed sample (``tensorfold.engine.exact_sampling``: seeded Gumbel over top-k/top-p, ties
by token id) of this engine's logits at its position, so a drafted round keeps a draft exactly when it equals
what serial decoding samples there. A round verifies the pending token and up to ``depth`` MTP drafts as one
chain window, keeps rows up to the first mismatch (``forward.commit``), then the MTP head absorbs the kept
positions and chains the next drafts. With two ranks each holds half of the vocabulary: both gather every
row's top candidates and draw with the same rule, so both ranks agree without a broadcast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from .forward import Buffers, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_stage
from .weights import Weights


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
        w.comm.all_gather(packed.view(-1), got)
        g = got.view(w.world, R, 2 * k).cpu()
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Each row's probability of its chosen token under the top-k / top-p distribution the sampler draws from
    (the draft's own confidence; greedy uses temperature 1 over the candidates)."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


class Engine:
    """Weights, one sequence's state, and buffers for windows (main model and MTP head)."""

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = 64,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = ()) -> None:
        self.w = w
        w.meta["long_context"] = long_context
        rows = max(max_rows, prefill_rows)
        self.rows = rows
        self.prefill_rows = prefill_rows
        self.buf = Buffers(w, rows, capacity)
        if taps:
            self.buf.set_taps(tuple(taps), w.cfg.hidden)         # before any graph capture
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.st = State(w, capacity, rows)
        self.last_hidden: torch.Tensor | None = None
        self.draft_n = w.head.n
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, graph_rows, graph_rows)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g = self.graphs.main.get((R, self.st.parity)) if self.graphs is not None and dense else None
        if g is not None:
            g.replay()
            return self.buf.logits[:R]
        return compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world]."""

        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        g = self.graphs.mtp.get(n) if self.graphs is not None and not self.mbuf.zero_first and dense else None
        if g is not None:
            g.replay()
            return self.mbuf.logits[:1, :self.draft_n]
        from .attention import CHUNK

        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // CHUNK),
                           host_pos=self.st.mtp_len)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def tap_rows(self, n: int) -> torch.Tensor:
        """The last forward's first n rows of DFlash2 taps, concatenated in layer order: [n, taps * D]."""

        return torch.cat([t[:n] for t in self.buf.taps], dim=1)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """The main model's rows the MTP head reads (after a forward that computed logits): the final-normed rows,
        as vLLM's GLM-5.3 MTP reads them."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions (main-model hidden rows [n, D], the tokens after them); logits of the last.
    More rows than the head's buffers hold go in chunks (rows never depend on their chunk-mates)."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb the kept positions, then chain up to ``count`` drafts for positions position, position + 1, ...
    ``confidence`` > 0: stop once the product of the drafts' own probabilities falls below it (the first draft is
    always kept), so a verify row is spent only on a draft likely to be accepted."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    """The committed state after ``ids``, for a later prompt that extends them. The KDA states and conv windows are
    copied; the attention caches (the model's, the MTP head's, DFlash2's) stay where they are, because a request
    resumed from here writes only past ``len(ids)``. ``pending``: the MTP input rows of the last committed
    positions the head has not absorbed yet (their next tokens come from the new prompt). ``mtp_len`` and
    ``drafter_end``: how far the MTP and DFlash2 caches are valid, -1 when they are not usable."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    drafter_end: int


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool,
                  drafter=None) -> Snapshot:
    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    return Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    st.mtp_len - st.mtp_drafted if mtp and pending is not None else -1,
                    drafter.context_end if drafter is not None else -1)


def restore(e: Engine, snap: Snapshot, drafter=None) -> None:
    st = e.st
    if st.cur:
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0
    if drafter is not None:
        drafter.context_end = snap.drafter_end
        drafter.pos_dev.fill_(snap.drafter_end)


# -- prefill ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
            resume: Snapshot | None = None) -> int:
    """Commit the prompt in chains of up to ``prefill_rows`` rows (the MTP cache absorbing every position whose
    next token is known; a DFlash2 ``drafter`` taking every position's taps), sample the first output token, and
    keep the last hidden row for the first draft. ``resume``: start from that snapshot, whose ids begin the prompt
    (rows never depend on their chunk-mates, so the state ends with the bits of a prefill from the start)."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.buf
    use_mtp = mtp and w.mtp is not None
    begin = 0
    if resume is None:
        e.reset()
        if drafter is not None:
            drafter.reset()
    else:
        begin = len(resume.ids)
        if begin >= len(prompt) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        restore(e, resume, drafter)
        if use_mtp:
            k = resume.pending.shape[0]
            absorb(e, resume.pending, list(prompt[begin - k + 1:begin + 1]))
    last = None
    for start in range(begin, len(prompt), e.prefill_rows):
        chunk = list(prompt[start:start + e.prefill_rows])
        R = len(chunk)
        logits = compute(w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, R), host_pos=st.pos)
        last = logits[R - 1:R].clone()
        e.last_hidden = e.main_hidden(slice(R - 1, R)).clone()
        if use_mtp:
            nxt = list(prompt[start + 1:start + R + 1])
            if nxt:
                absorb(e, e.main_hidden(slice(0, len(nxt))), nxt)
        if drafter is not None:
            drafter.add_taps(e.tap_rows(R))
        commit(w, st, b, R, R)
    return e.sample(last, [len(prompt)], sampling)[0]


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    arms: str = ""                          # auto_decode: the drafter of each round, "m" (MTP) or "f" (DFlash2)
    pending: torch.Tensor | None = None     # auto_decode: MTP input rows of committed positions not absorbed yet

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = e.forward([out[-1]])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Drafts a round: fixed, or from the running acceptance (the Flash Next engine's rule)."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.rate = 0.8

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify the pending token and its MTP drafts in one window, keep up to the first mismatch, draft again.
    Starts from the state ``prefill`` left (the MTP cache holds every prompt position but the last)."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    drafts = draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence) if depth > 0 else []
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                        policy.confidence) if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


@torch.no_grad()
def dflash_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                  policy: DepthPolicy | None = None, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """``mtp_decode`` with DFlash2 drafts: a round verifies the pending token and the drafter's chain for the
    positions after it, keeps up to the first mismatch, and the drafter takes the kept rows' taps. Starts from
    ``prefill(..., drafter=drafter)``."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy(3, fixed=True)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        drafts = drafter.propose(out[-1], depth, sampling, policy.confidence) if depth > 0 else []
        t1 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        drafter.add_taps(e.tap_rows(keep))
        t5 = time.perf_counter()
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


# -- drafter chosen per request ------------------------------------------------------------------------------------
def _other(arm: str) -> str:
    return "f" if arm == "m" else "m"


class DrafterChoice:
    """Which drafter a round uses, MTP chains ("m") or DFlash2 blocks ("f"), from the tokens each has committed per
    millisecond in this request. The milliseconds come from ``costs`` (a verify window of R rows, an MTP step, a
    DFlash2 block, the catch-up of rows a drafter missed), timed at load and made the same on both ranks, so both
    ranks choose alike without exchanging anything; committed tokens are the same on both by construction.

    The first ``explore`` rounds use ``first``, the next ``explore`` the other drafter; then the one with the higher
    rate over its last ``window`` rounds, switching only for a rate ``margin`` higher, and one round of the other
    every ``every`` rounds so its rate stays current."""

    def __init__(self, costs: dict, *, first: str, explore: int = 2, every: int = 8, margin: float = 0.03,
                 window: int = 6) -> None:
        self.costs = costs
        self.first = first
        self.explore, self.every, self.margin, self.window = explore, every, margin, window
        self.rounds: list[tuple[str, int, float]] = []       # (drafter, tokens committed, model ms)
        self.choice = first
        self.run = 0

    def cost(self, arm: str, rows: int, steps: int, backlog: int) -> float:
        """Model ms of a round: its verify window, then MTP steps (``steps`` head runs) or a DFlash2 block, plus
        the catch-up of ``backlog`` rows."""

        c = self.costs
        verify = c["verify"][min(rows, len(c["verify"])) - 1]
        if arm == "m":
            return verify + c["mtp"] + c["mtp_step"] * max(steps - 1, 0) + c["mtp_row"] * max(backlog - 1, 0)
        return verify + c["block"] + c["taps_row"] * backlog

    def rate(self, arm: str) -> float | None:
        rs = [r for r in self.rounds if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self) -> str:
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(_other(cur))
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin      # no bias at the first choice
        if ro is not None and (rc is None or ro > rc * need):
            self.choice = cur = _other(cur)
            self.run = 0
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(cur)
        self.run += 1
        return cur

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        self.rounds.append((arm, keep, self.cost(arm, rows, steps, backlog)))


@torch.no_grad()
def auto_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                choice: DrafterChoice | None,
                m_policy: DepthPolicy, f_policy: DepthPolicy, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Drafted decoding where ``choice`` picks the drafter of each round (MTP only when it is None). Each
    drafter keeps a backlog of the committed rows it has not taken (MTP input rows and their next tokens; DFlash2
    taps) and takes them when it next drafts, so switching costs one catch-up step. Drafts only propose, so the
    reply equals serial decoding whatever the choice. Starts from ``prefill(..., mtp=True, drafter=drafter)``."""

    w, st, b = e.w, e.st, e.buf
    cap = e.rows * 4
    m_rows = torch.empty((cap, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    m_rows[:1].copy_(e.last_hidden)
    m_next: list[int] = [pending]
    f_taps = None
    n_f = 0
    if drafter is not None:
        f_taps = torch.empty((cap, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    arms: list[str] = []
    last = {"m": (0, 0), "f": (0, 0)}
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        arm = choice.pick() if choice is not None else "m"
        room = count - len(out)
        t0 = time.perf_counter()
        if arm == "m":
            backlog = len(m_next)
            depth = max(1, min(m_policy.next(*last["m"]), room))
            drafts = draft(e, m_rows[:backlog], m_next, st.pos + 1, depth, sampling, m_policy.confidence)
            steps = 1 + st.mtp_drafted
            m_next = []
        else:
            backlog = n_f
            if n_f:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            depth = max(1, min(f_policy.next(*last["f"]), room))
            drafts = drafter.propose(out[-1], depth, sampling, f_policy.confidence)
            steps = 0
        t1 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        # the kept rows join both backlogs (a full backlog is taken first)
        if len(m_next) + keep > cap:
            absorb(e, m_rows[:len(m_next)], m_next)
            m_next = []
        m_rows[len(m_next):len(m_next) + keep].copy_(e.main_hidden(slice(0, keep)))
        m_next.extend(sampled[:keep])
        if drafter is not None:
            if n_f + keep > cap:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            f_taps[n_f:n_f + keep].copy_(e.tap_rows(keep))
            n_f += keep
        t5 = time.perf_counter()
        if choice is not None:
            choice.record(arm, R, steps, backlog, keep)
        last[arm] = (len(drafts), keep - 1)
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        arms.append(arm)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
    _sync(w)
    seconds = time.perf_counter() - start
    if drafter is not None and n_f:
        drafter.add_taps(f_taps[:n_f])          # DFlash2's context ends where the committed rows end
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, stages, depths, keeps, "".join(arms),
                        m_rows[:len(m_next)])
