"""Flash Next decode on CUDA: prefill, serial decoding, and MTP-drafted decoding that stays byte-identical to it.

Every emitted token is the keyed sample (``tensorfold.engine.exact_sampling``: seeded Gumbel over top-k/top-p,
ties by token id) of this engine's logits at its position, so a drafted round keeps a draft exactly when it
equals what serial decoding samples there. A round verifies the pending token and up to ``depth`` MTP drafts
as one chain window, keeps rows up to the first mismatch (``forward.commit``), then the MTP head absorbs the
kept positions and chains the next drafts (drafts are sampled with the same keyed sampler at their positions).
A chain ends before a draft the head gives less than ``confidence``: a rejected draft costs a verify row, and
drafts change speed only, never the output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import CONFIDENCE, DEPTH
from .forward import CAND, Buffers, State, commit, forward
from .mtp import mtp_forward
from .weights import Weights


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Each row from its own logits and absolute position: CUDA picks the candidates (argmax, or the top-k plus
    a margin), the host's keyed rule draws, ties by token id. A serial row and the same row in a verify window
    go through this one function."""

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


def sample_mapped(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  id_map: torch.Tensor) -> list[int]:
    """Rows of logits over a token subset (column j is token id_map[j]) -> tokens, with the keyed rule on the real
    ids (the draft head over the draft vocabulary)."""

    if sampling is None or sampling.temperature <= 0:
        return [int(t) for t in id_map[logits.argmax(dim=-1)].cpu().tolist()]
    k = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    vals, idx = torch.topk(logits.float(), k, dim=-1, sorted=False)
    return choose_rows(vals.cpu().numpy(), id_map[idx].cpu().numpy().astype(np.int64), positions, sampling)


def tp_sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                   offset: int = 0, id_map: torch.Tensor | None = None, with_prob: bool = False):
    """Tensor parallel: each rank holds a slice of the vocabulary. Every rank sends its top candidates (value,
    global id) to all ranks; each rank then draws with the same keyed rule on the union, so all ranks agree.
    ``with_prob``: also each chosen token's probability at temperature 1 (the ranks' log-sum-exps travel with
    the candidates), returned as (tokens, probabilities)."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if greedy:
        # argmax takes the first (lowest-id) maximum whatever the row count; topk promises no order among ties
        ids = logits.argmax(dim=-1, keepdim=True)
        vals = torch.gather(logits, 1, ids).float()
    else:
        vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (id_map[ids] if id_map is not None else ids + int(offset)).to(torch.int32)
    parts = [vals, ids.view(torch.float32)]
    if with_prob:
        parts.append(torch.logsumexp(logits.float(), dim=-1, keepdim=True))
    packed = torch.cat(parts, dim=1).contiguous()
    width = packed.shape[1]
    world = int(w.meta["world"])
    got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=logits.device)
    w.comm.all_gather(packed.view(-1), got)
    g = got.view(world, R, width).cpu()
    values = torch.cat([g[r, :, :k] for r in range(world)], dim=1).numpy().astype(np.float32)
    tokens = torch.cat([g[r, :, k:2 * k].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
    tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * k].numpy().astype(np.float64)                       # [world, R]
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def choose_gathered(w: Weights, cand_all: torch.Tensor, R: int, positions: Sequence[int], sampling: Sampling | None,
                    with_prob: bool = False):
    """Tensor parallel: tokens from the candidates a step gathered in its graph (``forward.candidates``), with the
    keyed rule on the union of the ranks' candidates; ``with_prob`` also returns each token's probability."""

    world, width = int(w.meta["world"]), 2 * CAND + 1
    g = cand_all[:world * R * width].view(world, R, width).cpu().numpy()
    values = np.concatenate([g[r, :, :CAND] for r in range(world)], axis=1).astype(np.float32)
    tokens = np.concatenate([np.ascontiguousarray(g[r, :, CAND:2 * CAND]).view(np.int32) for r in range(world)],
                            axis=1).astype(np.int64)
    if sampling is None or sampling.temperature <= 0:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * CAND].astype(np.float64)
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def _gathered_fits(sampling: Sampling | None) -> bool:
    """Whether a step's gathered candidates (CAND a rank) cover the sampler's top-k plus its margin."""

    return sampling is None or sampling.temperature <= 0 or (bool(sampling.top_k) and sampling.top_k + MARGIN <= CAND)


class Engine:
    """Weights, one sequence's state, and buffers for windows (main model and MTP head)."""

    def __init__(self, w: Weights, *, capacity: int = 4096, max_rows: int = 8, prefill_rows: int = 64,
                 graphs: bool = False) -> None:
        self.w = w
        rows = max(max_rows, prefill_rows)
        self.capacity = capacity
        self.rows = rows
        self.buf = Buffers(w, rows, capacity)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.st = State(w, capacity, rows)
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, max_rows=max_rows)

    def reset(self) -> None:
        self.st.reset(self.w)

    def twin(self) -> "Engine":
        """An engine over the same weights and scratch buffers with its own committed state, no CUDA graphs and no
        MTP head: serial requests decode there and leave this engine's sequence, and a prefix cache over it, as
        they are. Requests run one at a time, so the scratch buffers are free between them."""

        other = object.__new__(Engine)
        other.w, other.capacity, other.rows = self.w, self.capacity, self.rows
        other.buf, other.mbuf, other.graphs = self.buf, None, None
        other.st = State(self.w, self.capacity, self.rows)
        return other

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A decode step's forward (a CUDA graph when enabled): logits [R, V]."""

        if self.graphs is not None:
            return self.graphs.forward(tokens)
        return forward(self.w, self.st, self.buf, tokens)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False) -> list[int]:
        """Rows of logits at their positions -> tokens (``draft``: logits of the MTP's draft head)."""

        mapped = draft and self.w.draft_ids is not None
        if self.w.comm is not None:
            b = self.mbuf if draft else self.buf
            if logits.data_ptr() == b.logits.data_ptr() and _gathered_fits(sampling):
                return choose_gathered(self.w, b.cand_all, logits.shape[0], positions, sampling)
            return tp_sample_rows(self.w, logits, positions, sampling, offset=self.w.meta["vocab_offset"],
                                  id_map=self.w.draft_ids if mapped else None)
        if mapped:
            return sample_mapped(logits, positions, sampling, self.w.draft_ids)
        return sample_rows(logits, positions, sampling)

    def sample_draft(self, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
        """The MTP head's draft at ``position`` (keyed like every sample) and its probability at temperature 1
        under the head's distribution: the confidence that ends a draft chain early (speed only)."""

        w = self.w
        mapped = w.draft_ids is not None
        if w.comm is not None:
            if logits.data_ptr() == self.mbuf.logits.data_ptr() and _gathered_fits(sampling):
                toks, probs = choose_gathered(w, self.mbuf.cand_all, 1, [position], sampling, with_prob=True)
            else:
                toks, probs = tp_sample_rows(w, logits[:1], [position], sampling, offset=w.meta["vocab_offset"],
                                             id_map=w.draft_ids if mapped else None, with_prob=True)
            return toks[0], probs[0]
        if mapped and getattr(self, "_draft_host", None) is None:
            self._draft_host = w.draft_ids.cpu().numpy()
        row = logits[:1].float()
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        if sampling is None or sampling.temperature <= 0:
            top, col = row.max(dim=-1, keepdim=True)            # the first maximum: argmax's (and serial's) choice
            got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]       # one sync
            c = int(got[2])
            tok = int(self._draft_host[c]) if mapped else c
            return tok, float(np.exp(float(got[0]) - float(got[1])))
        k = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]          # one sync
        cols = got[k + 1:].astype(np.int64)
        ids = self._draft_host[cols] if mapped else cols
        tok = choose_rows(got[None, :k].astype(np.float32), ids[None, :], [position], sampling)[0]
        hit = np.nonzero(ids == tok)[0]
        return int(tok), float(np.exp(float(got[hit[0]]) - float(got[k]))) if len(hit) else 0.0

    def mtp_forward(self, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
        if self.graphs is not None:
            return self.graphs.mtp_forward(next_tokens, streams)
        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, streams)


# -- MTP drafts -----------------------------------------------------------------------------------------
def absorb(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions with main-model streams [n, S*D] and next tokens; logits of the last."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = e.mtp_forward(next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb the kept positions, then chain up to ``count`` drafts for positions position, position + 1, ...
    With ``confidence`` > 0 the chain ends before a draft whose probability under the head is below it."""

    st = e.st
    logits = absorb(e, streams, next_tokens)
    drafts: list[int] = []
    for j in range(count):
        if confidence > 0:
            d, p = e.sample_draft(logits, position + j, sampling)
            if p < confidence:
                break
        else:
            d = e.sample(logits[:1], [position + j], sampling, draft=True)[0]
        drafts.append(d)
        if j + 1 < count:
            prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
            logits = e.mtp_forward([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
            next_tokens = [d]
    return drafts


# -- prefill ------------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
            resume: dict | None = None) -> int:
    """Commit the prompt in chains of up to ``e.rows`` rows (the MTP cache absorbing every position whose next
    token is known) and sample the first output token; the last prompt position is absorbed with it when drafting
    starts. ``resume``: a sequence the prompt extends, whose cache rows are still in place (``State.snapshot``,
    plus the streams of its last position when the MTP head has not absorbed it): only the new tokens run. Rows
    never depend on their chunk, so a resumed prompt ends in the state a fresh one does."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.buf
    use_mtp = mtp and w.mtp is not None and e.mbuf is not None
    begin = 0
    if resume is None:
        e.reset()
    else:
        st.restore(resume["state"])
        begin = st.pos
        if not 0 < begin < len(prompt):
            raise ValueError("a resumed prompt must extend the cached tokens")
        if use_mtp and resume.get("tail") is not None:
            absorb(e, resume["tail"], [prompt[begin]])
    last = None
    for start in range(begin, len(prompt), e.rows):
        chunk = list(prompt[start:start + e.rows])
        R = len(chunk)
        logits = forward(w, st, b, chunk)
        last = logits[R - 1:R].clone()
        streams_last = b.streams[R - 1:R].clone()
        if use_mtp:
            nxt = list(prompt[start + 1:start + R + 1])
            if nxt:
                absorb(e, b.streams[:len(nxt)], nxt)
        commit(w, st, b, R, R)
    first = e.sample(last, [len(prompt)], sampling)[0]
    e.last_streams = streams_last
    e.first = first
    return first


# -- decode loops --------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token.
    ``on_tokens(new)`` hears each step's token; it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        logits = e.forward([out[-1]])
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        commit(w, st, b, 1, 1)
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, committed=out[:-1])


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, depth: int = DEPTH,
               confidence: float = CONFIDENCE, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify the pending token and its MTP drafts in one window, keep up to the first mismatch, draft again.
    Starts from the state ``prefill`` left (the MTP cache holds every prompt position but the last).
    ``on_tokens(new)`` hears each round's kept tokens (after ``pending``); it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    pos0 = st.pos
    unabsorbed = None                                  # the last round's kept rows, not yet in the MTP cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    drafts = draft(e, e.last_streams, [pending], st.pos + 1, min(depth, count - len(out)), sampling, confidence)
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        commit(w, st, b, R, keep)
        unabsorbed = (keep, sampled[:keep])
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        new = sampled[:keep][:max(0, count - len(out))]
        out.extend(sampled[:keep])
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        n = min(depth, count - len(out))
        drafts = []
        if n > 0:
            drafts = draft(e, b.streams[:keep], sampled[:keep], st.pos + 1, n, sampling, confidence)
            unabsorbed = None
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if unabsorbed is not None:              # the MTP cache takes the last kept rows: it then covers the sequence
        absorb(e, b.streams[:unabsorbed[0]], unabsorbed[1])
    committed = out[:st.pos - pos0]
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, keeps, committed)
