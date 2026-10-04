"""Nemotron-H decoding: window row j samples position pos + j + 1 with the keyed sampler, so kept drafts are serial."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from tensorfold.cuda.draft_depth import DepthRule
from tensorfold.engine.exact_sampling import Sampling

from .engine import Engine
from .mtp import MTPHead


@dataclass
class Prefilled:
    prompt: list[int]
    pending: int                       # the first sampled token (position len(prompt))
    last_hidden: torch.Tensor          # the prompt's last row's final hidden state (1, D)
    engine: dict                       # engine snapshot after the prompt
    mtp: dict | None                   # head snapshot: every prompt position but the last absorbed
    kept: dict | None = None


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    widths: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def prefill(eng: Engine, mtp: MTPHead | None, prompt: Sequence[int], sampling: Sampling | None, *,
            resume: tuple | None = None, constraint=None, keep_at: int | None = None) -> Prefilled:
    """``resume`` = (engine snapshot, head snapshot, kept length, the last hidden state the head has not absorbed)."""

    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prefill needs at least one token")
    begin = 0
    if resume is None:
        eng.reset()
        if mtp is not None:
            mtp.reset()
    else:
        eng.restore(resume[0])
        if mtp is not None and resume[1] is not None:
            mtp.restore(resume[1])
        begin = resume[2]
        if not 0 < begin < len(prompt):
            raise ValueError("a resumed prompt must extend the kept tokens")
        if mtp is not None and resume[3] is not None:
            mtp.absorb_rows(resume[3], [prompt[begin]])
    if keep_at is not None and not begin <= keep_at <= len(prompt):
        raise ValueError("the kept prefix must lie in the prompt's prefill")
    partial = tail = None
    if keep_at == begin and resume is not None:
        partial, tail = resume[0], resume[3]
    eng.set_sampling(sampling)
    if constraint is not None:                           # a reply's grammar masks the first token's row
        eng.mask(constraint, constraint.window([0], [-1]))
    step = eng.prefill_rows
    last = None
    for s in range(begin, len(prompt), step):
        chunk = prompt[s:s + step]
        cut = keep_at - s if keep_at is not None and s < keep_at < s + len(chunk) else 0
        mid = eng.prefill_chunk(chunk, cut=cut)
        if keep_at is not None and s < keep_at <= s + len(chunk):
            partial = mid if mid is not None else {
                "ssm": eng.ssm.clone(), "conv_base": eng.conv_base.clone(),
                "host": (eng.pos, eng.parity, eng.prev_keep)}
            row = keep_at - s - 1
            tail = eng.p_hidden[row:row + 1].clone()
        if mtp is not None:
            known = min(len(chunk), len(prompt) - 1 - s)          # rows whose next token is in the prompt
            if known > 0:
                mtp.absorb_rows(eng.p_hidden[:known], prompt[s + 1:s + 1 + known])
        last = len(chunk) - 1
    last_hidden = eng.p_hidden[last:last + 1].clone()
    pending = eng.prefill_token()
    if constraint is not None:
        eng.mask(None, None)
        constraint.advance([pending])
    torch.cuda.synchronize()
    state, head = eng.snapshot(), mtp.snapshot() if mtp is not None else None
    kept = None
    if partial is not None:
        kept = {"engine": {**state, **partial}, "mtp": {**head, "pos": keep_at - 1} if head else None,
                "tail": tail}
    return Prefilled(prompt, pending, last_hidden, state, head, kept)


@torch.no_grad()
def serial_decode(eng: Engine, pre: Prefilled, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens: Callable[[list[int]], bool | None] | None = None, constraint=None) -> DecodeResult:
    """``count`` tokens (the first is ``pre.pending``), one window of one row a token: the reference."""

    eng.restore(pre.engine)
    eng.set_sampling(sampling)
    out = [pre.pending]
    eos = set(eng.c.eos)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        while len(out) < count and not (stop_eos and out[-1] in eos):
            if constraint is not None:
                eng.mask(constraint, constraint.window([out[-1]], [-1]))
            eng.forward([out[-1]])
            tok = eng.tokens()[0]
            eng.commit(1)
            out.append(tok)
            if constraint is not None:
                constraint.advance([tok])
            if on_tokens is not None and on_tokens([tok]):
                break
    finally:
        eng.mask(None, None)
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, widths=[1] * (len(out) - 1))


class CopyIndex:
    """Longest continuation of an earlier copy of the last ``min_match`` tokens; n-gram starts keep a round O(new)."""

    def __init__(self, context: Sequence[int], min_match: int = 8):
        self.m = min_match
        self.ctx: list[int] = []
        self.starts: dict[tuple, list[int]] = {}
        self.extend(context)

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self.ctx.append(int(t))
            s = len(self.ctx) - self.m
            if s >= 0:
                self.starts.setdefault(tuple(self.ctx[s:]), []).append(s)

    def chain(self, max_nodes: int) -> list[int]:
        ctx, m = self.ctx, self.m
        if len(ctx) < 2 * m or max_nodes < 1:
            return []
        best: list[int] = []
        for start in reversed(self.starts.get(tuple(ctx[-m:]), [])):
            if start > len(ctx) - m - 1:                 # the needle itself
                continue
            cont = ctx[start + m:start + m + max_nodes]
            if len(cont) > len(best):
                best = cont
                if len(best) == max_nodes:
                    break
        return best if len(best) >= m else []


def _queue(mtp: MTPHead, keep: int, copied: list[int], drafts: int, rule: DepthRule | None) -> None:
    """Start a round's head work: an absorb for a copied chain, else level 1 (every level when no rule reads them)."""

    mtp.begin(keep)
    if copied:
        mtp.level(0)
        return
    for j in range(1, (drafts if rule is None else 1) + 1):
        mtp.level(j)


def _depth(mtp: MTPHead, drafts: int, rule: DepthRule | None) -> int:
    """The drafts this round verifies; a rule reads each level's confidence before drafting the next level."""

    if rule is None:
        return drafts
    run, n = 1.0, 0
    for j in range(1, drafts + 1):
        if j > 1:
            mtp.level(j)
        run *= mtp.confidence(j)
        if not rule.keep(j, run):
            break
        n = j
        if not rule.more(j, run):
            break
    return n


@torch.no_grad()
def draft_decode(eng: Engine, mtp: MTPHead, pre: Prefilled, count: int, sampling: Sampling | None, *,
                 drafts: int = 3, rule: DepthRule | None = None, copy: bool = True, stop_eos: bool = False,
                 on_tokens: Callable[[list[int]], bool | None] | None = None, constraint=None) -> DecodeResult:
    """Up to ``drafts`` MTP drafts a round: all of them without ``rule``, else while the rule finds each row pays."""

    if not 1 <= drafts <= mtp.most:
        raise ValueError(f"drafts must be between 1 and {mtp.most}")
    eng.restore(pre.engine)
    mtp.restore(pre.mtp)
    eng.set_sampling(sampling)
    out = [pre.pending]
    index = CopyIndex(list(pre.prompt) + out) if copy else None
    eos = set(eng.c.eos)
    res = DecodeResult([], 0.0, 0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    # the head's first draft reads the prompt's last row and the first sampled token
    eng.hidden[:1].copy_(pre.last_hidden)
    eng.sampled[:1].fill_(pre.pending)
    copied = index.chain(eng.max_rows - 1) if index is not None else []
    _queue(mtp, 1, copied, drafts, rule)
    while len(out) < count and not (stop_eos and out[-1] in eos):
        if copied:
            proposal, n = copied, len(copied)
        else:
            n = _depth(mtp, drafts, rule)
            proposal = None
            if constraint is not None:
                mtp.wait()
                proposal = mtp.drafts()[:n]
        if constraint is not None:                       # the grammar cuts the chain at its first rejected draft
            window = constraint.window([out[-1]] + proposal, list(range(-1, len(proposal))))
            proposal = window.tokens[1:]
            n = len(proposal)
            eng.mask(constraint, window)
        if copied:
            eng.forward([out[-1]] + proposal)
        else:
            eng.forward([out[-1]], rows=1 + n)
        sampled = eng.tokens()
        if not copied and constraint is None:
            proposal = mtp.drafts()[:n]
        accepted = 0
        while accepted < len(proposal) and proposal[accepted] == sampled[accepted]:
            accepted += 1
        if stop_eos:
            for j in range(accepted):
                if sampled[j] in eos:
                    accepted = j
                    break
        accepted = min(accepted, count - len(out) - 1)
        keep = accepted + 1
        eng.commit(keep)
        if rule is not None:
            rule.done(keep, 1 + len(proposal), 0 if copied else mtp.levels)
        new = sampled[:keep]
        if constraint is not None:
            constraint.advance(new)
        out.extend(new)
        if index is not None:
            index.extend(new)
        res.rounds += 1
        res.drafted += len(proposal)
        res.accepted += accepted
        res.widths.append(1 + len(proposal))
        if len(out) < count and not (stop_eos and out[-1] in eos):
            # queue the next round's head work before handing tokens over, so the caller's work overlaps it
            copied = index.chain(eng.max_rows - 1) if index is not None else []
            _queue(mtp, keep, copied, drafts, rule)
        if on_tokens is not None and on_tokens(new):
            break
    eng.mask(None, None)
    if mtp.pos < eng.pos:                   # the last round's kept rows, so the head covers every committed position
        mtp.round(eng.pos - mtp.pos, 0)
    torch.cuda.synchronize()
    res.seconds = time.perf_counter() - start
    res.tokens = out[:count]
    return res
