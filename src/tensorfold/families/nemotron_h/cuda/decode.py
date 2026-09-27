"""Nemotron-H decoding: window row j samples position pos + j + 1 with the keyed sampler, so kept drafts are serial."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

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
            resume: tuple | None = None) -> Prefilled:
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
    eng.set_sampling(sampling)
    step = eng.prefill_rows
    last = None
    for s in range(begin, len(prompt), step):
        chunk = prompt[s:s + step]
        eng.prefill_chunk(chunk)
        if mtp is not None:
            known = min(len(chunk), len(prompt) - 1 - s)          # rows whose next token is in the prompt
            if known > 0:
                mtp.absorb_rows(eng.p_hidden[:known], prompt[s + 1:s + 1 + known])
        last = len(chunk) - 1
    last_hidden = eng.p_hidden[last:last + 1].clone()
    pending = eng.prefill_token()
    torch.cuda.synchronize()
    return Prefilled(prompt, pending, last_hidden, eng.snapshot(), mtp.snapshot() if mtp is not None else None)


@torch.no_grad()
def serial_decode(eng: Engine, pre: Prefilled, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult:
    """``count`` tokens (the first is ``pre.pending``), one window of one row a token: the reference."""

    eng.restore(pre.engine)
    eng.set_sampling(sampling)
    out = [pre.pending]
    eos = set(eng.c.eos)
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in eos):
        eng.forward([out[-1]])
        tok = eng.tokens()[0]
        eng.commit(1)
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
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


@torch.no_grad()
def draft_decode(eng: Engine, mtp: MTPHead, pre: Prefilled, count: int, sampling: Sampling | None, *,
                 drafts: int = 3, confidence: float = 0.0, copy: bool = True, stop_eos: bool = False,
                 on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult:
    """``confidence`` verifies the first draft, then more while the running confidence product stays at or above it."""

    if not 1 <= drafts <= min(eng.max_rows - 1, 8):
        raise ValueError("drafts must be between 1 and 8")
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
    mtp.round(1, 0 if copied else drafts)
    while len(out) < count and not (stop_eos and out[-1] in eos):
        if copied:
            eng.forward([out[-1]] + copied)
            proposal = copied
        else:
            n = mtp._count
            if confidence > 0.0:
                mtp._copied.synchronize()
                torch.cuda.current_stream().synchronize()
                run, n = 1.0, 0
                for pr in mtp.confidences():
                    run *= pr
                    if n > 0 and run < confidence:
                        break
                    n += 1
            eng.forward([out[-1]], rows=1 + n)
        sampled = eng.tokens()
        if not copied:
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
        new = sampled[:keep]
        out.extend(new)
        if index is not None:
            index.extend(new)
        res.rounds += 1
        res.drafted += len(proposal)
        res.accepted += accepted
        res.widths.append(1 + len(proposal))
        if len(out) < count and not (stop_eos and out[-1] in eos):
            # queue the next round's drafts before handing tokens over, so the caller's work overlaps the head's graph
            copied = index.chain(eng.max_rows - 1) if index is not None else []
            mtp.round(keep, 0 if copied else drafts)
        if on_tokens is not None and on_tokens(new):
            break
    if mtp.pos < eng.pos:                   # the last round's kept rows, so the head covers every committed position
        mtp.round(eng.pos - mtp.pos, 0)
    torch.cuda.synchronize()
    res.seconds = time.perf_counter() - start
    res.tokens = out[:count]
    return res
