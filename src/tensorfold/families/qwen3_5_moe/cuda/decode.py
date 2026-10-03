"""MTP chains verified in one forward against the target's keyed samples: drafted tokens always equal serial ones."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import accept
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, clone_state
from tensorfold.families.qwen3_5.cuda.forward import State, _mm, commit, tree_forward
from tensorfold.families.qwen3_5.cuda.prefill import chunks, prefill_chunk

from .mtp import Cache, Head


@dataclass
class Carry:
    """Committed rows the head has not absorbed: their final normed states and the token after each."""

    states: torch.Tensor
    tokens: list[int]


@dataclass
class Result:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    widths: list[int] = field(default_factory=list)


def draft(logits: torch.Tensor, position: int, sampling: Sampling | None,
          ids: torch.Tensor | None = None) -> tuple[int, float]:
    """The head's guess for ``position`` by the target's own keyed rule, and its temperature-1 probability."""

    row = logits.float()[0]
    if sampling is None or sampling.temperature <= 0:
        pick = int(row.argmax().item())
    else:
        k = min(row.shape[0], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[0]
        values, cols = torch.topk(row, k)
        tokens = (ids[cols] if ids is not None else cols).cpu().numpy().astype(np.int64)
        chosen = choose_rows(values.cpu().numpy()[None], tokens[None], [position], sampling)[0]
        pick = int(cols[int(np.nonzero(tokens == chosen)[0][0])].item())
    token = int(ids[pick].item()) if ids is not None else pick
    return token, float(torch.softmax(row, -1)[pick].item())


def picks(logits: torch.Tensor, positions: Sequence[int], samplings: Sequence[Sampling | None],
          ids: np.ndarray | None = None) -> list[tuple[int, float]]:
    """``draft`` for every row of ``logits`` (a stream's row each; ``ids``: the draft vocabulary), read back together by candidate count."""

    rows = logits.float()
    probs = torch.softmax(rows, -1)
    width = rows.shape[1]
    groups: dict[int, list[int]] = {}
    for i, smp in enumerate(samplings):
        greedy = smp is None or smp.temperature <= 0
        groups.setdefault(0 if greedy else min(width, int(smp.top_k) + MARGIN) if smp.top_k else width, []).append(i)
    launched = []
    for k, members in groups.items():
        sub = rows if len(members) == len(samplings) else rows[members]
        values, cols = (None, sub.argmax(-1, keepdim=True)) if k == 0 else torch.topk(sub, k)
        mine = probs if len(members) == len(samplings) else probs[members]
        launched.append((members, values, cols, mine.gather(1, cols)))
    out: list[tuple[int, float]] = [(0, 0.0)] * len(samplings)
    for members, values, cols, p in launched:
        cols, p = cols.cpu().numpy(), p.cpu().numpy()
        values = values.cpu().numpy() if values is not None else None
        for j, i in enumerate(members):
            tokens = (ids[cols[j]] if ids is not None else cols[j]).astype(np.int64)
            if values is None:
                out[i] = int(tokens[0]), float(p[j, 0])
                continue
            chosen = choose_rows(values[j][None], tokens[None], [positions[i]], samplings[i])[0]
            at = int(np.nonzero(tokens == chosen)[0][0])
            out[i] = int(chosen), float(p[j, at])
    return out


@torch.no_grad()
def prefill(w, head: Head | None, prompt: Sequence[int], sampling: Sampling | None, *,
            state: State | None = None, cache: Cache | None = None, held: torch.Tensor | None = None,
            stops: Sequence[int] = (), keep: Callable | None = None,
            constraint=None, keep_at: int | None = None) -> tuple[State, Cache | None, int, Carry | None]:
    """Commit the prompt, sample its next token, absorb all prompt rows but the last into the head; ``keep(p, ...)`` gets each stop's state."""

    st = clone_state(state) if state is not None else State(w)
    if st.pos >= len(prompt):
        raise ValueError("a reused state must leave at least one prompt token to process")
    if keep_at is not None and (keep is None or not st.pos <= keep_at <= len(prompt)):
        raise ValueError("keep_at needs a callback and a point in the prefilled range")
    mc = None
    if head is not None:
        mc = cache.view(len(prompt)) if cache is not None else Cache(w, len(prompt))      # the prompt's rows only
    normed = None
    bounds = sorted({p for p in stops if st.pos < p < len(prompt)} | {len(prompt)}) if keep is not None else \
        [len(prompt)]
    for end in bounds:
        normed, held = extend(w, head, prompt, st, mc, held, end, keep_at=keep_at, keep=keep)
        if end < len(prompt):
            keep(end, clone_state(st), mc.view() if mc is not None else None, held)
    logits = _mm(normed[-1:], w.head)
    if constraint is not None:                           # a reply's grammar (tensorfold.engine.grammar)
        logits = constraint.mask(logits)
    first = sample_rows(logits, [len(prompt)], sampling)[0]
    if constraint is not None:
        constraint.advance([first])
    carry = Carry(held, [first]) if head is not None else None
    return st, mc, first, carry


@torch.no_grad()
def extend(w, head: Head | None, prompt: Sequence[int], st: State, mc: Cache | None, held: torch.Tensor | None,
           end: int, *, keep_at: int | None = None, keep: Callable | None = None
           ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Commit prompt[st.pos:end] in chunks (bits independent of ``end``); the head absorbs every row but the last, returned held."""

    ids = torch.tensor(list(prompt[st.pos:end]), dtype=torch.int32, device=w.norm.device)
    base, normed = st.pos, None
    saved = False
    for a, b in chunks(st.pos, end):
        if keep is not None and keep_at == a and not saved:
            keep(a, clone_state(st), mc.view() if mc is not None else None,
                 held.clone() if held is not None else None)
            saved = True
        cut = keep_at - a if keep_at is not None and a < keep_at < b else 0
        normed, _, *part = prefill_chunk(w, ids[a - base:b - base], st, every=head is not None, cut=cut)
        if head is not None:
            rows = normed if held is None else torch.cat([held, normed])
            start = a - (0 if held is None else 1)
            if rows.shape[0] > 1:
                head.forward(mc, rows[:-1], prompt[start + 1:b], start)
                mc.pos = b - 1
            held = rows[-1:]
        if keep is not None and keep_at is not None and a < keep_at <= b:
            snapshot = part[0] if part else clone_state(st)
            head_cache = mc.view() if mc is not None else None
            if head_cache is not None:
                head_cache.pos = keep_at - 1
            tail = normed[keep_at - a - 1:keep_at - a].clone() if head is not None else None
            keep(keep_at, snapshot, head_cache, tail)
            saved = True
    return normed, held


COPY_ROWS = 16       # a copied continuation's verify window


@torch.no_grad()
def mtp_decode(w, head: Head, st: State, mc: Cache, carry: Carry, pending: int, count: int,
               sampling: Sampling | None, *, depth: int, confidence: float, stop_eos: bool = True,
               on_tokens: Callable[[list[int]], bool | None] | None = None, runner=None,
               prompt: Sequence[int] = (), constraint=None) -> Result:
    """Each round: absorb the carry (its last row drafts first), then verify a copied continuation from the context or a chain of up to ``depth`` MTP drafts, and keep a path; ``runner``: a ``graphs.Graphs`` to decode in and replay."""

    if runner is not None:                               # copied into its fixed buffers; commits write in place
        st, mc = runner.load(st, mc, min(runner.capacity, st.pos + count + COPY_ROWS))
        verify, step = runner.verify, runner.draft
    else:
        st, mc = clone_state(st), mc.view(st.pos + count + COPY_ROWS)       # both write only past their positions

        def verify(tokens):
            return tree_forward(w, torch.tensor(tokens, dtype=torch.int32, device=w.norm.device),
                                list(range(-1, len(tokens) - 1)), st, hidden=True)

        def step(states, tokens, p0):
            normed = head.forward(mc, states, tokens, p0)
            return normed, head.logits(normed[-1:])
    out, rounds, drafted, kept, widths = [pending], 0, 0, 0, []
    context, copies = list(prompt) + [pending], CopyIndex()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.config.eos):
        tokens, path, new, carry = mtp_round(st, mc, carry, out[-1], count - len(out), sampling, context, copies,
                                             depth=depth, confidence=confidence, ids=head.ids, verify=verify,
                                             step=step, eos=w.config.eos if stop_eos else (),
                                             in_place=runner is not None, constraint=constraint)
        out.extend(new)
        context.extend(new)
        rounds, drafted, kept = rounds + 1, drafted + len(tokens) - 1, kept + len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None and on_tokens(new):
            break
    return Result(out, time.perf_counter() - start, rounds, drafted, kept, widths)


def mtp_round(st: State, mc: Cache, carry: Carry, pending: int, room: int, sampling: Sampling | None,
              context: Sequence[int], copies: CopyIndex, *, depth: int, confidence: float, ids, verify, step,
              eos: Sequence[int] = (), in_place: bool = False,
              constraint=None) -> tuple[list[int], list[int], list[int], Carry]:
    """One round (carry absorbed, copy or MTP chain proposed, verified, at most ``room`` rows kept): (tokens, path, new, carry)."""

    n = st.pos                                            # the pending token's position
    normed, logits = step(carry.states, carry.tokens, mc.pos)
    mc.pos += len(carry.tokens)
    guesses = copies.propose(context, COPY_ROWS - 1)      # an exact repeat of the context first: a long, likely window
    while not guesses:
        token, prob = draft(logits, n + 1, sampling, ids)
        guesses.append(token)
        while prob >= confidence and len(guesses) < depth:
            normed, logits = step(normed[-1:], [token], mc.pos + len(guesses) - 1)
            token, prob = draft(logits, n + 1 + len(guesses), sampling, ids)
            guesses.append(token)
    tokens = [pending] + guesses
    window = constraint.window(tokens, list(range(-1, len(tokens) - 1))) if constraint is not None else None
    if window is not None:                                # the drafts no accepted path can hold are cut first
        tokens = window.tokens
    logits, record, states = verify(tokens)
    if window is not None:
        logits = constraint.mask(logits, window)
    sampled = sample_rows(logits, [n + 1 + i for i in range(len(tokens))], sampling)
    path, terminal = accept(tokens, list(range(-1, len(tokens) - 1)), sampled, room, eos)
    commit(st, record, path, in_place=in_place)
    new = [tokens[r] for r in path[1:]] + [terminal]
    if constraint is not None:
        constraint.advance(new)
    return tokens, path, new, Carry(states[path[0]:path[-1] + 1], new)
