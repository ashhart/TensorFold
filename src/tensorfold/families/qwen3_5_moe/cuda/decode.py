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


@torch.no_grad()
def prefill(w, head: Head | None, prompt: Sequence[int], sampling: Sampling | None, *,
            state: State | None = None, cache: Cache | None = None, held: torch.Tensor | None = None,
            stops: Sequence[int] = (), keep: Callable | None = None) -> tuple[State, Cache | None, int, Carry | None]:
    """Commit the prompt, sample its next token, absorb all prompt rows but the last into the head; ``keep(p, ...)`` gets each stop's state."""

    st = clone_state(state) if state is not None else State(w)
    if st.pos >= len(prompt):
        raise ValueError("a reused state must leave at least one prompt token to process")
    mc = None
    if head is not None:
        mc = cache.view(len(prompt)) if cache is not None else Cache(w, len(prompt))      # the prompt's rows only
    ids = torch.tensor(list(prompt[st.pos:]), dtype=torch.int32, device=w.norm.device)
    base, normed = st.pos, None
    bounds = sorted({p for p in stops if base < p < len(prompt)} | {len(prompt)}) if keep is not None else \
        [len(prompt)]
    for end in bounds:
        for a, b in chunks(st.pos, end):
            normed, _ = prefill_chunk(w, ids[a - base:b - base], st, every=head is not None)
            if head is None:
                continue
            rows = normed if held is None else torch.cat([held, normed])
            start = a - (0 if held is None else 1)
            if rows.shape[0] > 1:
                head.forward(mc, rows[:-1], prompt[start + 1:b], start)
                mc.pos = b - 1
            held = rows[-1:]
        if end < len(prompt):
            keep(end, clone_state(st), mc.view() if mc is not None else None, held)
    first = sample_rows(_mm(normed[-1:], w.head), [len(prompt)], sampling)[0]
    carry = Carry(held, [first]) if head is not None else None
    return st, mc, first, carry


COPY_ROWS = 16       # a copied continuation's verify window


@torch.no_grad()
def mtp_decode(w, head: Head, st: State, mc: Cache, carry: Carry, pending: int, count: int,
               sampling: Sampling | None, *, depth: int, confidence: float, stop_eos: bool = True,
               on_tokens: Callable[[list[int]], bool | None] | None = None, runner=None,
               prompt: Sequence[int] = ()) -> Result:
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
        n = st.pos                                        # the pending token's position
        normed, logits = step(carry.states, carry.tokens, mc.pos)
        mc.pos += len(carry.tokens)
        guesses = copies.propose(context, COPY_ROWS - 1)  # an exact repeat of the context first: a long, likely window
        while not guesses:
            token, prob = draft(logits, n + 1, sampling, head.ids)
            guesses.append(token)
            while prob >= confidence and len(guesses) < depth:
                normed, logits = step(normed[-1:], [token], mc.pos + len(guesses) - 1)
                token, prob = draft(logits, n + 1 + len(guesses), sampling, head.ids)
                guesses.append(token)
        tokens = [out[-1]] + guesses
        parents = list(range(-1, len(tokens) - 1))
        logits, record, states = verify(tokens)
        sampled = sample_rows(logits, [n + 1 + i for i in range(len(tokens))], sampling)
        path, terminal = accept(tokens, parents, sampled, count - len(out), w.config.eos if stop_eos else ())
        commit(st, record, path, in_place=runner is not None)
        new = [tokens[r] for r in path[1:]] + [terminal]
        carry = Carry(states[path[0]:path[-1] + 1], new)
        out.extend(new)
        context.extend(new)
        rounds, drafted, kept = rounds + 1, drafted + len(guesses), kept + len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None and on_tokens(new):
            break
    return Result(out, time.perf_counter() - start, rounds, drafted, kept, widths)
