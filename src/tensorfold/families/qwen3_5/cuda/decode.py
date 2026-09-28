"""One-stream 128-lane exact CUDA decode with DFlash2 and copy proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Sequence

import torch

from tensorfold.engine.exact_sampling import Sampling

from .forward import State, _paths, commit, tree_forward
from tensorfold.cuda.sampling import sample_rows
from .weights import Weights


def clone_state(st: State) -> State:
    """The committed tensors are immutable; commits replace their list entries."""

    other = object.__new__(State)
    other.pos, other.limit = st.pos, st.limit
    other.rope_delta = st.rope_delta
    other.conv = st.conv.copy()
    other.rec = st.rec.copy()
    other.kv = st.kv.copy()
    return other


def _tokens(ids: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device=device)


def prefill_stops(w: Weights, prompt: Sequence[int], st: State, draft=None, *, stops: Sequence[int] = (),
                  keep: Callable | None = None, tp: bool = False, keep_at: int | None = None, vision=None):
    """Commit the rest of the prompt into ``st``, handing ``keep(p, state, drafter context)`` the state after each stop (``keep_at``: ``prefill_state``'s)."""

    from .prefill import prefill_state

    if vision is not None and stops:
        raise ValueError("an image prompt keeps no prompt states")
    for p in stops:
        if st.pos < p < len(prompt) and keep is not None:
            prefill_state(w, prompt[:p], st, tp=tp, draft=draft)
            keep(p, clone_state(st), draft.snapshot() if draft is not None else None)
    return prefill_state(w, prompt, st, tp=tp, draft=draft, keep_at=keep_at, vision=vision)


@torch.no_grad()
def prefill(w: Weights, prompt: Sequence[int], sampling: Sampling | None,
            draft=None, *, state: State | None = None, limit: int = 0, stops: Sequence[int] = (),
            keep: Callable | None = None, keep_at: int | None = None, vision=None):
    """Commit the prompt and sample the first token; resuming a kept ``state`` gives a fresh prefill's bits (``keep_at`` adds a third item: the state after prompt[:keep_at] and the drafter's snapshot there)."""

    from .forward import _mm

    if not prompt:
        raise ValueError("prefill requires at least one token")
    st = clone_state(state) if state is not None else State(w)
    if state is None:
        st.limit = limit                    # a fresh state's attention caches stop here; a resumed one keeps its own
    if st.pos >= len(prompt):
        raise ValueError("a reused state must leave at least one prompt token to process")
    out = prefill_stops(w, prompt, st, draft, stops=stops, keep=keep, keep_at=keep_at, vision=vision)
    normed = out if keep_at is None else out[0]
    pending = sample_rows(_mm(normed, w.head), [len(prompt)], sampling)[0]
    return (st, pending) if keep_at is None else (st, pending, out[1])


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted_rows: int
    accepted_drafts: int
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    sample_seconds: float = 0.0
    commit_seconds: float = 0.0
    widths: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(w: Weights, st: State, pending: int, count: int,
                  sampling: Sampling | None, *, stop_eos: bool = True) -> DecodeResult:
    """Serial baseline sharing the verifier and sampler's arithmetic."""

    if count < 1:
        raise ValueError("count must include the first sampled token")
    st = clone_state(st)
    out = [pending]
    start = time.perf_counter()
    while len(out) < count and (not stop_eos or out[-1] not in w.config.eos):
        logits, record = tree_forward(w, _tokens([out[-1]], w.norm.device), [-1], st)
        next_token = sample_rows(logits, [st.pos + 1], sampling)[0]
        commit(st, record, [0])
        out.append(next_token)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    rounds = len(out) - 1
    return DecodeResult(out, seconds, rounds, 0, 0, widths=[1] * rounds)


class CopyIndex:
    """Index earlier suffix matches and propose the longest continuation, breaking ties by latest occurrence and requiring ``min_match`` tokens."""

    def __init__(self, min_match: int = 8):
        self.n = min_match
        self.positions: dict[tuple[int, ...], list[int]] = {}
        self.indexed = 0            # grams starting before this position are in the index

    def update(self, context: Sequence[int]) -> None:
        n = self.n
        # Index complete grams except the trailing query itself so matches always refer to earlier occurrences.
        last = len(context) - n
        for start in range(self.indexed, last):
            self.positions.setdefault(tuple(context[start:start + n]), []).append(start)
        self.indexed = max(self.indexed, last)

    def propose(self, context: Sequence[int], max_nodes: int = 127) -> list[int]:
        n = self.n
        if len(context) < 2 * n:
            return []
        self.update(context)
        starts = self.positions.get(tuple(context[-n:]))
        if not starts:
            return []
        best: list[int] = []
        for start in reversed(starts):
            continuation = list(context[start + n:start + n + max_nodes])
            if len(continuation) > len(best):
                best = continuation
                if len(best) == max_nodes:
                    break
        return best if len(best) >= n else []


def copy_chain(context: Sequence[int], max_nodes: int = 127,
               min_match: int = 8) -> list[int]:
    """Reuse a repeated suffix's next tokens as a speculative chain."""

    if len(context) < 2 * min_match:
        return []
    needle = list(context[-min_match:])
    best: list[int] = []
    for start in range(len(context) - min_match - 1, -1, -1):
        if list(context[start:start + min_match]) != needle:
            continue
        continuation = list(context[start + min_match:start + min_match + max_nodes])
        if len(continuation) > len(best):
            best = continuation
            if len(best) == max_nodes:
                break
    return best if len(best) >= min_match else []


@torch.no_grad()
def _round_record(tokens: list[int], parents: list[int], depths: list[int], path: list[int], terminal: int,
                  stop: str, source: str, draft, spent: dict[str, float]) -> dict:
    """One round of ``draft_decode(trace=...)``: what was proposed, how far it held, and why it stopped."""

    last = path[-1]
    siblings = sum(1 for row in range(1, len(tokens)) if parents[row] == last)
    if stop == "miss" and siblings == 0:
        stop = "horizon"                     # the proposal had nothing past the accepted prefix
    top = max(depths)
    rec = {"source": source, "rows": len(tokens), "max_depth": top,
           "nodes_per_depth": [depths.count(d) for d in range(1, top + 1)],
           "accepted": len(path) - 1, "stop": stop, "siblings_at_stop": siblings,
           **{k + "_ms": round(1000 * v, 3) for k, v in spent.items()}}
    if stop == "miss" and source == "tree":
        cands = getattr(draft, "last_candidates", None)
        depth = len(path)                    # depth of the position the target rejected
        if cands is not None and depth - 1 < len(cands):
            rec["candidate_hit"] = bool(terminal in set(int(t) for t in cands[depth - 1]))
        in_vocab = getattr(draft, "in_vocab", None)
        if in_vocab is not None:
            rec["vocab_miss"] = not in_vocab(terminal)
    return rec


def draft_decode(w: Weights, st: State, prompt: Sequence[int], pending: int,
                 count: int, sampling: Sampling | None, draft,
                 *, max_rows: int = 128, tree_rows: int | None = None,
                 allow_copy: bool = True, stop_eos: bool = True,
                 on_tokens: Callable[[list[int]], bool | None] | None = None,
                 trace: list | None = None, inplace: bool = False) -> DecodeResult:
    """Verify trees and replay matching paths (host-only traces leave tokens unchanged); ``inplace``: commit into ``st`` itself, which nothing else holds."""

    if count < 1 or not 1 <= max_rows <= 128:
        raise ValueError("count >= 1 and 1 <= max_rows <= 128 required")
    tree_rows = max_rows if tree_rows is None else tree_rows
    if not 1 <= tree_rows <= max_rows:
        raise ValueError("tree_rows must be between 1 and max_rows")
    st = st if inplace else clone_state(st)
    out = [pending]
    context = list(prompt) + out
    copies = CopyIndex() if allow_copy else None
    stages = dict(draft=0.0, verify=0.0, sample=0.0, commit=0.0)
    rounds = drafted_rows = accepted_drafts = 0
    widths: list[int] = []
    start = time.perf_counter()
    stopped = False
    while len(out) < count and (not stop_eos or out[-1] not in w.config.eos) and not stopped:
        stage = time.perf_counter()
        copied = copies.propose(context, max_rows - 1) if copies is not None else []
        if copied:
            guesses = copied
            parents = list(range(-1, len(guesses) - 1))
        elif draft is not None:
            guesses, parents = draft.propose_tree(
                out[-1], len(context), tree_rows - 1, sampling)
        else:                                # no draft model: one row a round unless the context offers a copy
            guesses, parents = [], []
        tokens = [out[-1]] + guesses
        tree_parents = [-1] + [0 if p < 0 else p + 1 for p in parents]
        torch.cuda.synchronize()
        spent = {"draft": time.perf_counter() - stage}
        stages["draft"] += spent["draft"]
        stage = time.perf_counter()
        logits, record, *tapped = tree_forward(
            w, _tokens(tokens, w.norm.device), tree_parents, st, capture_taps=draft is not None)
        torch.cuda.synchronize()
        spent["verify"] = time.perf_counter() - stage
        stages["verify"] += spent["verify"]
        stage = time.perf_counter()
        depths, _ = _paths(tree_parents)
        sampled = sample_rows(logits, [st.pos + d + 1 for d in depths], sampling)
        children: dict[tuple[int, int], int] = {}
        for row in range(1, len(tokens)):
            children.setdefault((tree_parents[row], tokens[row]), row)
        path = [0]
        terminal = sampled[0]
        stop = "length"
        while len(out) + len(path) < count:
            if stop_eos and terminal in w.config.eos:
                stop = "eos"
                break
            child = children.get((path[-1], terminal))
            if child is None:
                stop = "miss"
                break
            path.append(child)
            terminal = sampled[child]
        spent["sample"] = time.perf_counter() - stage
        stages["sample"] += spent["sample"]
        if trace is not None:
            trace.append(_round_record(tokens, tree_parents, depths, path, terminal, stop,
                                       "copy" if copied else "tree", draft, spent))
        stage = time.perf_counter()
        commit(st, record, path)
        if draft is not None:
            draft.add_taps(tapped[0][path])
        out.extend(tokens[row] for row in path[1:])
        out.append(terminal)
        context.extend(tokens[row] for row in path[1:])
        context.append(terminal)
        torch.cuda.synchronize()
        stages["commit"] += time.perf_counter() - stage
        if trace is not None:
            trace[-1]["commit_ms"] = round(1000 * (time.perf_counter() - stage), 3)
        rounds += 1
        drafted_rows += len(guesses)
        accepted_drafts += len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None:
            stopped = bool(on_tokens([tokens[row] for row in path[1:]] + [terminal]))
    return DecodeResult(out, time.perf_counter() - start, rounds, drafted_rows,
                        accepted_drafts, stages["draft"], stages["verify"],
                        stages["sample"], stages["commit"], widths)
