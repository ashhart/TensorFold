"""Where a prompt's prefill chunks start, from its tokens alone: prompts that agree up to a start cut alike up to it."""

from __future__ import annotations

import bisect
from typing import Any, Sequence

import numpy as np


class PrefillPlan:
    """Chunk starts: 0, then the first resume point ``min_chunk`` or more tokens on, else ``step`` tokens on."""

    def __init__(self, step: int = 2048, openers: Sequence[int] = (), min_chunk: int = 256,
                 assistant: Sequence[int] = ()) -> None:
        self.step = int(step)
        self.openers = tuple(sorted({int(t) for t in openers}))          # any message's first token
        self.assistant = tuple(int(t) for t in assistant)                # the tokens that open an assistant message
        self.min_chunk = int(min_chunk)
        if self.step < 1 or self.min_chunk < max(1, len(self.assistant)) or (
                (self.openers or self.assistant) and self.min_chunk > self.step):
            raise ValueError("step >= 1 and len(assistant) <= min_chunk <= step required")

    @property
    def name(self) -> str:
        """The scheme, for snapshot keys: states computed under another scheme are not this one's."""

        if not (self.openers or self.assistant):
            return f"grid{self.step}"
        marks = f"{'.'.join(map(str, self.openers))}:{'.'.join(map(str, self.assistant))}"
        return f"grid{self.step}+msg{self.min_chunk}:{marks}"

    def points(self, ids: Any) -> list[int]:
        """Resume points: assistant message starts, and the second message's (where sessions share a system block)."""

        found: set[int] = set()
        if self.openers:
            at = np.flatnonzero(np.isin(ids, self.openers))
            if len(at) > 1:
                found.add(int(at[1]))
        k, n = len(self.assistant), len(ids)
        if k and n >= k:
            hit = np.ones(n - k + 1, dtype=bool)
            for j, token in enumerate(self.assistant):
                hit &= ids[j:n - k + 1 + j] == token
            found.update(np.flatnonzero(hit).tolist())
        return sorted(q for q in found if q > 0)

    def chunks(self, tokens: Sequence[int]) -> "PromptChunks":
        n = len(tokens)
        found: list[int] = []
        if (self.openers or self.assistant) and n > 1:
            found = self.points(np.fromiter((int(t) for t in tokens), dtype=np.int64, count=n))
        starts, last, i = [0], 0, 0
        while True:
            while i < len(found) and found[i] < last + self.min_chunk:
                i += 1                          # too close to the last start: merged into its chunk
            nxt = last + self.step
            if i < len(found) and found[i] < nxt:
                nxt = found[i]
            if nxt >= n:
                return PromptChunks(starts, n)
            starts.append(nxt)
            last = nxt


class PromptChunks:
    """One prompt's chunk starts (``None``: any position, chunks of ``step`` from wherever a prefill begins)."""

    def __init__(self, starts: list[int] | None, length: int, step: int = 2048) -> None:
        self.starts = starts
        self.length = int(length)
        self.step = int(step)
        self._set = frozenset(starts) if starts is not None else None

    def __contains__(self, position: Any) -> bool:
        """Whether a stored state after ``position`` tokens resumes this prompt exactly."""

        position = int(position)
        if not 0 < position < self.length:
            return False
        return self._set is None or position in self._set

    def floor(self, position: int) -> int:
        """The chunk start at or before ``position``."""

        position = max(0, min(int(position), self.length))
        if self.starts is None:
            return position
        return self.starts[bisect.bisect_right(self.starts, position) - 1]

    def between(self, begin: int, end: int) -> list[tuple[int, int]]:
        """The chunks covering [begin, end), ``begin`` a start (or 0) and ``end`` a start or the prompt's end."""

        if self.starts is None:
            return [(b, min(b + self.step, end)) for b in range(begin, end, self.step)]
        inner = [s for s in self.starts if begin < s < end]
        edges = [begin, *inner, end]
        return [(a, b) for a, b in zip(edges, edges[1:]) if a < b]


def message_markers(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(the special tokens that open a message, the tokens that open an assistant message) in this chat template."""

    if getattr(type(tokenizer), "message_markers", None) is not None:     # markers the probe cannot see (not special)
        return tokenizer.message_markers
    from tensorfold.server.text import render_prompt_ids, template_late_system

    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special = {int(i) for i, t in decoder.items() if getattr(t, "special", False)}
    special |= {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or ())}
    late = template_late_system(tokenizer)
    pieces: dict[str, list[list[int]]] = {"user": [], "assistant": []}
    parted: list[tuple[list[int], list[int], str]] = []      # renders that change the end of the one before
    for words in (("Alpha", "Beta", "Gamma", "Delta"), ("one two", "three four", "five six", "seven eight")):
        talk = [{"role": role, "content": text} for role, text in zip(("user", "assistant") * 2, words)]
        for thinking in (False, True):
            def render(k: int, generate: bool = False) -> list[int]:
                return render_prompt_ids(tokenizer, talk[:k], enable_thinking=thinking, late_system=late,
                                         add_generation_prompt=generate)

            try:
                renders = [render(k) for k in (1, 2, 3, 4)]
                generation = [(render(1), render(1, True)), (render(3), render(3, True))]
            except Exception:  # noqa: BLE001 - a template that cannot render the probe gets the grid alone
                return (), ()
            pairs = [(renders[0], renders[1], "assistant"), (renders[1], renders[2], "user"),
                     (renders[2], renders[3], "assistant"), *((a, b, "assistant") for a, b in generation)]
            for before, after, role in pairs:
                if len(after) > len(before) and after[:len(before)] == before:
                    pieces[role].append([int(t) for t in after[len(before):]])
                else:
                    parted.append((before, after, role))
    # a template may render the last reply unlike the same reply in history (Qwen3.5 and 3.6 drop its empty think
    # block once a user message follows): the new message then starts at the only opener past where the renders differ
    openers = _openers(pieces, special)
    for before, after, role in parted:
        split = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), min(len(before), len(after)))
        starts = [i for i in range(split, len(after)) if after[i] in openers]
        if len(starts) == 1:
            pieces[role].append([int(t) for t in after[starts[0]:]])
    if not pieces["user"] or not pieces["assistant"]:
        return (), ()
    openers = _openers(pieces, special)
    header, user = _common(pieces["assistant"]), _common(pieces["user"])
    cut = next((i for i, t in enumerate(header) if i >= len(user) or user[i] != t), None)
    assistant = tuple(header[:cut + 1]) if cut is not None and header and header[0] in special else ()
    return openers, assistant


def _openers(pieces: dict[str, list[list[int]]], special: set[int]) -> tuple[int, ...]:
    """The special tokens every piece of a role starts with."""

    return tuple(sorted({group[0][0] for group in pieces.values()
                         if group and group[0][0] in special and all(p[0] == group[0][0] for p in group)}))


def _common(pieces: list[list[int]]) -> list[int]:
    """The longest prefix every piece shares."""

    first = pieces[0]
    n = min(len(p) for p in pieces)
    return next((first[:i] for i in range(n) if any(p[i] != first[i] for p in pieces)), first[:n])


def block_jobs(plan: PrefillPlan | None, block: Sequence[int], pad: int) -> list[tuple[list[int], int]]:
    """The prompts that compute a stored block a chunk at a time, each with the chunk start it checkpoints."""

    if plan is None:
        return [([*block, pad], len(block))]
    probe = [*block, plan.openers[0] if plan.openers else pad]      # an opener after it keeps the block's end a start
    return [(probe[:at + 1], at) for at in plan.chunks(probe).starts[1:]]


__all__ = ["PrefillPlan", "PromptChunks", "block_jobs", "message_markers"]
