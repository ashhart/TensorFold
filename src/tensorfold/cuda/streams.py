"""Concurrent CUDA requests: a ``Stream`` each, the rows a verified window keeps, and prompt ends to resume."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class Stream:
    """One request. ``emit`` receives each round's new tokens and returns True to stop the stream."""

    prompt: list[int]
    count: int                                    # tokens to produce, the first sampled one included
    sampling: Any = None
    draft: bool = True                            # False: one row a round, the serial reference
    stop_eos: bool = True                         # False: an end token does not end it (ignore_eos)
    vision: Any = None
    emit: Callable[[list[int]], bool | None] | None = None
    sid: int = 0
    st: Any = None                                # the committed model state
    snap: Any = None                              # the drafter's context
    copies: Any = None
    out: list[int] = field(default_factory=list)
    context: list[int] = field(default_factory=list)
    committed: list[int] = field(default_factory=list)     # tokens committed after the prompt, on every rank
    drafts: list[int] = field(default_factory=list)       # an MTP family's drafts for the next round
    stops: list[int] = field(default_factory=list)        # prompt positions whose states the prefill keeps
    constraint: Any = None                                # the reply's grammar (tensorfold.engine.grammar), or None
    background: bool = False                              # priority "background": after, and yielding to, the rest
    carry: dict | None = None                             # the stats of the stream this one continues
    owed: list[int] = field(default_factory=list)         # a replay's tokens sent before it gave way: checked, not resent
    error: Exception | None = None                        # why a stream ended without finishing
    waiting: bool = False                                 # held out of rounds until its caches can grow
    done: bool = False
    rounds: int = 0
    min_rows: int = 0
    drafted: int = 0                                      # drafted rows its rounds verified, and the ones kept
    accepted: int = 0
    cached: int = 0
    prefill_s: float = 0.0
    started: float = 0.0
    finished: float = 0.0

    def take(self, new: list[int], eos: Sequence[int] = ()) -> None:
        """Append a round's tokens and emit them; the stream ends at its count, an end token or a stop."""

        self.out.extend(new)
        self.context.extend(new)
        self.accepted += max(0, len(new) - 1)             # a round's kept drafts come before its own token
        fresh = list(new)
        if self.owed:                                     # a replay writes the tokens it sent before again first
            k = min(len(self.owed), len(fresh))
            if fresh[:k] != self.owed[:k]:
                self.error = RuntimeError("a background request's replay differs from the reply it sent")
            self.owed, fresh = self.owed[k:], fresh[k:]
        stop = bool(self.emit(fresh)) if self.emit is not None and fresh else False
        if self.error is not None or stop or len(self.out) >= self.count or self.out[-1] in eos:
            self.done = True
            self.finished = time.perf_counter()

    def counted(self, rows: int) -> None:
        self.rounds += 1
        self.drafted += rows - 1
        self.min_rows = rows if self.min_rows == 0 else min(self.min_rows, rows)

    def stats(self) -> dict:
        own = {"prefill_s": round(self.prefill_s, 4), "decode_s": round(max(self.finished - self.started, 0.0), 4),
               "rounds": self.rounds, "drafts": self.draft, "cached": self.cached, "min_rows": self.min_rows,
               "drafted": self.drafted, "accepted": self.accepted}
        if self.carry is None:
            return own
        both = {k: round(self.carry[k] + own[k], 4) for k in ("prefill_s", "decode_s")}
        both.update({k: self.carry[k] + own[k] for k in ("rounds", "drafted", "accepted")})
        rows = [r for r in (self.carry["min_rows"], own["min_rows"]) if r]
        return {**own, **both, "cached": self.carry["cached"], "min_rows": min(rows, default=0)}

    def continued(self) -> "Stream":
        """This stream again from its prompt, for later (as the Mac replays): what it sent is owed, not sent again."""

        return Stream(self.prompt, self.count, self.sampling, draft=self.draft, stop_eos=self.stop_eos, emit=self.emit,
                      background=self.background, carry=self.stats(), owed=[*self.out, *self.owed])


def next_fill(filling: list[Stream]) -> Stream:
    """The queued prompt to prefill next: the oldest foreground one, else the oldest."""

    return next((s for s in filling if not s.background), filling[0])


def accept(tokens: Sequence[int], parents: Sequence[int], sampled: Sequence[int], room: int,
           eos: Sequence[int] = ()) -> tuple[list[int], int]:
    """The kept rows (root, then children equal to their parent's sample, <= ``room``, none past an end token)."""

    children: dict[tuple[int, int], int] = {}
    for row in range(1, len(tokens)):
        children.setdefault((parents[row], tokens[row]), row)
    path, terminal = [0], sampled[0]
    while len(path) < room and terminal not in eos:
        child = children.get((path[-1], terminal))
        if child is None:
            break
        path.append(child)
        terminal = sampled[child]
    return path, terminal


class PrefixCache:
    """Private prompt-end states by ids (never decoded rows: prefill and decode bits differ), newest last."""

    def __init__(self, keep: int = 8) -> None:
        self.keep = keep
        self.entries: list[tuple[list[int], Any, Any]] = []
        self.hit: set[tuple[int, ...]] = set()             # entries a later prompt resumed from

    def longest(self, prompt: Sequence[int]):
        """The longest entry the prompt strictly extends (one prompt token is always left to prefill), now newest."""

        best = None
        for entry in self.entries:
            ids = entry[0]
            if len(ids) < len(prompt) and list(prompt[:len(ids)]) == ids and (best is None or len(ids) > len(best[0])):
                best = entry
        return self._touch(best)

    def named(self, prompt: Sequence[int], length: int):
        """The entry holding the prompt's first ``length`` ids (a follower rank finds the leader's pick), now newest."""

        return self._touch(next((e for e in self.entries if len(e[0]) == length and list(prompt[:length]) == e[0]),
                                None))

    def _touch(self, entry):
        """A hit becomes the newest entry, so a shared system block outlives the prompts that reuse it."""

        if entry is not None:
            self.entries = [e for e in self.entries if e is not entry] + [entry]
            self.hit.add(tuple(entry[0]))
        return entry

    def add(self, ids: list[int], state: Any, snap: Any) -> None:
        """Newest last; past ``keep``, the oldest entry never resumed from goes first, else the oldest."""

        self.entries = [e for e in self.entries if e[0] != ids] + [(ids, state, snap)]
        while len(self.entries) > self.keep:
            self._drop(self.entries[:-1])

    def evict(self) -> bool:
        """Memory is short: drop the entry ``add`` would drop next; False when none is left."""

        if not self.entries:
            return False
        self._drop(self.entries)
        return True

    def _drop(self, among: list) -> None:
        cold = [e for e in among if tuple(e[0]) not in self.hit]
        gone = cold[0] if cold else among[0]
        self.entries = [e for e in self.entries if e is not gone]
        self.hit &= {tuple(e[0]) for e in self.entries}
