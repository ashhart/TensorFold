"""Drafters propose tokens and depth policies pick how many to verify; verification alone decides what a lane keeps."""
# Every rank runs these with the same inputs in the same order, so ranks agree on every window without sending drafts.

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

DECAY = 0.97  # evidence at a draft position fades by this each round that verifies it
PRIOR = 2.0  # the calibration's prior weight, in summed confidence
AHEAD = 0.8  # a position's acceptance before any evidence, without confidences
Q_MAX = 0.995  # a calibrated chance never reaches 1
F_MIN, F_MAX = 0.2, 3.0  # the confidence correction's range


@dataclass
class Proposals:
    """Each lane's drafted tokens after its pending one, and the drafter's chance for each (None: it gives none)."""

    tokens: list[list[int]]
    confidence: list[list[float]] | None = None


class Drafter(Protocol):
    """Up to ``block`` tokens a lane; lane state that must survive a resume belongs in the forward's snapshot."""

    block: int

    def reset(self, lane: int, prompt: Sequence[int]) -> None: ...

    def observe(self, lane: int, tokens: Sequence[int]) -> None: ...

    def propose(
        self,
        lanes: Sequence[int],
        pendings: Sequence[int],
        starts: Sequence[int],
        depths: Sequence[int],
        samplings: Sequence[Any],
    ) -> Proposals: ...


class WindowCosts:
    """Milliseconds of a verify window by its total rows (any lanes), one table on every rank; ``draft``: one pass."""

    US = 1000

    def __init__(self, ms: Sequence[float], draft: float = 0.0) -> None:
        if not ms:
            raise ValueError("window costs need at least the one-row window")
        self.table = [float(ms[0])]
        for v in ms[1:]:
            self.table.append(max(float(v), self.table[-1]))
        self.draft = float(draft)

    @classmethod
    def linear(cls, fixed: float, row: float, rows: int = 16, draft: float = 0.0) -> WindowCosts:
        return cls([fixed + row * r for r in range(1, rows + 1)], draft)

    def ms(self, rows: int) -> float:
        """A window of ``rows`` rows; past the table, on the line of its last two entries."""

        t, r = self.table, max(1, int(rows))
        if r <= len(t):
            return t[r - 1]
        slope = t[-1] - t[-2] if len(t) > 1 else 0.0
        return t[-1] + slope * (r - len(t))

    def encode(self) -> list[int]:
        return [round(v * self.US) for v in (self.draft, *self.table)]

    @classmethod
    def decode(cls, ints: Sequence[int]) -> WindowCosts:
        values = [int(v) / cls.US for v in ints]
        return cls(values[1:], values[0])


class DepthPolicy(Protocol):
    """How many of a lane's proposed drafts its next window verifies (at most ``most``)."""

    def reset(self, lane: int) -> None: ...

    def depth(self, lane: int, confidence: Sequence[float] | None, most: int) -> int: ...

    def record(self, lane: int, verified: int, kept: int) -> None: ...


class StaticDepth:
    """Always ``k`` drafts (fewer when fewer are proposed or allowed)."""

    def __init__(self, k: int) -> None:
        self.k = int(k)

    def reset(self, lane: int) -> None:
        return None

    def depth(self, lane: int, confidence: Sequence[float] | None, most: int) -> int:
        return max(0, min(self.k, int(most)))

    def record(self, lane: int, verified: int, kept: int) -> None:
        return None


def expected_tokens(qs: Sequence[float]) -> list[float]:
    """[E(0) .. E(len(qs))]: tokens a round verifying k drafts keeps on average (q_j: j kept if those before were)."""

    out, run, e = [1.0], 1.0, 1.0
    for q in qs:
        run *= q
        e += run
        out.append(e)
    return out


def best_k(qs: Sequence[float], ms, rate: float, least: int = 1) -> int:
    """The k from ``least`` to len(qs) maximising E(k) - rate x ms(k + 1); ties to the smaller k."""

    es = expected_tokens(qs)
    best, value = 0, float("-inf")
    for k in range(min(max(least, 0), len(qs)), len(qs) + 1):
        v = es[k] - rate * ms(k + 1)
        if v > value + 1e-12:
            best, value = k, v
    return best


class ExpectedRate:
    """The k maximising E(k) - rate x ms(k) at the lane's recent rate: at the rate it achieves, the most tokens a ms."""

    def __init__(self, costs: WindowCosts, *, positions: int = 16, rounds: int = 16, least: int = 1) -> None:
        self.costs, self.rounds, self.least = costs, int(rounds), int(least)
        self.kept = [0.0] * positions  # decayed kept drafts, summed confidences and trials by position
        self.prob = [0.0] * positions
        self.tried = [0.0] * positions
        self.lanes: dict[int, dict] = {}

    def reset(self, lane: int) -> None:
        self.lanes[lane] = {"rounds": deque(maxlen=self.rounds), "used": None}

    def _lane(self, lane: int) -> dict:
        if lane not in self.lanes:
            self.reset(lane)
        return self.lanes[lane]

    def rate(self, lane: int) -> float:
        r = self._lane(lane)["rounds"]
        if r:
            return sum(t for t, _ in r) / sum(ms for _, ms in r)
        return 2.0 / (self.costs.ms(2) + self.costs.draft)

    def q(self, j: int, p: float | None) -> float:
        """Draft j's chance to be kept given those before were: its confidence corrected, else the position's record."""

        j = min(j, len(self.kept) - 1)
        if p is None:
            return min((self.kept[j] + AHEAD * PRIOR) / (self.tried[j] + PRIOR), Q_MAX)
        factor = min(max((self.kept[j] + PRIOR) / (self.prob[j] + PRIOR), F_MIN), F_MAX)
        return min(max(float(p), 0.0) * factor, Q_MAX)

    def depth(self, lane: int, confidence: Sequence[float] | None, most: int) -> int:
        most = max(0, int(most))
        conf = list(confidence[:most]) if confidence is not None else [None] * most
        k = best_k([self.q(j, p) for j, p in enumerate(conf)], self.costs.ms, self.rate(lane), self.least)
        self._lane(lane)["used"] = conf[:k]
        return k

    def record(self, lane: int, verified: int, kept: int) -> None:
        st = self._lane(lane)
        used, st["used"] = st["used"] or [], None
        for j in range(min(verified, kept + 1, len(self.kept))):
            p = used[j] if j < len(used) else None
            self.kept[j] = self.kept[j] * DECAY + (1.0 if j < kept else 0.0)
            self.prob[j] = self.prob[j] * DECAY + (max(float(p), 0.0) if p is not None else 1.0)
            self.tried[j] = self.tried[j] * DECAY + 1.0
        st["rounds"].append((kept + 1, self.costs.ms(verified + 1) + (self.costs.draft if verified else 0.0)))

    def state(self) -> tuple:
        return (
            tuple(self.kept),
            tuple(self.prob),
            tuple(self.tried),
            tuple(sorted((k, tuple(v["rounds"])) for k, v in self.lanes.items())),
        )


class Lookup:
    """Prompt lookup: after the latest earlier match of the context's last ``n`` tokens, the tokens that followed it."""

    def __init__(self, block: int = 8, n: int = 3) -> None:
        self.block, self.n = int(block), int(n)
        self.context: dict[int, list[int]] = {}

    def reset(self, lane: int, prompt: Sequence[int]) -> None:
        self.context[lane] = [int(t) for t in prompt]

    def observe(self, lane: int, tokens: Sequence[int]) -> None:
        self.context[lane].extend(int(t) for t in tokens)

    def propose(self, lanes, pendings, starts, depths, samplings) -> Proposals:
        return Proposals([self._one(self.context[lane], int(d)) for lane, d in zip(lanes, depths)])

    def _one(self, ctx: list[int], most: int) -> list[int]:
        for n in range(min(self.n, len(ctx) - 1), 0, -1):
            tail = ctx[-n:]
            for at in range(len(ctx) - n - 1, -1, -1):
                if ctx[at : at + n] == tail:
                    return ctx[at + n : at + n + most]
        return []


__all__ = [
    "DepthPolicy",
    "Drafter",
    "ExpectedRate",
    "Lookup",
    "Proposals",
    "StaticDepth",
    "WindowCosts",
    "best_k",
    "expected_tokens",
]
