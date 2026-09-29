"""Streams take memory as they grow: before each round the live streams' next growth fits, or the newest wait."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

HORIZON = 2048      # tokens of growth held ahead of a stream (the alternating KV's spare step)


@dataclass
class Plan:
    run: list[str] = field(default_factory=list)       # decode this round
    paused: list[str] = field(default_factory=list)    # the newest, waiting until memory frees; state untouched
    ended: list[str] = field(default_factory=list)     # stopped with an error so the older streams can finish


class StreamGate:
    """Reserve each stream's next ``horizon`` tokens, free short prefixes, then pause or end a stream."""

    def __init__(self, memory: Any, per_token: float, work: int, budget: int, horizon: int = HORIZON,
                 lanes: int = 1) -> None:
        self.memory = memory                 # the server's PromptMemory: MLX's use, freed buffers, retained prefixes
        self.per_token, self.work, self.lanes = float(per_token), int(work), max(1, int(lanes))
        self.budget, self.horizon = int(budget), int(horizon)
        self.waits = self.ends = 0

    def growth(self, now: int, most: int) -> int:
        """What a stream at ``now`` tokens that may reach ``most`` can grow by before the next check."""

        return int(min(max(0, int(most) - int(now)), self.horizon) * self.per_token)

    def need(self, streams: Sequence[tuple[str, int, int]]) -> int:
        work = -(-self.work * min(len(streams), self.lanes) // self.lanes)
        return int(self.memory._used()) + sum(self.growth(now, most) for _, now, most in streams) + work

    def _freeable(self) -> int:
        store = self.memory.store
        return int(self.memory.runtime.get_cache_memory()) + (store.nbytes if store is not None else 0)

    def plan(self, streams: Sequence[tuple[str, int, int]]) -> Plan:
        """``streams`` oldest first as (id, tokens now, tokens it may reach)."""

        running, paused = list(streams), []
        with self.memory._memory_lock:
            while True:
                # freed buffers and retained prefixes go first, only while what is left could make room (#74)
                while self.need(running) > self.budget and self.need(running) - self._freeable() <= self.budget:
                    if not self.memory._reclaim():
                        break
                if self.need(running) <= self.budget or len(running) == 1:
                    break
                paused.insert(0, running.pop()[0])          # then the newest waits
        plan = Plan([sid for sid, _, _ in running], paused)
        if paused and self.need(running) > self.budget:     # even the oldest alone can't grow: the newest ends
            plan.ended, plan.paused = [paused[-1]], paused[:-1]
        self.waits += bool(plan.paused)
        self.ends += len(plan.ended)
        return plan


__all__ = ["HORIZON", "Plan", "StreamGate"]
