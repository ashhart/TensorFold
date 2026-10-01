"""Stream caches take memory as they grow: the room the startup plan left them, what they hold, and a reserve."""

from __future__ import annotations

from typing import Callable


class NoRoom(RuntimeError):
    """A request that can't start until a live stream finishes and frees its caches."""


class MemoryGate:
    """``room`` bytes for every stream's growing caches; ``fits`` also asks the host (MemAvailable on a unified GPU)."""

    def __init__(self, room: int, reserve: int, live: Callable[[], int] | None = None) -> None:
        self.room, self.reserve, self.held = int(room), int(reserve), 0
        self.live = live
        self.waits = self.ends = 0

    def fits(self, extra: int) -> bool:
        """Whether ``extra`` more bytes can be allocated now (a resize holds its old buffers until the copy lands)."""

        if self.held + int(extra) > self.room - self.reserve:
            return False
        return self.live is None or self.live() >= int(extra) + self.reserve

    def take(self, extra: int) -> None:
        self.held += int(extra)

    def give(self, freed: int) -> None:
        self.held = max(0, self.held - int(freed))


def torch_live(torch, available: Callable) -> Callable[[], int]:
    """What the host has free now plus what torch's allocator holds freed (it reuses those without asking)."""

    return lambda: int(available(torch)) + int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())


__all__ = ["MemoryGate", "NoRoom", "torch_live"]
