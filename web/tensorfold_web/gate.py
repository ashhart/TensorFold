"""One reservation for everything that changes the engine or the files it serves.

Starting or stopping the engine, a comparison of configurations (for its whole length), a settings change that
touches the engine, a Bench run's start, a download's start and a model's removal each take the gate; whoever
finds it taken is refused at once with what holds it, rather than waiting or racing. A check followed by an action
is thus one step: nothing can start in between.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class Busy(RuntimeError):
    pass


class Gate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.holder: str | None = None

    def take(self, who: str) -> None:
        """Take the gate for ``who`` or raise :class:`Busy`; :meth:`give` releases it, from any thread."""

        if not self._lock.acquire(blocking=False):
            raise Busy(f"{self.holder or 'another change'} is going; wait or cancel it")
        self.holder = who

    def give(self) -> None:
        self.holder = None
        self._lock.release()

    @contextmanager
    def hold(self, who: str) -> Iterator[None]:
        self.take(who)
        try:
            yield
        finally:
            self.give()
