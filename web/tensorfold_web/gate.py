"""One reservation for everything that changes the engine or the files it serves."""

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
