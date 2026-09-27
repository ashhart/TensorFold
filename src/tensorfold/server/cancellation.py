import select
import socket
import threading
from typing import Any, Callable


class RequestCancelled(Exception):
    pass


class Cancellation:
    def __init__(self, disconnected: Callable[[], bool] | None = None):
        self._event = threading.Event()
        self._disconnected = disconnected

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        if not self._event.is_set() and self._disconnected is not None and self._disconnected():
            self.cancel()
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise RequestCancelled("request cancelled")


def socket_cancellation(connection: socket.socket) -> Cancellation:
    def disconnected() -> bool:
        try:
            if connection.fileno() < 0:
                return True
            ready, _, _ = select.select([connection], [], [], 0)
            if not ready:
                return False
            return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True
        except ValueError:
            return False

    return Cancellation(disconnected)


class PrefillGuard:
    def __init__(self, cancellation: Cancellation, memory: Any = None):
        self.cancellation, self.memory = cancellation, memory

    def before_chunk(self, cache: Any, tokens: int) -> None:
        self.cancellation.check()
        if self.memory is not None:
            self.memory.before_chunk(cache, tokens)
        self.cancellation.check()

    def after_chunk(self, cache: Any, tokens: int) -> None:
        self.cancellation.check()
        if self.memory is not None:
            self.memory.after_chunk(cache, tokens)
        self.cancellation.check()

    def allow_checkpoint(self, cache: Any) -> bool:
        self.cancellation.check()
        return self.memory is None or self.memory.allow_checkpoint(cache)
