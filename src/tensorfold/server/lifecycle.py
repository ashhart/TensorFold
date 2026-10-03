"""Drain accepted requests before releasing a runtime; publish a wake only after restoration succeeds."""

from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Callable


class LifecycleError(RuntimeError):
    """An HTTP lifecycle refusal without references to failed runtime allocations."""

    def __init__(self, message: str, *, status: int = 409, code: str = "model_lifecycle_error") -> None:
        super().__init__(message)
        self.status, self.code = status, code


def _attempt(operation: Callable[[], None]) -> str | None:
    """Return only an error string, so exception frames die before failed-load cleanup runs."""

    try:
        operation()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


class Lifecycle:
    """One replaceable runtime; callbacks never retain its weights outside the live app."""

    def __init__(self, *, release: Callable[[], None], restore: Callable[[], None],
                 cleanup: Callable[[], None], drain_timeout: float = 120.0,
                 preflight: Callable[[], None] | None = None) -> None:
        if not math.isfinite(drain_timeout) or drain_timeout <= 0:
            raise ValueError("sleep drain timeout must be finite and greater than zero")
        self._release, self._restore, self._cleanup = release, restore, cleanup
        self._preflight = preflight
        self._timeout = float(drain_timeout)
        self._condition = threading.Condition()
        self._local = threading.local()
        self._state, self._active, self._level = "awake", 0, None
        self._last_error: str | None = None

    def snapshot(self) -> dict:
        with self._condition:
            return {"state": self._state, "ready": self._state == "awake",
                    "is_sleeping": self._state == "sleeping", "level": self._level,
                    "active_requests": self._active, "last_error": self._last_error}

    @contextmanager
    def admit(self):
        """Cover preparation through final response; nested API translations share the outer lease."""

        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return
        with self._condition:
            if self._state != "awake":
                raise LifecycleError(f"model is {self._state}; wake it before sending inference requests",
                                     status=503, code="model_not_ready")
            self._active += 1
            self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _set(self, state: str, error: str | None = None) -> None:
        with self._condition:
            self._state, self._last_error = state, error
            self._condition.notify_all()

    def sleep(self, level: int = 2) -> dict:
        """Close admission, drain its finite accepted set and release Level 2 storage."""

        if type(level) is not int or level != 2:
            raise LifecycleError("this runtime supports sleep level 2 only", status=400,
                                 code="unsupported_sleep_level")
        if getattr(self._local, "depth", 0):
            raise LifecycleError("cannot sleep from inside an admitted request")
        with self._condition:
            if self._state == "sleeping":
                return self.snapshot()
            if self._state != "awake":
                raise LifecycleError(f"cannot sleep while model is {self._state}")
            self._set("draining")
            deadline = time.monotonic() + self._timeout
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._set("awake", "request drain timed out; runtime remains awake")
                    raise LifecycleError(self._last_error, code="sleep_drain_timeout")
                self._condition.wait(remaining)
        if self._preflight is not None:
            error = _attempt(self._preflight)
            if error is not None:
                self._set("awake", error)
                raise LifecycleError(f"sleep refused; runtime remains awake: {error}",
                                     code="model_sleep_preflight_failed")
        error = _attempt(self._release)
        if error is not None:
            self._set("error", error)
            raise LifecycleError(f"runtime release failed; restart the server: {error}", status=500)
        with self._condition:
            self._level = level
            self._set("sleeping")
            return self.snapshot()

    def wake_up(self) -> dict:
        """Keep requests out until weights, workers and callbacks have all been restored."""

        with self._condition:
            if self._state == "awake":
                return self.snapshot()
            if self._state != "sleeping":
                raise LifecycleError(f"cannot wake while model is {self._state}")
            self._set("waking")
        error = _attempt(self._restore)
        if error is not None:
            cleanup_error = _attempt(self._cleanup)
            if cleanup_error is not None:
                self._set("error", f"{error}; cleanup failed: {cleanup_error}")
                raise LifecycleError(f"failed wake cleanup; restart the server: {cleanup_error}", status=500)
            self._set("sleeping", error)
            raise LifecycleError(f"wake failed; model remains sleeping: {error}", status=503,
                                 code="model_wake_failed")
        with self._condition:
            self._level = None
            self._set("awake")
            return self.snapshot()


__all__ = ["Lifecycle", "LifecycleError"]
