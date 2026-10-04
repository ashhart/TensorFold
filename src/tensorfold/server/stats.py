"""One JSON snapshot of the live numbers behind ``--dashboard``, sampled once a second for the browser page.

The sampler thread only reads what the scheduler and admission already compute; nothing here takes engine
state. It stays off unless the server starts it (``--dashboard``), and a route reads the newest snapshot
rather than the engine, so a dashboard tab costs a decode of a few hundred bytes, never a lock.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable

WINDOW = 300.0          # seconds the decode min/max/mean average over (the page's "5 minute" tile)
EVERY = 1.0             # seconds between samples
HISTORY_MAX = 300       # points the page may plot (one a sample, the window's worth)
PREFILL_MAX = 256       # completed prefill chunks kept for the average rate
HISTORY_MAX_REQUESTS = 32   # completed requests the page's recent list shows


class RollingRate:
    """Decode tok/s as samples land: ``min``/``max``/``mean`` over the last ``window`` seconds."""

    def __init__(self, window: float = WINDOW, clock: Callable[[], float] = time.time) -> None:
        self.window, self.clock = float(window), clock
        self._points: collections.deque[tuple[float, float]] = collections.deque()
        self._lock = threading.Lock()

    def add(self, tok_s: float) -> None:
        if tok_s > 0:
            with self._lock:
                self._points.append((self.clock(), float(tok_s)))
                self._evict(self.clock())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._evict(self.clock())
            values = [rate for _, rate in self._points]
            return {"window_s": self.window, "count": len(values),
                    "min": min(values) if values else None, "max": max(values) if values else None,
                    "mean": (sum(values) / len(values)) if values else None,
                    "history": [{"t": t, "tok_s": rate} for t, rate in self._points]}

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()
        while len(self._points) > HISTORY_MAX:
            self._points.popleft()


class PrefillAverage:
    """Completed prefill chunks: their mean tok/s and the newest chunk's own rate."""

    def __init__(self, capacity: int = PREFILL_MAX) -> None:
        self._rates: collections.deque[float] = collections.deque(maxlen=int(capacity))
        self._lock = threading.Lock()

    def add(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0:
            with self._lock:
                self._rates.append(int(tokens) / float(seconds))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"avg_tok_s": (sum(self._rates) / len(self._rates)) if self._rates else None,
                    "last_tok_s": self._rates[-1] if self._rates else None, "count": len(self._rates)}


class StatsCollector:
    """Samples ``app``'s live numbers every ``every`` seconds; the ``/stats`` route serves the newest snapshot."""

    def __init__(self, app: Any, *, every: float = EVERY, clock: Callable[[], float] = time.time) -> None:
        self.app, self.every, self.clock = app, float(every), clock
        self.decode = RollingRate(clock=clock)
        self.prefill_average = PrefillAverage()
        self._lock = threading.Lock()
        self._last: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._tick, name="tensorfold-dashboard", daemon=True)
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=HISTORY_MAX_REQUESTS)

    def start(self) -> "StatsCollector":
        """Begin sampling and hand this collector to the scheduler, whose fill loop feeds the prefill average."""

        scheduler = getattr(self.app, "scheduler", None)
        if scheduler is not None:
            scheduler.stats = self
        self.app.stats = self                   # the routes find the collector through the app, like the live line
        self.sample()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, Any]:
        """A fresh build, remembered as the thread's newest: a poll never serves a sample older than itself."""

        built = self._build()
        with self._lock:
            self._last = built
        return built

    def sample(self) -> dict[str, Any]:
        """One sampler tick; the 1 Hz thread calls this, and tests call it to step time deterministically."""

        return self.snapshot()

    def record_prefill(self, tokens: int, seconds: float) -> None:
        """A prompt chunk prefilled (the scheduler's own call site; a raising page must never stop a serve)."""

        try:
            self.prefill_average.add(tokens, seconds)
        except Exception:  # noqa: BLE001 - the dashboard is never a way to fail a request
            pass

    def record_request(self, summary: dict[str, Any]) -> None:
        """A finished request's stats, for the page's recent list (scheduler thread; a raising page must not)."""

        try:
            with self._lock:
                self._history.appendleft(summary)
        except Exception:  # noqa: BLE001
            pass

    def _tick(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - a status sampler must never take the server down
                pass

    def _build(self) -> dict[str, Any]:
        app, scheduler = self.app, getattr(self.app, "scheduler", None)
        live = {"decode_tok_s": None, "prefill_tok_s": None}
        connections: dict[str, Any] = {"active": 0, "waiting": 0, "filling": False}
        context: dict[str, Any] = {"used_tokens": 0, "window_tokens": int(getattr(app, "context_window", 0) or 0),
                                   "cached_tokens": 0}
        if scheduler is not None:
            live["decode_tok_s"] = scheduler.decoded.rate()
            live["prefill_tok_s"] = scheduler.prefilled.rate()
            connections = {"active": scheduler.active, "waiting": scheduler.waiting,
                           "filling": bool(scheduler.filling)}
            if live["decode_tok_s"] > 0:
                self.decode.add(live["decode_tok_s"])
            context.update(scheduler.context_snapshot())
        speculative = scheduler.stream_spec_snapshot() if scheduler is not None else {}
        drafted = speculative.get("drafted", 0)
        return {"ts": self.clock(), "model": getattr(app, "served_name", ""),
                "warming": bool(getattr(app, "warming", False)),
                "connections": connections, "live": live, "context": context,
                "fill_progress": scheduler.fill_progress if scheduler is not None else [],
                "memory": _memory(app), "decode_5m": self.decode.snapshot(),
                "prefill": self.prefill_average.snapshot(), "requests": list(self._history),
                "speculative": {**speculative,
                                "acceptance_rate": (speculative["accepted"] / drafted) if drafted else None}}


def _memory(app: Any) -> dict[str, int] | None:
    """MLX's memory and the process's budget; None on a backend that reports neither."""

    prompt_memory = getattr(app, "prompt_memory", None)
    if prompt_memory is not None:
        memory = dict(prompt_memory.memory_snapshot())
        physical = _physical_bytes()
        if physical:
            memory["physical"] = physical
        return memory
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001 - the CUDA server, or no MLX at all
        return None
    try:
        return {"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory()),
                "peak": int(mx.get_peak_memory()), "physical": _physical_bytes()}
    except Exception:  # noqa: BLE001
        return None


def _physical_bytes() -> int:
    try:
        from tensorfold.server.memory_budget import physical_memory_bytes

        return int(physical_memory_bytes())
    except Exception:  # noqa: BLE001 - the page shows what it has
        return 0


__all__ = ["EVERY", "HISTORY_MAX", "HISTORY_MAX_REQUESTS", "PREFILL_MAX", "WINDOW", "PrefillAverage",
           "RollingRate", "StatsCollector"]