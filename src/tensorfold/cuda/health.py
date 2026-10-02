"""The CUDA server's /health counters: finished requests' own engine stats, plus live replies read off the rounds."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

from tensorfold.server import metrics

STATS = {"prefill_s": "prefill_seconds_total", "decode_s": "decode_seconds_total", "cached": "cached_tokens_total",
         "rounds": "rounds_total", "drafted": "drafted_total", "accepted": "accepted_total"}
_MADE = threading.Lock()


class Request:
    """One running request: its prompt length and the server's own list of its reply tokens (only ever read here)."""

    def __init__(self, prompt: int, out: list[int], arrived: float | None = None) -> None:
        self.prompt, self.out, self.stats = prompt, out, None
        self.started = time.perf_counter() if arrived is None else float(arrived)
        self.first: float | None = None

    def saw(self) -> None:
        """The first generated token has landed in ``out``."""

        if self.first is None and self.out:
            self.first = time.perf_counter()


class Health:
    """Totals of finished requests and the requests running now; the engine's rounds never call in here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set[Request] = set()
        self.totals: dict[str, float] = dict.fromkeys(("requests_total", "prompt_tokens_total",
                                                       "completion_tokens_total", *STATS.values()), 0)

    @contextmanager
    def running(self, prompt: int, out: list[int], arrived: float | None = None):
        """Count a request as running while its ``generate`` runs, then fold its reply and ``stats`` into the totals."""

        request = Request(prompt, out, arrived)
        with self.lock:
            self.live.add(request)
        try:
            yield request
        finally:
            with self.lock:
                self.live.discard(request)
                self._fold(request)
            self._metrics(request)

    def _fold(self, request: Request) -> None:
        t = self.totals
        t["requests_total"] += 1
        t["prompt_tokens_total"] += request.prompt
        t["completion_tokens_total"] += len(request.out)
        for key, name in STATS.items():
            value = (request.stats or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                t[name] += value

    def _metrics(self, request: Request) -> None:
        stats = request.stats or {}
        metrics.note(getattr(self, "app", None), prompt=request.prompt, generation=len(request.out),
                     drafted=_stat(stats, "drafted"), accepted=_stat(stats, "accepted"),
                     latency=max(0.0, time.perf_counter() - request.started),
                     ttft=(request.first - request.started) if request.first is not None else None)

    def snapshot(self, app) -> dict[str, Any]:
        """The counters now: finished totals, live replies' tokens so far, and a concurrent engine's streams."""

        with self.lock:
            body: dict[str, Any] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.totals.items()}
            body["completion_tokens_total"] += sum(len(r.out) for r in self.live)
            running = len(self.live)
        body = {"ok": True, "backend": "tensorfold", "busy": running > 0, "requests_running": running, **body}
        scheduler = getattr(getattr(app, "engine", None), "scheduler", None)     # /health answers whatever the app
        decoder = getattr(scheduler, "decoder", None)
        if decoder is not None:                         # read, never locked: sizes of the decoder's own tables
            body["streams"] = {"decoding": len(getattr(decoder, "streams", ())),
                               "prefilling": len(getattr(decoder, "filling", ())), "max": scheduler.max_streams}
            more = getattr(decoder, "health", None)       # a decoder's own counts (GLM: paused streams, its pool)
            if callable(more):
                try:
                    extra = dict(more())
                except Exception:                         # noqa: BLE001  (a table changing under the read)
                    extra = {}
                body["streams"].update(extra.pop("streams", {}))
                body.update(extra)
        window = getattr(app, "effective_context_window", None)
        if window:
            body["context_length"] = int(window)
        return body


def of(app) -> Health:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("health")
        if found is None:
            found = app.__dict__["health"] = Health()
        found.app = app
        return found


def _stat(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


__all__ = ["Health", "Request", "of"]
