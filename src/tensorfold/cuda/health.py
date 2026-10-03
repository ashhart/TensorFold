"""The CUDA server's /health counters: finished requests' own engine stats, plus live replies read off the rounds."""

from __future__ import annotations

import os
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
        body.update(progress(scheduler, decoder))
        body["ok"] = body["fatal"] is None and not body["stalled"]
        window = getattr(app, "effective_context_window", None)
        if window:
            body["context_length"] = int(window)
        return body


def stall_seconds() -> float:
    """``TF_STALL_S``: how long one engine call (an admission, a round, a finish) may run before /health calls the
    engine stalled; 0 (the default) never does."""

    try:
        return max(0.0, float(os.environ.get("TF_STALL_S") or 0))
    except ValueError:
        return 0.0


def progress(scheduler, decoder) -> dict[str, Any]:
    """Whether the engine can still serve: ``fatal`` (a decoder whose ranks fell out of step: every later request
    fails until both restart) and ``stalled`` (one engine call running past ``TF_STALL_S``: a hung collective or
    rank). The idea of a fatal field and forward-progress checks follows the GLM Spark engine's health.py of
    deepseek-v41-tensorfold-spark (Jay Leaton); no code is copied. Only engine calls are timed, never a request's
    wait in the queue, so a full queue behind long replies is not a stall."""

    broken = getattr(decoder, "broken", None)
    now = time.monotonic()
    since = getattr(scheduler, "call_since", None)
    age = now - since if isinstance(since, (int, float)) else None
    limit = stall_seconds()
    last = getattr(scheduler, "last_round", None)
    return {"fatal": repr(broken) if broken else None, "stalled": bool(limit and age is not None and age > limit),
            "call_age_s": round(age, 3) if age is not None else None,
            "last_round_age_s": round(now - last, 3) if isinstance(last, (int, float)) else None}


def status(app) -> tuple[int, dict[str, Any]]:
    """/health's (code, body): ``TF_HEALTH=strict`` answers 503 when the engine is not ok (for a watchdog or a load
    balancer); otherwise always 200, the body saying what it found."""

    body = of(app).snapshot(app)
    strict = (os.environ.get("TF_HEALTH") or "").strip().lower() == "strict"
    return (503 if strict and not body["ok"] else 200), body


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


__all__ = ["Health", "Request", "of", "progress", "stall_seconds", "status"]
