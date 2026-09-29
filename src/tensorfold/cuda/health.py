"""The CUDA server's /health counters: finished requests' own engine stats, plus live replies read off the rounds."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

STATS = {"prefill_s": "prefill_seconds_total", "decode_s": "decode_seconds_total", "cached": "cached_tokens_total",
         "rounds": "rounds_total", "drafted": "drafted_total", "accepted": "accepted_total"}
_MADE = threading.Lock()


class Request:
    """One running request: its prompt length and the server's own list of its reply tokens (only ever read here)."""

    def __init__(self, prompt: int, out: list[int]) -> None:
        self.prompt, self.out, self.stats = prompt, out, None


class Health:
    """Totals of finished requests and the requests running now; the engine's rounds never call in here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set[Request] = set()
        self.totals: dict[str, float] = dict.fromkeys(("requests_total", "prompt_tokens_total",
                                                       "completion_tokens_total", *STATS.values()), 0)

    @contextmanager
    def running(self, prompt: int, out: list[int]):
        """Count a request as running while its ``generate`` runs, then fold its reply and ``stats`` into the totals."""

        request = Request(prompt, out)
        with self.lock:
            self.live.add(request)
        try:
            yield request
        finally:
            with self.lock:
                self.live.discard(request)
                self._fold(request)

    def _fold(self, request: Request) -> None:
        t = self.totals
        t["requests_total"] += 1
        t["prompt_tokens_total"] += request.prompt
        t["completion_tokens_total"] += len(request.out)
        for key, name in STATS.items():
            value = (request.stats or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                t[name] += value

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
        return found


__all__ = ["Health", "Request", "of"]
