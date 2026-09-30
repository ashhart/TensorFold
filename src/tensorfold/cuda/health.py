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

    def __init__(self, prompt: int, out: list[int]) -> None:
        self.prompt, self.out, self.stats = prompt, out, None
        self.started_at = time.perf_counter()
        self.first_token_at = 0.0
        self.admitted = False
        self.processed_prompt = False

    def admit(self, admitted: bool = True) -> None:
        self.admitted = bool(admitted)
        if self.admitted:
            self.processed_prompt = True

    def mark_tokens(self, tokens: list[int]) -> None:
        if tokens and not self.first_token_at:
            self.first_token_at = time.perf_counter()


class Health:
    """Totals of finished requests and the requests running now; the engine's rounds never call in here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set[Request] = set()
        self.totals: dict[str, float] = dict.fromkeys(("requests_total", "prompt_tokens_total",
                                                       "completion_tokens_total", *STATS.values()), 0)
        self.metrics = metrics.Metrics()

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
        prompt = request.prompt if request.processed_prompt else 0
        t["requests_total"] += 1
        t["prompt_tokens_total"] += prompt
        t["completion_tokens_total"] += len(request.out)
        for key, name in STATS.items():
            value = (request.stats or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                t[name] += value
        stats = request.stats or {}
        self.metrics.observe(
            prompt_tokens=prompt,
            generation_tokens=len(request.out),
            latency=max(0.0, time.perf_counter() - request.started_at),
            ttft=(max(0.0, request.first_token_at - request.started_at) if request.first_token_at else None),
            mtp_drafted=int(stats.get("mtp_drafted") or 0),
            mtp_accepted=int(stats.get("mtp_accepted") or 0),
        )

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

    def prometheus(self, app: Any) -> bytes:
        """Prometheus exposition of these same counters plus live scheduler state."""

        with self.lock:
            live = tuple(self.live)
            counters = {
                "prompt_tokens_total": self.totals["prompt_tokens_total"],
                "generation_tokens_total": self.totals["completion_tokens_total"] + sum(len(r.out) for r in live),
            }
        running, waiting, pools = metrics.cuda_state(app, live)
        return self.metrics.render(
            running=running,
            waiting=waiting,
            pools=pools,
            counters=counters,
        )


def of(app) -> Health:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("health")
        if found is None:
            found = app.__dict__["health"] = Health()
        return found


__all__ = ["Health", "Request", "of"]
