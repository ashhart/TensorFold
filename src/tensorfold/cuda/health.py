"""The CUDA server's /health counters and its /metrics exposition: finished requests' own engine stats, plus
live replies read off the rounds, the queues and each request's own timings."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

from tensorfold.server.metrics import (E2E_REQUEST_LATENCY_BUCKETS, TIME_TO_FIRST_TOKEN_BUCKETS, Gauge, Histogram,
                                       counter_lines)

STATS = {"prefill_s": "prefill_seconds_total", "decode_s": "decode_seconds_total", "cached": "cached_tokens_total",
         "rounds": "rounds_total", "drafted": "drafted_total", "accepted": "accepted_total"}
_MADE = threading.Lock()


class Request:
    """One running request: its prompt length and the server's own list of its reply tokens (only ever read here)."""

    def __init__(self, prompt: int, out: list[int]) -> None:
        self.prompt, self.out, self.stats = prompt, out, None
        self.started = 0.0                 # when the request began; the TTFT and e2e clocks start here
        self.first = 0.0                   # when its first token arrived (0: none yet)


class Health:
    """Totals of finished requests and the requests running now; the engine's rounds never call in here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set[Request] = set()
        self.totals: dict[str, float] = dict.fromkeys(("requests_total", "prompt_tokens_total",
                                                       "completion_tokens_total", *STATS.values()), 0)
        self.ttft = Histogram(TIME_TO_FIRST_TOKEN_BUCKETS)
        self.e2e = Histogram(E2E_REQUEST_LATENCY_BUCKETS)

    @contextmanager
    def running(self, prompt: int, out: list[int], started: float | None = None):
        """Count a request as running while its ``generate`` runs, then fold its reply and ``stats`` into the totals.

        ``started`` is the request's own arrival time: its TTFT and e2e latencies are measured from it.
        """

        request = Request(prompt, out)
        request.started = float(started) if started is not None else time.perf_counter()
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
        if request.first > 0.0:
            self.ttft.observe(request.first - request.started)
        self.e2e.observe(time.perf_counter() - request.started)

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

    def _kv_pools(self, app) -> list[tuple[str, int, int]]:
        """Each stream's own KV pool as (label, rows held, rows the pool holds).

        The engines give no shared pool to take a fraction of: a concurrent engine sizes each stream's attention
        caches to a per-stream row count, and a serialized engine keeps one pool for its one request. Read, never
        locked, as the decoder's table sizes; a serial engine's pool is only reported while a request runs it.
        """

        engine = getattr(app, "engine", None)
        if engine is None:
            return []
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            decoder = getattr(scheduler, "decoder", None)
            if decoder is None:
                return []
            bound = int(getattr(decoder, "context", 0) or getattr(decoder, "capacity", 0) or 0)
            table = getattr(decoder, "streams", None)
            if isinstance(table, dict):
                streams = list(table.values())
            else:
                streams = list(table or [])
            streams += list(getattr(decoder, "filling", []))
        else:
            if not self.live:
                return []
            state = getattr(getattr(engine, "e", None), "st", None) or getattr(engine, "e", None)
            if state is None:
                return []
            bound = 0
            streams = [state]
        pools: list[tuple[str, int, int]] = []
        for stream in streams:
            state = getattr(stream, "st", None)
            if state is None:
                state = stream                                # a serial engine's one pool is its engine itself
            used = getattr(state, "pos", None)
            used = int(used) if used is not None else len(getattr(stream, "context", ()))
            pool = int(getattr(state, "limit", 0) or getattr(state, "capacity", 0) or getattr(state, "max_len", 0)
                        or bound)
            if pool > 0 and 0 <= used <= pool:
                pools.append((str(getattr(stream, "sid", "0")), used, pool))
        return pools

    def exposition(self, app) -> str:
        """The server's state in Prometheus text format (0.0.4): /health's counters, the queues and the latencies."""

        snap = self.snapshot(app)
        engine = getattr(app, "engine", None)
        scheduler = getattr(engine, "scheduler", None) if engine is not None else None
        if scheduler is not None and getattr(scheduler, "decoder", None) is not None:
            # the engine's own lanes, not the app's list: a submitted request is admitted or queued, never both
            decoder = scheduler.decoder
            running = len(getattr(decoder, "streams", ())) + len(getattr(decoder, "filling", ()))
            queue = getattr(scheduler, "waiting", None)
            waiting = queue.qsize() if queue is not None else 0
        else:
            running = snap["requests_running"]
            turns = getattr(app, "turns", None)
            waiting = int(getattr(turns, "waiting", 0)) if turns is not None else 0
        lines: list[str] = []
        lines += Gauge(running).render("tensorfold:num_requests_running", "Number of running requests")
        lines += Gauge(waiting).render("tensorfold:num_requests_waiting", "Number of requests waiting to be processed")
        lines += counter_lines("tensorfold:prompt_tokens_total", "Number of prefill tokens processed",
                               snap["prompt_tokens_total"])
        lines += counter_lines("tensorfold:generation_tokens_total", "Number of generation tokens processed",
                                snap["completion_tokens_total"])
        for label, used, pool in self._kv_pools(app):
            lines += Gauge(used / pool).render("tensorfold:kv_cache_usage_perc", "KV cache usage percentage, in [0,1]",
                                                {"stream": label})
        lines += counter_lines("tensorfold:spec_decode_num_draft_tokens_total", "SpecDecoding: Number of draft tokens",
                                snap["drafted_total"])
        lines += counter_lines("tensorfold:spec_decode_num_accepted_tokens_total",
                                "SpecDecoding: Number of accepted tokens", snap["accepted_total"])
        lines += self.ttft.render("tensorfold:time_to_first_token_seconds", "Latency until first output")
        lines += self.e2e.render("tensorfold:e2e_request_latency_seconds", "E2E request latency")
        return "\n".join(lines) + "\n"

def of(app) -> Health:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("health")
        if found is None:
            found = app.__dict__["health"] = Health()
        return found


def exposition(app) -> str:
    """The app's state for ``GET /metrics``, in Prometheus text format (0.0.4)."""

    return of(app).exposition(app)


__all__ = ["Health", "Request", "exposition", "of"]
