"""Prometheus text for GET /metrics. Both servers scrape one module; gauges are read at scrape time.

Each reading is taken on its own, not as one atomic snapshot. vLLM-named families carry an identical reading of the
same state under the name a vLLM dashboard already knows, so a dashboard copied from vLLM fills its panels by
swapping the ``tensorfold:`` prefix for the name alone. A family the server cannot count honestly is left out,
never emitted at a fabricated zero.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from tensorfold.server.memory_budget import process_footprint

PREFIX = "tensorfold:"
# Request and time-to-first-token histograms share these upper edges. +Inf is added when rendered.
BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
_MADE = threading.Lock()
_local = threading.local()


class Histogram:
    """Counts in one bucket each. Render adds them up into Prometheus's cumulative buckets."""

    def __init__(self) -> None:
        self.counts = [0] * (len(BUCKETS) + 1)
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.n += 1
        self.total += value
        for i, edge in enumerate(BUCKETS):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def copy(self) -> "Histogram":
        other = Histogram()
        other.counts = list(self.counts)
        other.total, other.n = self.total, self.n
        return other


class Metrics:
    """Counters and histograms of finished requests. Gauges are not stored here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt = 0
        self.generation = 0
        self.drafted = 0
        self.accepted = 0
        self.latency = Histogram()
        self.ttft = Histogram()
        self.decode = Histogram()

    def add(self, *, prompt: int, generation: int, drafted: int, accepted: int,
            latency: float | None, ttft: float | None, decode: float | None = None) -> None:
        with self.lock:
            self.prompt += int(prompt)
            self.generation += int(generation)
            self.drafted += int(drafted)
            self.accepted += int(accepted)
            if latency is not None:
                self.latency.observe(latency)
            if ttft is not None:
                self.ttft.observe(ttft)
            if decode is not None:
                self.decode.observe(decode)


def of(app: Any) -> Metrics:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("metrics")
        if found is None:
            found = app.__dict__["metrics"] = Metrics()
        return found


def note(app: Any, *, prompt: int = 0, generation: int = 0, drafted: int = 0, accepted: int = 0,
         latency: float | None = None, ttft: float | None = None, decode: float | None = None) -> None:
    """Fold one finished request. A missing app is a no-op."""

    if app is None:
        return
    of(app).add(prompt=prompt, generation=generation, drafted=drafted, accepted=accepted,
                latency=latency, ttft=ttft, decode=decode)


def begin(app: Any, prompt: int, started: float) -> None:
    """The Mac request on this thread, from arrival, counted once when ``finish_request`` runs."""

    _local.armed = True
    _local.app = app
    _local.prompt = int(prompt)
    _local.generation = 0
    _local.started = float(started)
    _local.first = 0.0
    _local.job = None


def bind(job: Any) -> None:
    """The job whose stream holds this request's draft counts (a rerun replaces a preempted one)."""

    if getattr(_local, "armed", False):
        _local.job = job


def tokens(count: int, first: float) -> None:
    """Generated tokens so far, and the clock time of the first one."""

    if not getattr(_local, "armed", False):
        return
    _local.generation = int(count)
    if first and not _local.first:
        _local.first = float(first)


def finish_request() -> None:
    """Count the Mac request begun on this thread. Safe when none was begun."""

    if not getattr(_local, "armed", False):
        return
    _local.armed = False
    job = _local.job
    stream = getattr(job, "stream", None) if job is not None else None
    ended = time.perf_counter()
    ttft = (_local.first - _local.started) if _local.first else None
    # Decode runs from the first generated token to the end; a request that generated nothing has none.
    decode = max(0.0, ended - _local.first) if _local.first else None
    note(_local.app, prompt=_local.prompt, generation=_local.generation,
         drafted=int(getattr(stream, "drafted", 0) or 0),
         accepted=int(getattr(stream, "accepted", 0) or 0),
         latency=max(0.0, ended - _local.started), ttft=ttft, decode=decode)


def render(app: Any) -> str:
    """The scrape body, ending in a newline."""

    metrics = of(app)
    with metrics.lock:
        prompt, generation = metrics.prompt, metrics.generation
        drafted, accepted = metrics.drafted, metrics.accepted
        latency, ttft, decode = metrics.latency.copy(), metrics.ttft.copy(), metrics.decode.copy()
    running, waiting = _requests(app)
    pools = _pools(app)
    lines: list[str] = []
    _family(lines, "requests_running", "gauge", "Requests in prefill or decode.",
            [f"{PREFIX}requests_running {running}"])
    _family(lines, "requests_waiting", "gauge", "Requests queued or held until a lane is free.",
            [f"{PREFIX}requests_waiting {waiting}"])
    _family(lines, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.",
            [f"{PREFIX}prompt_tokens_total {prompt}"])
    _family(lines, "generation_tokens_total", "counter", "Generated tokens of finished requests.",
            [f"{PREFIX}generation_tokens_total {generation}"])
    _family(lines, "kv_cache_usage_ratio", "gauge",
            "Tokens in a stream cache divided by that stream's context window.",
            [f'{PREFIX}kv_cache_usage_ratio{{pool="{pool}"}} {_num(ratio)}' for pool, ratio in pools])
    _family(lines, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.",
            [f"{PREFIX}mtp_drafted_total {drafted}"])
    _family(lines, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.",
            [f"{PREFIX}mtp_accepted_total {accepted}"])
    _histogram(lines, "request_latency_seconds", "Seconds from arrival to the reply leaving.", latency)
    _histogram(lines, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", ttft)
    _histogram(lines, "request_decode_seconds",
               "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.",
               decode)
    # the process's own footprint, where the platform counts it (macOS): weights, caches and streams as one number
    footprint = process_footprint()
    if footprint is not None:
        _family(lines, "process_footprint_bytes", "gauge",
                "This process's physical footprint as the OS counts it, Metal buffers included; "
                "only where the platform reports one (macOS).",
                [f"{PREFIX}process_footprint_bytes {footprint}"])
    # vLLM names, identical values: a vLLM dashboard needs only the "tensorfold:" prefix swapped.
    _family(lines, "num_requests_running", "gauge",
            "Requests in prefill or decode. A mirror of tensorfold:requests_running.",
            [f"{PREFIX}num_requests_running {running}"])
    _family(lines, "num_requests_waiting", "gauge",
            "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.",
            [f"{PREFIX}num_requests_waiting {waiting}"])
    _family(lines, "kv_cache_usage_perc", "gauge",
            "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.",
            [f'{PREFIX}kv_cache_usage_perc{{stream="{pool}"}} {_num(ratio)}' for pool, ratio in pools])
    _family(lines, "spec_decode_num_draft_tokens_total", "counter",
            "Draft tokens verified on finished requests, this server's single draft counter.",
            [f"{PREFIX}spec_decode_num_draft_tokens_total {drafted}"])
    _family(lines, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.",
            [f"{PREFIX}spec_decode_num_accepted_tokens_total {accepted}"])
    _histogram(lines, "e2e_request_latency_seconds",
               "Seconds from arrival to the reply leaving, under vLLM's name.", latency)
    _histogram(lines, "request_decode_time_seconds",
               "Seconds a finished request spent decoding, under vLLM's name.", decode)
    # per-request event counts; a family is left out where this server doesn't count the event, never a fake zero
    disconnects, preempted = _endings(app)
    if disconnects is not None:
        _family(lines, "client_disconnections_total", "counter",
                "Requests a client left before the reply left the server.",
                [f"{PREFIX}client_disconnections_total {disconnects}"])
    if preempted is not None:
        _family(lines, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.",
                [f"{PREFIX}preemptions_total {preempted}"])
    return "\n".join(lines) + "\n"


def send(handler: Any, app: Any) -> None:
    """Write ``render`` as Prometheus text, version 0.0.4."""

    body = render(app).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True


def _family(lines: list[str], name: str, kind: str, help_text: str, samples: list[str]) -> None:
    full = PREFIX + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} {kind}")
    lines.extend(samples)


def _histogram(lines: list[str], name: str, help_text: str, hist: Histogram) -> None:
    full = PREFIX + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} histogram")
    cumulative = 0
    for edge, count in zip(BUCKETS, hist.counts):
        cumulative += count
        lines.append(f'{full}_bucket{{le="{_edge(edge)}"}} {cumulative}')
    lines.append(f'{full}_bucket{{le="+Inf"}} {cumulative + hist.counts[-1]}')
    lines.append(f"{full}_sum {_num(hist.total)}")
    lines.append(f"{full}_count {hist.n}")


def _requests(app: Any) -> tuple[int, int]:
    """(running, waiting). Prefilling Mac prompts are running and not yet in the active set."""

    scheduler = getattr(app, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "active") and hasattr(scheduler, "waiting"):
        filling = len(getattr(scheduler, "filling", None) or ())
        return int(scheduler.active) + filling, int(scheduler.waiting)
    engine = getattr(app, "engine", None)
    sched = getattr(engine, "scheduler", None) if engine is not None else None
    decoder = getattr(sched, "decoder", None) if sched is not None else None
    if decoder is not None and hasattr(decoder, "live"):
        live = decoder.live
        running = int(live() if callable(live) else live)
        waiting = 0
        queue = getattr(sched, "waiting", None)
        if queue is not None and hasattr(queue, "qsize"):
            waiting += int(queue.qsize())
        if getattr(sched, "held", None) is not None:
            waiting += 1
        return running, waiting
    health = getattr(app, "health", None)
    running = len(getattr(health, "live", ()) or ())
    turns = getattr(app, "turns", None)
    if turns is None:
        return running, 0
    parked = getattr(turns, "parked", None)
    return running, int(parked if parked is not None else getattr(turns, "waiting", 0) or 0)


def _endings(app: Any) -> tuple[int | None, int | None]:
    """(client disconnections, preemptions) at scrape time, or None where nothing counts them.

    The Mac scheduler counts cancelled requests and preempted background jobs per request. The CUDA
    scheduler raises RequestCancelled without counting disconnects and counts lane yields (requeued
    background streams) in ``yields``; neither server counts per-request failures.
    """

    scheduler = getattr(app, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "active") and hasattr(scheduler, "waiting"):
        return (_count(scheduler, "cancelled"), _count(scheduler, "preemptions"))
    engine = getattr(app, "engine", None)
    sched = getattr(engine, "scheduler", None) if engine is not None else None
    if sched is not None and hasattr(sched, "yields"):
        return (None, _count(sched, "yields"))
    return (None, None)


def _count(owner: Any, name: str) -> int | None:
    """A scheduler's own count, or None where this server keeps no such count."""

    value = getattr(owner, name, None)
    return None if value is None else max(0, int(value))


def _pools(app: Any) -> list[tuple[str, float]]:
    """One ratio per live stream. An idle server still publishes pool 0 at 0."""

    window = _window(app)
    lengths = _lengths(app)
    if not lengths:
        return [("0", 0.0)]
    if window <= 0:
        return [(str(i), 0.0) for i in range(len(lengths))]
    return [(str(i), min(1.0, n / window)) for i, n in enumerate(lengths)]


def _window(app: Any) -> int:
    engine = getattr(app, "engine", None)
    decoder = getattr(getattr(engine, "scheduler", None), "decoder", None) if engine is not None else None
    for owner, name in ((engine, "context_window"), (app, "context_window"), (decoder, "context")):
        n = _positive(getattr(owner, name, None) if owner is not None else None)
        if n:
            return n
    return 0


def _lengths(app: Any) -> list[int]:
    engine = getattr(app, "engine", None)
    decoder = getattr(getattr(engine, "scheduler", None), "decoder", None) if engine is not None else None
    if decoder is not None:
        streams = list(getattr(decoder, "streams", {}).values())
        streams += list(getattr(decoder, "filling", ()) or ())
        return [_occupied(stream) for stream in streams]
    live = getattr(engine, "_live", None) if engine is not None else None
    if not live:
        return []
    return [_occupied(stream) for stream, _cache in live if not getattr(stream, "finished", False)]


def _occupied(stream: Any) -> int:
    """Mac streams publish ``cache_len``. A CUDA stream's ``context`` gains the prompt when decode starts."""

    cache = getattr(stream, "cache_len", None)
    if isinstance(cache, int):
        return cache
    context = getattr(stream, "context", None) or ()
    if len(context):
        return len(context)
    prompt = getattr(stream, "prompt", None) or ()
    return len(prompt)


def _positive(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _edge(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


__all__ = ["BUCKETS", "PREFIX", "Histogram", "Metrics", "begin", "bind", "finish_request", "note", "of", "render",
           "send", "tokens"]
