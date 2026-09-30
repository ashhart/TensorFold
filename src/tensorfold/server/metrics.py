"""Dependency-free Prometheus metrics shared by the MLX and CUDA HTTP servers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Iterable, Mapping

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "tensorfold:"
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
_MADE = threading.Lock()


@dataclass
class Histogram:
    """Cumulative Prometheus histogram state."""

    bounds: tuple[float, ...]
    buckets: list[int] = field(init=False)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.buckets = [0] * len(self.bounds)

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.total += value
        self.count += 1
        for index, bound in enumerate(self.bounds):
            if value <= bound:
                self.buckets[index] += 1

    def copy(self) -> "Histogram":
        copied = Histogram(self.bounds)
        copied.buckets, copied.total, copied.count = list(self.buckets), self.total, self.count
        return copied


class Metrics:
    """Process-lifetime counters and request histograms for one served app."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counters = {
            "prompt_tokens_total": 0,
            "generation_tokens_total": 0,
            "mtp_drafted_total": 0,
            "mtp_accepted_total": 0,
        }
        self.histograms = {
            "request_latency_seconds": Histogram(LATENCY_BUCKETS),
            "time_to_first_token_seconds": Histogram(TTFT_BUCKETS),
        }

    def observe(
        self,
        *,
        prompt_tokens: int,
        generation_tokens: int,
        latency: float,
        ttft: float | None = None,
        mtp_drafted: int = 0,
        mtp_accepted: int = 0,
    ) -> None:
        with self.lock:
            self.counters["prompt_tokens_total"] += max(0, int(prompt_tokens))
            self.counters["generation_tokens_total"] += max(0, int(generation_tokens))
            self.counters["mtp_drafted_total"] += max(0, int(mtp_drafted))
            self.counters["mtp_accepted_total"] += max(0, int(mtp_accepted))
            self.histograms["request_latency_seconds"].observe(latency)
            if ttft is not None:
                self.histograms["time_to_first_token_seconds"].observe(ttft)

    def snapshot(self) -> tuple[dict[str, int], dict[str, Histogram]]:
        with self.lock:
            return dict(self.counters), {name: histogram.copy() for name, histogram in self.histograms.items()}

    def render(
        self,
        *,
        running: int,
        waiting: int,
        pools: Mapping[str, float] | None = None,
        counters: Mapping[str, int | float] | None = None,
    ) -> bytes:
        stored, histograms = self.snapshot()
        if counters is not None:
            for name in stored:
                if name in counters:
                    stored[name] = max(0, int(counters[name]))
        lines: list[str] = []
        _simple(lines, "requests_running", "Requests currently prefilling or decoding.", "gauge", max(0, int(running)))
        _simple(lines, "requests_waiting", "Requests accepted but not yet admitted.", "gauge", max(0, int(waiting)))
        for name, help_text in (
            ("prompt_tokens_total", "Prompt tokens processed."),
            ("generation_tokens_total", "Generation tokens emitted."),
            ("mtp_drafted_total", "MTP draft tokens verified."),
            ("mtp_accepted_total", "MTP draft tokens accepted."),
        ):
            _simple(lines, name, help_text, "counter", stored[name])
        name = PREFIX + "kv_cache_usage_ratio"
        lines.extend((f"# HELP {name} Logical KV cache positions in use divided by pool capacity.",
                      f"# TYPE {name} gauge"))
        for pool, ratio in sorted((pools or {}).items()):
            value = min(1.0, max(0.0, float(ratio)))
            lines.append(f'{name}{{pool="{_escape_label(pool)}"}} {_number(value)}')
        for metric, help_text in (
            ("request_latency_seconds", "End-to-end request latency in seconds."),
            ("time_to_first_token_seconds", "Time from request receipt to first generated token in seconds."),
        ):
            _histogram(lines, metric, help_text, histograms[metric])
        return ("\n".join(lines) + "\n").encode()


def of(app: Any) -> Metrics:
    """The metrics collector owned by ``app``, made on first use."""

    with _MADE:
        found = app.__dict__.get("metrics")
        if found is None:
            found = app.__dict__["metrics"] = Metrics()
        return found


def mlx_state(app: Any) -> tuple[int, int, dict[str, float]]:
    """Running/waiting requests and logical KV occupancy for the MLX lane pool."""

    scheduler = getattr(app, "scheduler", None)
    if scheduler is None:
        return 0, 0, {}
    snapshot = getattr(scheduler, "metrics_snapshot", None)
    state = snapshot() if callable(snapshot) else {"running": 0, "waiting": 0, "tokens": 0}
    lanes = max(1, int(getattr(scheduler, "lanes", 1) or 1))
    window = int(getattr(app, "context_window", 0) or 0)
    preparing_lock = getattr(app, "_preparing_lock", None)
    if preparing_lock is not None:
        with preparing_lock:
            preparing = max(0, int(getattr(app, "_preparing", 0) or 0))
    else:
        preparing = max(0, int(getattr(app, "_preparing", 0) or 0))
    pools = {}
    if window > 0:
        pools["mlx"] = min(1.0, max(0.0, int(state["tokens"]) / (lanes * window)))
    return int(state["running"]), int(state["waiting"]) + preparing, pools


def cuda_state(app: Any, live: Iterable[Any]) -> tuple[int, int, dict[str, float]]:
    """Running/waiting requests and logical KV occupancy for a CUDA stream pool."""

    requests = list(live)
    scheduler = getattr(getattr(app, "engine", None), "scheduler", None)
    decoder = getattr(scheduler, "decoder", None)
    if scheduler is not None:
        waiting = int(getattr(getattr(scheduler, "waiting", None), "qsize", lambda: 0)())
        running = max(0, len(requests) - waiting)
    else:
        waiting = sum(not bool(getattr(request, "admitted", True)) for request in requests)
        running = len(requests) - waiting
    pools: dict[str, float] = {}
    capacity = int(getattr(decoder, "capacity", 0) or 0)
    decoding = list(getattr(decoder, "streams", {}).values()) if decoder is not None else []
    filling = list(getattr(decoder, "filling", ())) if decoder is not None else []
    streams = list({id(stream): stream for stream in (*decoding, *filling)}.values())
    slots = max(1, int(getattr(scheduler, "max_streams", 1) or 1)) if scheduler else 1
    if capacity > 0:
        used = sum(max(len(getattr(stream, "context", ())), int(getattr(getattr(stream, "st", None), "pos", 0) or 0))
                   for stream in streams)
        pools["cuda"] = min(1.0, max(0.0, used / (slots * capacity)))
    return running, waiting, pools


def _simple(lines: list[str], metric: str, help_text: str, kind: str, value: int | float) -> None:
    name = PREFIX + metric
    lines.extend((f"# HELP {name} {_escape_help(help_text)}", f"# TYPE {name} {kind}", f"{name} {_number(value)}"))


def _histogram(lines: list[str], metric: str, help_text: str, histogram: Histogram) -> None:
    name = PREFIX + metric
    lines.extend((f"# HELP {name} {_escape_help(help_text)}", f"# TYPE {name} histogram"))
    for bound, count in zip(histogram.bounds, histogram.buckets):
        lines.append(f'{name}_bucket{{le="{_number(bound)}"}} {count}')
    lines.append(f'{name}_bucket{{le="+Inf"}} {histogram.count}')
    lines.append(f"{name}_sum {_number(histogram.total)}")
    lines.append(f"{name}_count {histogram.count}")


def _number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return format(value, ".12g")


def _escape_help(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(value: str) -> str:
    return _escape_help(value).replace('"', '\\"')


__all__ = ["CONTENT_TYPE", "Histogram", "Metrics", "cuda_state", "mlx_state", "of"]
