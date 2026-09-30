"""Hand-written Prometheus text exposition (0.0.4) shared by both servers: no client library, no new dependency.

Metric names mirror vLLM's ``vllm:*`` names with the ``tensorfold:`` prefix, so a vLLM dashboard scrapes this
endpoint with a prefix swap. The histogram bucket edges are vLLM's defaults, copied so the dashboards'
percentiles keep their meaning.
"""

from __future__ import annotations

import threading

# vLLM's defaults (vllm/v1/metrics/buckets.py)
TIME_TO_FIRST_TOKEN_BUCKETS = (0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
                                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
                                float("inf"))
E2E_REQUEST_LATENCY_BUCKETS = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0,
                                6.0, 8.0, 10.0, 20.0, 40.0, 60.0, float("inf"))


def _bucket(upper: float) -> str:
    return "+Inf" if upper == float("inf") else repr(upper)


def escape_label(value: str) -> str:
    """0.0.4 label-value escaping: backslash, double quote and line break."""

    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return "{" + ", ".join(f'{key}="{escape_label(value)}"' for key, value in labels.items()) + "}"


def counter_lines(name: str, help: str, value: float) -> list[str]:
    """A counter's three lines from a value read at scrape time (the counters a server's health counters hold)."""

    return [f"# HELP {name} {help}", f"# TYPE {name} counter", f"{name} {_value(value)}"]


def _value(number: float) -> str:
    if isinstance(number, int) or float(number).is_integer():
        return str(int(number))
    return repr(float(number))


class Counter:
    """Monotone since the server started; the value, not deltas, is what a scraper reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0.0

    def add(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def render(self, name: str, help: str, labels: dict[str, str] | None = None) -> list[str]:
        return [f"# HELP {name} {help}", f"# TYPE {name} counter",
                f"{name}{_labels(labels)} {_value(self.value)}"]


class Gauge:
    """The value set at the last update; the exposition reads it as-is."""

    def __init__(self, value: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._value = float(value)

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def render(self, name: str, help: str, labels: dict[str, str] | None = None) -> list[str]:
        return [f"# HELP {name} {help}", f"# TYPE {name} gauge",
                f"{name}{_labels(labels)} {_value(self.value)}"]


class Histogram:
    """Fixed buckets (vLLM's defaults), cumulative counts, sum and sample count; thread-safe."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._buckets = buckets
        self._lock = threading.Lock()
        self._counts = [0] * len(buckets)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for i, upper in enumerate(self._buckets):
                if value <= upper:
                    for j in range(i, len(self._buckets)):
                        self._counts[j] += 1
                    return
            self._counts[-1] += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum

    def render(self, name: str, help: str, labels: dict[str, str] | None = None) -> list[str]:
        lines = [f"# HELP {name} {help}", f"# TYPE {name} histogram"]
        with self._lock:
            for upper, hits in zip(self._buckets, self._counts):
                lines.append(f"{name}_bucket{_labels({**(labels or {}), 'le': _bucket(upper)})} {hits}")
            lines.append(f"{name}_sum{_labels(labels)} {repr(self._sum)}")
            lines.append(f"{name}_count{_labels(labels)} {self._count}")
        return lines


class Metrics:
    """The Mac server's request metrics: the scheduler's queue, the token counts and the two vLLM histograms."""

    def __init__(self) -> None:
        self.prompt_tokens = Counter()
        self.generation_tokens = Counter()
        self.ttft = Histogram(TIME_TO_FIRST_TOKEN_BUCKETS)
        self.e2e = Histogram(E2E_REQUEST_LATENCY_BUCKETS)

    def observe_ttft(self, seconds: float) -> None:
        if seconds >= 0.0:
            self.ttft.observe(seconds)

    def observe_e2e(self, seconds: float) -> None:
        if seconds >= 0.0:
            self.e2e.observe(seconds)

    def render(self, running: int, waiting: int, drafted: int | None, accepted: int | None) -> list[str]:
        """The endpoint's body lines: gauges of the queue, the counters, then the histograms.

        ``drafted``/``accepted`` None: the engine does not count them, so the metrics stay out of the body
        rather than reading as zero (a drafter engine counts them per stream and reports the totals).
        """

        lines: list[str] = []
        lines += Gauge(running).render("tensorfold:num_requests_running", "Number of running requests")
        lines += Gauge(waiting).render("tensorfold:num_requests_waiting", "Number of requests waiting to be processed")
        lines += self.prompt_tokens.render("tensorfold:prompt_tokens_total", "Number of prefill tokens processed")
        lines += self.generation_tokens.render("tensorfold:generation_tokens_total",
                                                "Number of generation tokens processed")
        if drafted is not None:
            lines += ["# HELP tensorfold:spec_decode_num_draft_tokens_total SpecDecoding: Number of draft tokens",
                      "# TYPE tensorfold:spec_decode_num_draft_tokens_total counter",
                      f"tensorfold:spec_decode_num_draft_tokens_total {drafted}"]
        if accepted is not None:
            lines += ["# HELP tensorfold:spec_decode_num_accepted_tokens_total SpecDecoding: Number of accepted tokens",
                      "# TYPE tensorfold:spec_decode_num_accepted_tokens_total counter",
                      f"tensorfold:spec_decode_num_accepted_tokens_total {accepted}"]
        lines += self.ttft.render("tensorfold:time_to_first_token_seconds", "Latency until first output")
        lines += self.e2e.render("tensorfold:e2e_request_latency_seconds", "E2E request latency")
        return lines


__all__ = ["Counter", "Gauge", "Histogram", "Metrics", "E2E_REQUEST_LATENCY_BUCKETS",
           "TIME_TO_FIRST_TOKEN_BUCKETS", "counter_lines", "escape_label"]
