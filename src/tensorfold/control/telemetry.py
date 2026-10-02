"""Read-only telemetry. Missing values stay unknown; resets/gaps never become throughput spikes."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import re
import time
from urllib import error, parse, request

from .safety import ControlError, clean, redact

LIMIT = 1 << 20


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def base_url(value: str) -> str:
    if not isinstance(value, str) or any(ord(c) < 33 for c in value):
        raise ControlError("endpoint must be an HTTP(S) URL without whitespace")
    try:
        url = parse.urlsplit(value)
        port = url.port
    except ValueError as exc:
        raise ControlError("invalid endpoint URL") from exc
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ControlError("endpoint must be HTTP(S), without embedded credentials")
    if url.query or url.fragment or (port is not None and not 1 <= port <= 65535):
        raise ControlError("endpoint must not contain a query, fragment, or invalid port")
    path = url.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return parse.urlunsplit((url.scheme, url.netloc, path, "", ""))


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes


class Client:
    def __init__(self, endpoint: str, token: str | None = None, timeout: float = 2):
        self.endpoint = base_url(endpoint)
        if not 0 < timeout <= 30 or not math.isfinite(timeout):
            raise ControlError("HTTP timeout must be > 0 and <= 30 seconds")
        if token is not None and any(ord(c) < 32 for c in token):
            raise ControlError("invalid API token")
        self.token, self.timeout = token, timeout
        # Avoid leaking credentials to proxy environment variables or redirected origins.
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def _read(self, stream, deadline: float) -> bytes:
        parts, size = [], 0
        while True:
            if time.monotonic() >= deadline:
                raise ControlError("telemetry body deadline exceeded")
            # read1 returns after available bytes, so a trickle cannot extend the body forever.
            chunk = stream.read1(min(65536, LIMIT + 1 - size))
            if not chunk:
                break
            parts.append(chunk)
            size += len(chunk)
            if size > LIMIT:
                raise ControlError("telemetry response exceeds 1 MiB")
        return b"".join(parts)

    def get(self, path: str) -> Response:
        headers = {"Accept": "application/json, text/plain", "User-Agent": "TensorFold-Control/0.1",
                   "Connection": "close"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        req = request.Request(self.endpoint + path, headers=headers, method="GET")
        deadline = time.monotonic() + self.timeout
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return Response(response.status, self._read(response, deadline))
        except error.HTTPError as exc:
            with exc:
                return Response(exc.code, self._read(exc, deadline))
        except (error.URLError, TimeoutError, OSError) as exc:
            raise ControlError(f"endpoint unreachable ({type(exc).__name__})") from exc

    def sample(self) -> "Sample":
        now = time.monotonic()
        try:
            health = self.get("/health")
            if health.status in {401, 403}:
                return Sample(now, phase="unauthorized", error=f"HTTP {health.status}; check --token-env")
            data = json.loads(health.body)
            if not isinstance(data, dict):
                raise ValueError("health must be an object")
            ready = data.get("ok") is True or data.get("status") in {"ok", "healthy", "ready"}
            warming = data.get("warming") is True
            if health.status != 200 or not (ready or warming):
                return Sample(now, phase="warming" if warming else "unhealthy", error=f"health HTTP {health.status}")
            warning = ""
            try:
                metrics_response = self.get("/metrics")
                metrics = parse_metrics(metrics_response.body.decode("utf-8", errors="replace")) \
                    if metrics_response.status == 200 else {}
                if metrics_response.status != 200:
                    warning = f"metrics unavailable (HTTP {metrics_response.status})"
            except ControlError as exc:
                metrics, warning = {}, str(exc)
            sample = normalize(now, data, metrics)
            sample.warning = warning
            return sample
        except (ControlError, ValueError, UnicodeError, TypeError) as exc:
            return Sample(now, error=redact(str(exc), 240))


def numeric(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


# Accept labels without interpreting them. We need family aggregates, never arbitrary dynamic labels.
_METRIC = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?:[^"{}]|"(?:[^"\\]|\\.)*")*\})?'
                     r'\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+\d+)?\s*$')


def parse_metrics(text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for line in text[:LIMIT].splitlines():
        if line.startswith("#") or len(line) > 16384:
            continue
        match = _METRIC.fullmatch(line)
        if match:
            value = numeric(float(match[2]))
            if value is not None:
                metrics.setdefault(match[1], []).append(value)
    return metrics


def metric(values: dict[str, list[float]], *names: str, maximum: bool = False) -> float | None:
    # The first recognized family wins. Alias families never get added together.
    for name in names:
        found = values.get(name)
        if found:
            return max(found) if maximum else sum(found)
    return None


@dataclass
class Sample:
    when: float
    online: bool = False
    phase: str = "offline"
    model: str = ""
    counters: dict[str, float] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    running: float | None = None
    waiting: float | None = None
    memory: float | None = None
    cache: float | None = None
    peak: float | None = None
    context: float | None = None
    kv_ratio: float | None = None
    acceptance: float | None = None
    live: dict[str, float] = field(default_factory=dict)     # the server's own live line, when /health has it
    ttft_mean: float | None = None
    error: str = ""
    warning: str = ""


def normalize(now: float, health: dict, values: dict[str, list[float]]) -> Sample:
    sample = Sample(now, True, "warming" if health.get("warming") else "ready",
                    clean(health.get("model", ""), 160))
    live = health.get("live")
    if isinstance(live, dict):
        for key in ("connections", "waiting", "decode_tokens_per_second", "prefill_tokens_per_second"):
            value = numeric(live.get(key))
            if value is not None and value >= 0:
                sample.live[key] = value
    sample.running = metric(values, "tensorfold:requests_running", "tensorfold:num_requests_running",
                            "vllm:num_requests_running")
    if sample.running is None:
        sample.running = numeric(health.get("requests_running"))
    sample.waiting = metric(values, "tensorfold:requests_waiting", "tensorfold:num_requests_waiting",
                            "vllm:num_requests_waiting")
    families = {
        "generation": ("tensorfold:generation_tokens_total", "vllm:generation_tokens_total"),
        "prompt": ("tensorfold:prompt_tokens_total", "vllm:prompt_tokens_total"),
        "drafted": ("tensorfold:mtp_drafted_total", "tensorfold:spec_decode_num_draft_tokens_total"),
        "accepted": ("tensorfold:mtp_accepted_total", "tensorfold:spec_decode_num_accepted_tokens_total"),
    }
    for key, names in families.items():
        value = metric(values, *names)
        if value is not None:
            sample.counters[key] = value
            sample.sources[key] = "completed requests"
    # CUDA /health reports tokens while they are emitted, unlike its /metrics in 0.6.0.
    live_generation = numeric(health.get("completion_tokens_total"))
    if live_generation is not None:
        sample.counters["generation"] = live_generation
        sample.sources["generation"] = "live counter"
    if "prompt" not in sample.counters and numeric(health.get("prompt_tokens_total")) is not None:
        sample.counters["prompt"] = float(health["prompt_tokens_total"])
        sample.sources["prompt"] = "completed requests"
    memory = health.get("memory")
    if isinstance(memory, dict):
        sample.memory = numeric(memory.get("active"))
        sample.cache = numeric(memory.get("cache"))
        sample.peak = numeric(memory.get("peak"))
    sample.context = numeric(health.get("context_length"))
    sample.kv_ratio = metric(values, "tensorfold:kv_cache_usage_ratio", "tensorfold:kv_cache_usage_perc",
                             "vllm:kv_cache_usage_perc", maximum=True)
    if sample.kv_ratio is not None and not 0 <= sample.kv_ratio <= 1:
        sample.kv_ratio = None
    drafted, accepted = sample.counters.get("drafted"), sample.counters.get("accepted")
    if drafted and accepted is not None and accepted <= drafted:
        sample.acceptance = accepted / drafted
    total = metric(values, "tensorfold:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_sum")
    count = metric(values, "tensorfold:time_to_first_token_seconds_count", "vllm:time_to_first_token_seconds_count")
    if total is not None and count:
        sample.ttft_mean = total / count
    return sample


class Rates:
    """Rolling aggregate counters/sec, not a per-stream benchmark. A failure invalidates every baseline."""
    def __init__(self, window: float = 10, max_gap: float = 8):
        self.window, self.max_gap = window, max_gap
        self.history: dict[str, deque[tuple[float, float, str]]] = {}

    def update(self, sample: Sample) -> dict[str, float | None]:
        result: dict[str, float | None] = {key: None for key in ("generation", "prompt")}
        if not sample.online:
            self.history.clear()
            return result
        for key in list(self.history):
            if key not in sample.counters:
                self.history.pop(key)
        for key in result:
            value = sample.counters.get(key)
            if value is None:
                continue
            source = sample.sources.get(key, "unknown")
            history = self.history.setdefault(key, deque(maxlen=128))
            if history:
                when, before, previous_source = history[-1]
                if (sample.when <= when or sample.when - when > self.max_gap or value < before
                        or source != previous_source):
                    history.clear()
            history.append((sample.when, value, source))
            while len(history) > 2 and sample.when - history[0][0] > self.window:
                history.popleft()
            elapsed = sample.when - history[0][0]
            if elapsed > 0:
                result[key] = max(0, (value - history[0][1]) / elapsed)
        return result
