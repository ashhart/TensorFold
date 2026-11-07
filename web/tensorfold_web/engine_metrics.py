"""Engine metrics."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

COUNTERS = ("completion_tokens_total", "decode_seconds_total", "prompt_tokens_total", "prefill_seconds_total",
            "drafted_total", "accepted_total", "prompt_tokens_cached_total")


KV_FILL = ("kv_cache_usage_ratio", "kv_cache_usage_perc", "gpu_cache_usage_perc")


METRICS = {"generation_tokens_total": "completion_tokens_total", "request_decode_seconds_sum": "decode_seconds_total",
           "prompt_tokens_total": "prompt_tokens_total", "request_prefill_seconds_sum": "prefill_seconds_total",
           "mtp_drafted_total": "drafted_total", "mtp_accepted_total": "accepted_total",
           "prompt_tokens_cached_total": "prompt_tokens_cached_total"}


@dataclass
class Health:
    ok: bool
    status: str                     # "stopped", "starting", "running", "unauthorized", "failed", "unreachable", "error"
    model: str | None = None
    model_ids: list[str] | None = None
    warming: bool = False
    connections: int = 0
    waiting: int = 0
    decode_tps: float | None = None     # None: this engine gives no live figure (the page shows a dash, not 0)
    prefill_tps: float | None = None
    memory: dict | None = None
    context_length: int | None = None   # the CUDA server reports it in /health
    counters: dict[str, float] | None = None   # cumulative totals (CUDA /health, or /metrics), for rates between polls
    draft_acceptance: float | None = None      # drafted tokens kept since the engine started, when it counts them
    kv_fill: float | None = None               # share of the KV cache held by requests and kept prompts, 0 to 1
    streams: dict[str, int] | None = None      # requests decoding, prefilling and paused, when the engine says it
    kept_prompts: int | None = None            # prompt states kept for reuse (the CUDA server)
    preemptions: float | None = None           # requests set aside to free memory, since the engine started
    prompt_cache: dict | None = None           # prompt tokens restored from kept states, of all prompt tokens (1.0.5)
    totals: dict | None = None                 # tokens and requests since the engine started, for the usage history
    source: str = "health"                     # "health", or "metrics" when /health is bare behind an API key
    load: dict | None = None                   # TensorFold's GET /v1/load: model, parked, lanes and the queue
    parked: bool = False                       # the idle unload has closed the engine; the next request reloads it
    raw: dict | None = None
    error: str | None = None
    generation: int | None = None              # the engine target it was read under (Engine.target)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "generation"}


def _num(body: dict, *keys: str) -> float | None:
    for k in keys:
        v = body.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def parse_health(body: dict) -> Health:
    """The MLX server's /health (status, model, live, memory) and the CUDA server's counters, read the same way."""

    status = body.get("status", "ok")
    if status != "ok" and body.get("ok") is not True:
        return Health(ok=False, status="error", raw=body, error=f"engine reports status {status!r}")
    live = body.get("live") if isinstance(body.get("live"), dict) else None
    ctx = _num(body, "context_length", "max_model_len")
    counters = {k: float(body[k]) for k in COUNTERS if isinstance(body.get(k), (int, float))} or None
    return Health(ok=True, status="starting" if body.get("warming") else "running", model=body.get("model"),
                  model_ids=body.get("model_ids"), warming=bool(body.get("warming")),
                  connections=int(_num(live or {}, "connections") or _num(body, "requests_running") or 0),
                  waiting=int(_num(live or {}, "waiting") or _num(body, "requests_waiting") or 0),
                  decode_tps=_num(live, "decode_tokens_per_second") if live else None,
                  prefill_tps=_num(live, "prefill_tokens_per_second") if live else None,
                  memory=body.get("memory") or None, context_length=int(ctx) if ctx else None,
                  counters=counters, kv_fill=kv_fill(body), streams=streams_of(body),
                  kept_prompts=int(body["kept_prompts"]) if isinstance(body.get("kept_prompts"), int) else None,
                  raw=body)


def parse_load(body: object) -> dict | None:

    if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not isinstance(body.get("parked"), bool):
        return None
    out: dict = {"model": body["model"], "parked": body["parked"]}
    for k in ("lanes", "running", "waiting"):
        v = body.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            out[k] = v
    return out


def parse_metrics(text: str) -> dict[str, float]:
    """TensorFold's Prometheus page as numbers: the unlabelled `tensorfold:` series, the prefix dropped."""

    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("tensorfold:"):
            continue
        name, _, value = line[len("tensorfold:"):].partition(" ")
        if "{" in name:
            # a labelled series (per key, per stream, histogram buckets): only the KV fill is wanted, its largest pool
            base = name.split("{", 1)[0]
            if base in KV_FILL:
                with contextlib.suppress(ValueError):
                    out[base] = max(out.get(base, 0.0), float(value.strip()))
            continue
        with contextlib.suppress(ValueError):
            out[name] = float(value.strip())
    return out


def kv_fill(body: dict, m: dict[str, float] | None = None) -> float | None:
    """The share of the KV cache held, 0 to 1: the CUDA server's pool in /health, else the /metrics gauge."""

    pool, free = _num(body, "pool_tokens"), _num(body, "pool_free_tokens")
    if pool and free is not None:
        return min(1.0, max(0.0, 1 - free / pool))
    for k in KV_FILL:
        if m and k in m:
            v = m[k]
            return min(1.0, max(0.0, v / 100 if v > 1 else v))
    return None


def streams_of(body: dict) -> dict[str, int] | None:
    """Requests by phase, when the engine says it: writing, reading their prompt, set aside."""

    st = body.get("streams") if isinstance(body.get("streams"), dict) else None
    if not st:
        return None
    pick = {"decoding": ("decoding",), "prefilling": ("prefilling", "filling"), "paused": ("paused",)}
    out = {k: int(next((st[n] for n in names if isinstance(st.get(n), (int, float))), 0)) for k, names in pick.items()}
    return out


def rates_from_counters(previous: dict[str, float] | None, current: dict[str, float]) -> tuple[float | None, float | None]:

    if previous is None:
        return None, None

    def rate(tokens: str, seconds: str) -> float | None:
        if tokens not in current or tokens not in previous:
            return None
        dtok = current[tokens] - previous[tokens]
        dsec = current.get(seconds, 0.0) - previous.get(seconds, 0.0)
        if dtok <= 0:
            return 0.0
        return dtok / dsec if dsec > 0 else None

    def read(tokens: str, seconds: str, restored: str) -> float | None:
        # tokens restored from a kept prompt state take no prefill time: counting them would inflate the rate
        r = rate(tokens, seconds)
        if not r or restored not in current or restored not in previous:
            return r
        dtok = current[tokens] - previous[tokens] - (current[restored] - previous[restored])
        dsec = current.get(seconds, 0.0) - previous.get(seconds, 0.0)
        return max(0.0, dtok) / dsec if dsec > 0 else None

    return rate("completion_tokens_total", "decode_seconds_total"), \
        read("prompt_tokens_total", "prefill_seconds_total", "prompt_tokens_cached_total")


def totals_of(m: dict[str, float]) -> dict | None:
    """Prompt and written tokens, prompt tokens restored, and requests answered since the engine started."""

    if "generation_tokens_total" not in m and "prompt_tokens_total" not in m:
        return None
    return {"prompt": int(m.get("prompt_tokens_total", 0)), "completion": int(m.get("generation_tokens_total", 0)),
            "cached": int(m.get("prompt_tokens_cached_total", 0)), "requests": int(m.get("request_latency_seconds_count", 0))}


def prompt_cache_of(m: dict[str, float]) -> dict | None:
    """Prompt tokens restored from kept prompt states since the engine started, of all its prompt tokens."""

    if "prompt_tokens_cached_total" not in m:
        return None
    return {"restored": int(m["prompt_tokens_cached_total"]), "prompt": int(m.get("prompt_tokens_total", 0))}
