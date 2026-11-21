"""Prompt/TTFT rate, ITL, TPOT and round gaps from streamed chunk arrivals; docs/cache-benchmarks.md defines each."""
import math
import statistics

from bench_evidence import count

# Old receipt keys still written beside their replacements, so readers of older receipts keep working.
DEPRECATED = {"prefill_tps": "prompt_ttft_tps", "aggregate_prefill_tps": "aggregate_prompt_ttft_tps"}
NOTE = {old: f"deprecated: same value as {new}; prompt tokens over client TTFT, not the engine's prefill time"
        for old, new in DEPRECATED.items()}


def rank(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an ascending list (None for an empty one)."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(len(sorted_values) * q) - 1)]


def arrivals(run: dict) -> list[float] | None:
    """A reply's output chunk arrivals (every output kind); older run dicts without them give their text pieces."""
    if "output_times" in run:
        return run["output_times"]
    return [p[0] for p in run.get("pieces") or []]


def samples(run: dict) -> tuple[list[float], bool] | None:
    """A reply's chunk gaps in ms and whether each chunk is known to hold one token; None when unmatched."""
    times, n = arrivals(run), count(run.get("tokens"))
    if not run.get("complete") or times is None or n is None or n < 2 or len(times) < 2 or len(times) > n:
        return None
    if any(b < a for a, b in zip(times, times[1:])):
        return None
    return [(b - a) * 1000 for a, b in zip(times, times[1:])], len(times) == n


def summary(itl: list[float], gaps: list[float], tokens: int, chunks: int, spans: int) -> dict:
    """Token ITL/TPOT (ms; known only where every chunk held one token), chunk-level round gaps and estimates."""
    itl, gaps = sorted(itl), sorted(gaps)
    return {"itl_mean_ms": statistics.fmean(itl) if itl else None, "itl_p50_ms": rank(itl, 0.5),
            "itl_p95_ms": rank(itl, 0.95), "tpot_ms": statistics.fmean(itl) if itl else None,
            "chunk_tpot_estimate_ms": sum(gaps) / spans if gaps and spans else None,
            "round_gap_p50_ms": rank(gaps, 0.5), "round_gap_p95_ms": rank(gaps, 0.95),
            "round_gap_mean_ms": statistics.fmean(gaps) if gaps else None,
            "tokens_per_chunk": tokens / chunks if chunks else None}


def prompt_ttft_tps(run: dict) -> float | None:
    """prompt_tokens / client TTFT of a cold request (not engine prefill time); None for a reused or unverified one."""
    prompt, ttft = count(run.get("prompt_tokens")), run.get("ttft_s")
    if run.get("cache_state") != "cold" or not prompt or type(ttft) not in (int, float) or ttft <= 0:
        return None
    return prompt / ttft


def request(run: dict) -> dict:
    """One request's figures (None where its stream does not support them)."""
    rate = prompt_ttft_tps(run)
    out = {"prompt_ttft_tps": rate, "prefill_tps": rate, "token_timing": None, **summary([], [], 0, 0, 0)}
    got = samples(run)
    if got is not None:
        gaps, per_token = got
        out.update(token_timing="token" if per_token else "chunk",
                   **summary(gaps if per_token else [], gaps, run["tokens"], len(gaps) + 1, run["tokens"] - 1))
    return out


def pooled(runs: list[dict]) -> dict:
    """Every measurable stream's samples together (token ITL from one-token-a-chunk streams), and the aggregate rate."""
    itl, gaps, tokens, chunks, spans, timed, token_timed = [], [], 0, 0, 0, 0, 0
    for r in runs:
        got = samples(r)
        if got is None:
            continue
        itl += got[0] if got[1] else []
        gaps += got[0]
        tokens += r["tokens"]
        chunks += len(got[0]) + 1
        spans += r["tokens"] - 1
        timed += 1
        token_timed += got[1]
    out = {"streams_timed": timed, "streams_token_timed": token_timed, **summary(itl, gaps, tokens, chunks, spans),
           "aggregate_prompt_ttft_tps": None}
    cold = [r for r in runs if prompt_ttft_tps(r) is not None and type(r.get("first")) in (int, float)
            and type(r.get("sent")) in (int, float)]
    if cold and len(cold) == len(runs):
        span = max(r["first"] for r in cold) - min(r["sent"] for r in cold)
        out["aggregate_prompt_ttft_tps"] = sum(r["prompt_tokens"] for r in cold) / span if span > 0 else None
    out["aggregate_prefill_tps"] = out["aggregate_prompt_ttft_tps"]
    return out
