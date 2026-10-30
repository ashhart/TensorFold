"""TTFT, prefill tok/s, ITL and round gaps from streamed chunk arrivals; docs/cache-benchmarks.md defines each."""
import math
import statistics

from bench_evidence import count


def rank(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an ascending list (None for an empty one)."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(len(sorted_values) * q) - 1)]


def samples(run: dict) -> tuple[list[float], list[float]] | None:
    """A request's ITL samples and round gaps in ms; None when its chunks cannot be matched to its tokens."""
    pieces, n = run.get("pieces") or [], count(run.get("tokens"))
    if not run.get("complete") or n is None or n < 2 or len(pieces) < 2 or len(pieces) > n:
        return None
    times = [p[0] for p in pieces]
    if any(b < a for a, b in zip(times, times[1:])):
        return None
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return gaps + [0.0] * (n - len(pieces)), gaps


def summary(itl: list[float], gaps: list[float], tokens: int, chunks: int) -> dict:
    """ITL mean, median and p95, round gap mean, median and p95 (ms), and tokens a chunk."""
    itl, gaps = sorted(itl), sorted(gaps)
    return {"itl_mean_ms": statistics.fmean(itl) if itl else None, "itl_p50_ms": rank(itl, 0.5),
            "itl_p95_ms": rank(itl, 0.95), "round_gap_p50_ms": rank(gaps, 0.5), "round_gap_p95_ms": rank(gaps, 0.95),
            "round_gap_mean_ms": statistics.fmean(gaps) if gaps else None,
            "tokens_per_chunk": tokens / chunks if chunks else None}


def prefill_tps(run: dict) -> float | None:
    """prompt_tokens / TTFT of a cold request; None for a reused or unverified one."""
    prompt, ttft = count(run.get("prompt_tokens")), run.get("ttft_s")
    if run.get("cache_state") != "cold" or not prompt or type(ttft) not in (int, float) or ttft <= 0:
        return None
    return prompt / ttft


def request(run: dict) -> dict:
    """One request's figures (None where its stream does not support them)."""
    out = {"prefill_tps": prefill_tps(run), **summary([], [], 0, 0)}
    got = samples(run)
    if got is not None:
        out.update(summary(got[0], got[1], run["tokens"], len(run["pieces"])))
    return out


def pooled(runs: list[dict]) -> dict:
    """Every measurable stream's ITL samples and round gaps together, and the aggregate prefill rate."""
    itl, gaps, tokens, chunks, timed = [], [], 0, 0, 0
    for r in runs:
        got = samples(r)
        if got is None:
            continue
        itl += got[0]
        gaps += got[1]
        tokens += r["tokens"]
        chunks += len(r["pieces"])
        timed += 1
    out = {"streams_timed": timed, **summary(itl, gaps, tokens, chunks), "aggregate_prefill_tps": None}
    cold = [r for r in runs if prefill_tps(r) is not None and type(r.get("first")) in (int, float)
            and type(r.get("sent")) in (int, float)]
    if cold and len(cold) == len(runs):
        span = max(r["first"] for r in cold) - min(r["sent"] for r in cold)
        out["aggregate_prefill_tps"] = sum(r["prompt_tokens"] for r in cold) / span if span > 0 else None
    return out
