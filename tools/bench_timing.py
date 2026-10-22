"""Full-workload throughput and observed SSE timing without inferring token times from text chunks."""
import math
import statistics

from bench_evidence import count


def timestamp(value):
    """Only finite, nonnegative numeric timestamps can define an interval."""
    if type(value) not in (int, float):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except OverflowError:
        return False


def interval(run):
    """A completed request's full client interval, including prefill and final usage delivery."""
    start, end = run.get("sent"), run.get("completed")
    if run.get("complete") is not True or run.get("error"):
        return None
    if not timestamp(start) or not timestamp(end) or end <= start:
        return None
    return start, end


def workload(runs):
    """All output tokens over the entire makespan; partial or unverified workloads have no rate."""
    valid = [r for r in runs if interval(r) is not None and (count(r.get("tokens")) or 0) > 0]
    out = {"requested_requests": len(runs), "timed_requests": len(valid), "makespan_s": None,
           "end_to_end_tps": None, "timing_valid": False}
    if not runs or len(valid) != len(runs):
        return out
    elapsed = max(r["completed"] for r in runs) - min(r["sent"] for r in runs)
    out.update(makespan_s=elapsed, end_to_end_tps=sum(r["tokens"] for r in runs) / elapsed, timing_valid=True)
    return out


def request(run):
    """Observe chunk gaps and request-relative arrival times; a chunk may contain several tokens."""
    out = {"elapsed_s": None, "chunk_offsets_s": None, "chunk_gaps_ms": None}
    bounds = interval(run)
    if bounds is None:
        return out
    start, end = bounds
    out["elapsed_s"] = end - start
    pieces = run.get("pieces") or []
    arrivals = []
    for piece in pieces:
        if not isinstance(piece, (tuple, list)) or len(piece) != 2:
            return out
        at = piece[0]
        if not timestamp(at) or not start <= at <= end or (arrivals and at < arrivals[-1]):
            return out
        arrivals.append(at)
    out["chunk_offsets_s"] = [at - start for at in arrivals]
    if len(arrivals) < 2:
        return out
    gaps = sorted((right - left) * 1000 for left, right in zip(arrivals, arrivals[1:]))
    out["chunk_gaps_ms"] = {"count": len(gaps), "p50": statistics.median(gaps),
                            "p95": gaps[math.ceil(len(gaps) * 0.95) - 1], "max": gaps[-1]}
    return out
