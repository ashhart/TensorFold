"""Allocate shared rows by landing probability, preserving parent order and maximizing expected tokens per unit cost."""

from __future__ import annotations

import heapq
from typing import Sequence


def allocate(fixed: Sequence[int], probs: Sequence[Sequence[float]], costs: dict[int, float], overhead_ms: float,
             max_rows: int) -> list[int]:
    """Return draft prefixes per stream; fixed rows always run, and absent costs make the widest round win."""

    counts = [0] * len(probs)
    rows = sum(fixed)
    expected = float(rows)

    def rate(total: int) -> float:
        cost = costs.get(total) if costs else 1.0
        return -1.0 if cost is None else expected / (cost + overhead_ms)

    best, best_counts = rate(rows), list(counts)
    heap = [(-p[0], s) for s, p in enumerate(probs) if p]
    heapq.heapify(heap)
    while heap and rows < max_rows:
        neg, s = heapq.heappop(heap)
        counts[s] += 1
        rows += 1
        expected -= neg
        if counts[s] < len(probs[s]):
            heapq.heappush(heap, (-probs[s][counts[s]], s))
        now = rate(rows)
        if now > best:
            best, best_counts = now, list(counts)
    return best_counts


def chain_probabilities(rates: Sequence[float], count: int) -> list[float]:
    """A chain's nodes' chances from per-depth acceptance (the j-th lands if every earlier one did)."""

    out, reach = [], 1.0
    for j in range(count):
        reach *= rates[j] if j < len(rates) else rates[-1]
        out.append(reach)
    return out


__all__ = ["allocate", "chain_probabilities"]
