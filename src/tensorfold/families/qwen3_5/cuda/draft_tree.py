"""The DFlash2 tree policy: which of the drafter's candidates a round verifies, and where they hang."""

from __future__ import annotations

import heapq

import numpy as np

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

# the Metal engine's values; the target's keyed Gumbel noise is known ahead, so ``noise`` weighs it into draft scores
POLICY = {"edge": 0.6, "noise": 0.7, "scale": 1.5, "branch": 4, "nucleus": False}


def best_first(cands: np.ndarray, unary: np.ndarray, projected: np.ndarray,
               pred: np.ndarray, succ: np.ndarray, anchor: int, max_nodes: int,
               sampling: Sampling | None, first_position: int) -> tuple[list[int], list[int], list[float]]:
    """Metal's best-first lattice policy on the host: nodes, parents and path scores in pop order, parents first."""

    depth_count = cands.shape[0]
    temp = max(float(sampling.temperature), 1e-6) if sampling is not None else 1.0
    noise = None
    if sampling is not None:
        positions = first_position + np.arange(depth_count)
        noise = -np.log(-np.log(uniform_rows(sampling.seed, positions, cands)))
    successor = [succ[cands[d]].astype(np.float64) for d in range(depth_count)]
    heap: list[tuple[float, int, int, int]] = []
    tokens: list[int] = []
    parents: list[int] = []

    def expand(token: int, depth: int, parent: int, path_score: float) -> None:
        edge = successor[depth] @ (pred[token].astype(np.float64) * projected[depth])
        values = (unary[depth] + POLICY["edge"] * edge) / temp
        if noise is not None:
            if POLICY["nucleus"] and sampling.top_p < 1:
                probs = np.exp(values - values.max())
                probs /= probs.sum()
                order = np.argsort(-probs)
                keep = int(np.searchsorted(np.cumsum(probs[order]), sampling.top_p)) + 1
                values = np.full_like(values, -np.inf)
                values[order[:keep]] = ((unary[depth] + POLICY["edge"] * edge) / temp)[order[:keep]]
            values = values + POLICY["noise"] * noise[depth]
        values = values / POLICY["scale"]
        values -= values.max()
        logp = values - np.log(np.exp(values).sum())
        for i in np.argsort(-logp)[:POLICY["branch"]]:
            if not np.isfinite(logp[i]):
                break
            heapq.heappush(heap, (path_score - float(logp[i]), parent,
                                  int(cands[depth, i]), depth))

    expand(anchor, 0, -1, 0.0)
    scores: list[float] = []
    while heap and len(tokens) < max_nodes:
        score, parent, token, depth = heapq.heappop(heap)
        me = len(tokens)
        tokens.append(token)
        parents.append(parent)
        scores.append(score)
        if depth + 1 < depth_count:
            expand(token, depth + 1, me, score)
    return tokens, parents, scores


def allocate(scores: list[list[float]], rows: int, tokens: float, cost, overhead: float = 0.0) -> list[int]:
    """How many of each tree's pop-order nodes (at least one) a round verifies, for the most expected tokens per ms."""

    counts = [1 if s else 0 for s in scores]
    rows += sum(counts)
    tokens += sum(float(np.exp(-s[0])) for s in scores if s)
    heap = [(s[1], i, 1) for i, s in enumerate(scores) if len(s) > 1]
    heapq.heapify(heap)
    taken: list[int] = []
    best, best_k = tokens / (cost(rows) + overhead), 0
    while heap:
        score, i, k = heapq.heappop(heap)
        taken.append(i)
        tokens += float(np.exp(-score))
        rate = tokens / (cost(rows + len(taken)) + overhead)
        if rate > best:
            best, best_k = rate, len(taken)
        if k + 1 < len(scores[i]):
            heapq.heappush(heap, (scores[i][k + 1], i, k + 1))
    for i in taken[:best_k]:
        counts[i] += 1
    return counts
