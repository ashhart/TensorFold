"""CPU best-first lattice search and prior scoring for DFlash2 drafts."""

from __future__ import annotations

from typing import Any, Sequence


def best_first_tree(cands: Any, unary: Any, hproj: Any, noise: Any, anchor: int, pred_code: Any, succ_code: Any, *,
                    temperature: float, edge: float, noise_weight: float, tau: float, children: int,
                    max_nodes: int, prior: Any = None, history: Sequence[int] = (),
                    node_scores: list[float] | None = None) -> tuple[list[int], list[int]]:
    """Return tokens and parent indices by summed sibling log-probability, with optional history bonuses and popped node scores."""

    import heapq

    import numpy as np

    depth_count = int(cands.shape[0])
    count = int(children)                  # children expanded under each node
    t = max(float(temperature), 1e-6) if noise is not None else 1.0
    tables: list[Any] = [None] * depth_count

    def scores(edges: Any, d: int, bonus: Any = None) -> Any:
        s = unary[d] + edge * edges / t
        if noise is not None:
            s = s + noise_weight * noise[d]
        s = s / tau
        if bonus is not None:
            s = s + bonus
        s = s - s.max()
        return s - np.log(np.exp(s).sum())

    def expand(token: int, d: int, hist: tuple = ()) -> Any:
        # Build each depth's successor table only when a node first expands there.
        table = tables[d]
        if table is None:
            table = tables[d] = succ_code[cands[d]].astype(np.float64)
        return scores(table @ (pred_code[token] * hproj[d]), d, None if prior is None else prior(hist, d))

    hists: list[tuple] = []                # each node's last three tokens (with a prior only)
    root_hist = tuple(([None] * 3 + [int(x) for x in list(history)[-3:]])[-3:]) if prior is not None else ()
    root = expand(anchor, 0, root_hist)
    tokens: list[int] = []
    parents: list[int] = []
    heap: list[tuple[float, int, int, int, int]] = []
    for i in np.argsort(-root)[:count]:
        heapq.heappush(heap, (-float(root[i]), -1, int(cands[0][i]), 0, int(i)))
    while heap and len(tokens) < max_nodes:
        neg, parent, token, depth, index = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        if node_scores is not None:
            node_scores.append(-neg)
        me = len(tokens) - 1
        hist = ()
        if prior is not None:
            above = hists[parent] if parent >= 0 else root_hist
            hist = (above[1], above[2], token)
            hists.append(hist)
        if depth + 1 < depth_count:
            ls = expand(token, depth + 1, hist)
            for j in np.argsort(-ls)[:count]:
                heapq.heappush(heap, (neg - float(ls[j]), me, int(cands[depth + 1][j]), depth + 1, int(j)))
    return tokens, parents


def lattice_gain(cands: Any, unary: Any, hproj: Any, noise: Any, anchor: int, pred_code: Any, succ_code: Any,
                 truth: Sequence[int], prior: Any, history: Sequence[int], *, temperature: float, edge: float,
                 noise_weight: float, tau: float) -> float:
    """Sum the prior-induced gain in true-token sibling log-probability while truth remains in the lattice."""

    import numpy as np

    t = max(float(temperature), 1e-6) if noise is not None else 1.0
    hist = tuple(([None] * 3 + [int(x) for x in list(history)[-3:]])[-3:])
    parent = int(anchor)
    gain = 0.0
    for d in range(min(int(cands.shape[0]), len(truth))):
        token = int(truth[d])
        hits = np.flatnonzero(cands[d] == token)
        if not hits.size:
            break
        j = int(hits[0])
        s = unary[d] + edge * (succ_code[cands[d]].astype(np.float64) @ (pred_code[parent] * hproj[d])) / t
        if noise is not None:
            s = s + noise_weight * noise[d]
        s = s / tau
        b = s + prior(hist, d)
        s_top, b_top = s.max(), b.max()
        gain += float((b[j] - b_top - np.log(np.exp(b - b_top).sum())) - (s[j] - s_top - np.log(np.exp(s - s_top).sum())))
        hist = (hist[1], hist[2], token)
        parent = token
    return gain


__all__ = ["best_first_tree", "lattice_gain"]
