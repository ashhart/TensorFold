"""Draft-node landing probabilities by depth and path score, non-decreasing in score within each depth bin."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

DEPTH_EDGES = (0, 1, 2, 3, 5, 8)
SCORE_EDGES = (-6.0, -4.5, -3.5, -2.8, -2.2, -1.7, -1.3, -1.0, -0.75, -0.5, -0.3, -0.15, -0.05)


class Calibration:
    def __init__(self, table: Sequence[Sequence[float]], depth_edges: Sequence[float] = DEPTH_EDGES,
                 score_edges: Sequence[float] = SCORE_EDGES) -> None:
        self.table = [tuple(float(p) for p in row) for row in table]
        self.depth_edges, self.score_edges = tuple(depth_edges), tuple(score_edges)
        if len(self.table) != len(self.depth_edges) or any(len(r) != len(self.score_edges) + 1 for r in self.table):
            raise ValueError("a calibration table has a row a depth bin and a column a score bin")

    def as_dict(self) -> dict[str, Any]:
        return {"depth_edges": list(self.depth_edges), "score_edges": list(self.score_edges),
                "table": [[round(p, 4) for p in row] for row in self.table]}

    def probability(self, depth: int, score: float) -> float:
        row = max(0, bisect.bisect_right(self.depth_edges, int(depth)) - 1)
        return self.table[row][bisect.bisect_left(self.score_edges, float(score))]

    def probabilities(self, parents: Sequence[int], scores: Sequence[float]) -> list[float]:
        """Each node's probability; ``parents`` index the nodes (-1: the pending row), so depths follow."""

        depths: list[int] = []
        for q in parents:
            depths.append(0 if q < 0 else depths[q] + 1)
        return [self.probability(d, s) for d, s in zip(depths, scores)]


def load(path: str | Path) -> dict[str, Calibration]:
    """A calibration file's tables by sampling regime."""

    data = json.loads(Path(path).read_text())
    return {name: Calibration(t["table"], t["depth_edges"], t["score_edges"]) for name, t in data["tables"].items()}


def save(path: str | Path, tables: dict[str, Calibration], source: dict[str, Any]) -> None:
    data = {"source": source, "tables": {name: table.as_dict() for name, table in tables.items()}}
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def fit(samples: Iterable[tuple[int, float, bool]], depth_edges: Sequence[float] = DEPTH_EDGES,
        score_edges: Sequence[float] = SCORE_EDGES) -> Calibration:
    """Fit smoothed bin rates, pooling adjacent violators by count so each depth row is non-decreasing in score."""

    rows, cols = len(depth_edges), len(score_edges) + 1
    hits = [[0.0] * cols for _ in range(rows)]
    counts = [[0.0] * cols for _ in range(rows)]
    for depth, score, landed in samples:
        r = max(0, bisect.bisect_right(depth_edges, int(depth)) - 1)
        c = bisect.bisect_left(score_edges, float(score))
        hits[r][c] += 1.0 if landed else 0.0
        counts[r][c] += 1.0
    table = []
    for r in range(rows):
        rates = [(h + 0.5) / (n + 1.0) for h, n in zip(hits[r], counts[r])]
        table.append(_pooled(rates, [n + 1.0 for n in counts[r]]))
    return Calibration(table, depth_edges, score_edges)


def _pooled(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out: list[float] = []
    for v, _, n in blocks:
        out.extend([v] * int(n))
    return out


__all__ = ["Calibration", "DEPTH_EDGES", "SCORE_EDGES", "fit", "load", "save"]
