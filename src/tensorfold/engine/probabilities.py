"""Request-owned target probabilities, keyed by absolute position across replay."""

import math


class Probabilities:
    def __init__(self, top: int, start: int, count: int):
        self.top, self.start, self.count = top, start, count
        self.rows: dict[int, dict] = {}

    def add(self, positions, tokens, values, top_ids, top_values):
        for pos, token, value, ids, scores in zip(positions, tokens, values, top_ids, top_values, strict=True):
            if not self.start <= pos < self.start + self.count:
                continue
            if not all(math.isfinite(x) for x in (value, *scores)):
                raise RuntimeError("target log probabilities are not finite")
            row = {"id": int(token), "logprob": value, "top": list(zip(ids, scores, strict=True))}
            old = self.rows.get(pos)
            if old is not None and old != row:
                raise RuntimeError("a replay changed the target log probabilities")
            self.rows[pos] = row

    def emitted(self, tokens):
        rows = [self.rows[self.start + i] for i in range(len(tokens))]
        if [row["id"] for row in rows] != list(tokens):
            raise RuntimeError("target probabilities do not match emitted tokens")
        return rows
