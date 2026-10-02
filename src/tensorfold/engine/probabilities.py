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


class LabelProbabilities(Probabilities):
    """A decision's collector: at the prompt's last position, the logits of ``labels`` and the full-vocabulary
    logsumexp, besides the sampled token's row. ``capture`` fills ``label_logits`` and ``logsumexp``."""

    def __init__(self, labels, start: int):
        super().__init__(0, start, 1)
        self.labels = [int(token) for token in labels]
        self.label_logits: list[float] | None = None
        self.logsumexp: float | None = None

    def add_labels(self, position: int, logits: list[float], logsumexp: float) -> None:
        if position == self.start:
            self.label_logits, self.logsumexp = logits, logsumexp
