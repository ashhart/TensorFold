"""How deep a draft chain runs: a draft is verified while its chance of landing pays for the row it adds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    """Measured ms on this GPU: ``verify[r]`` for a window of r rows (index 0 unused), ``level`` for one chain level."""

    verify: tuple[float, ...]
    level: float

    @classmethod
    def measured(cls, verify, level: float) -> Costs:
        """Timed costs with each extra row's ms pooled to never rise with rows, as shared experts make them."""

        pools: list[list[float]] = []                  # [sum, count] of adjacent rows, means non-increasing
        for a, b in zip(verify[1:], verify[2:]):
            pools.append([b - a, 1])
            while len(pools) > 1 and pools[-2][0] * pools[-1][1] < pools[-1][0] * pools[-2][1]:
                total, count = pools.pop()
                pools[-1][0] += total
                pools[-1][1] += count
        out = [0.0, float(verify[1])]
        for total, count in pools:
            base = out[-1]
            out.extend(base + total / count * (i + 1) for i in range(count))
        return cls(tuple(round(v, 4) for v in out), round(max(float(level), 0.0), 4))

    def row(self, k: int) -> float:
        """The ms the row carrying draft ``k`` adds to its window."""

        return max(self.verify[k + 1] - self.verify[k], 0.0)


class DepthRule:
    """The first draft, then draft k while P_k ** power >= tokens a ms earned x its row's ms (or P_k >= ``floor``)."""

    def __init__(self, costs: Costs, most: int, *, floor: float | None = None, power: float = 1.0,
                 weight: float = 0.125):
        if not 1 <= most <= len(costs.verify) - 2:
            raise ValueError(f"a chain of {most} drafts needs window costs to {most + 1} rows")
        if floor is not None and not 0.0 <= floor <= 1.0:
            raise ValueError(f"a confidence floor is a probability, not {floor}")
        self.costs, self.most, self.floor, self.weight = costs, int(most), floor, float(weight)
        self.power = float(power)
        self.tokens, self.ms = 2.0, costs.verify[2] + costs.level      # a prior: two tokens from a two-row round

    @property
    def rate(self) -> float:
        """Tokens a ms the recent rounds earned, by the cost model."""

        return self.tokens / self.ms

    def keep(self, k: int, run: float) -> bool:
        """Verify draft ``k`` whose running confidence is ``run`` (every drafted window verifies its first draft)."""

        if k == 1:
            return True
        if self.floor is not None:
            return run >= self.floor
        return run ** self.power >= self.rate * self.costs.row(k)

    def more(self, k: int, run: float) -> bool:
        """Draft level k + 1 only if a certain draft there would pay for its row and its level."""

        if k >= self.most:
            return False
        if self.floor is not None:
            return run >= self.floor
        return run ** self.power >= self.rate * (self.costs.row(k + 1) + self.costs.level)

    def done(self, tokens: int, rows: int, levels: int) -> None:
        """Follow a round from its modeled cost, never a clock, so the ranks of a split model choose alike."""

        w = self.weight
        self.tokens += w * (tokens - self.tokens)
        self.ms += w * (self.costs.verify[rows] + levels * self.costs.level - self.ms)
