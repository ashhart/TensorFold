"""--mtp-cost: price a later MTP draft by its calibrated chance of being kept against the ms it adds to the round."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch

RATE = 0.05          # EMA weight of one round's observation


class DraftPrice:
    """The measured round costs and the head's chain product calibrated per depth by the kept/drafted counts of live
    rounds (greedy and sampled apart)."""

    def __init__(self, cost: float, verify_ms: Sequence[float], draft_ms: float, depth: int) -> None:
        ms = list(verify_ms)
        while len(ms) < depth + 3:                  # windows past the measured table repeat its last step
            ms.append(ms[-1] + (ms[-1] - ms[-2]) if len(ms) > 1 else ms[-1])
        self.cost, self.verify, self.draft_ms, self.depth = float(cost), ms, float(draft_ms), int(depth)
        self.kept = {False: [None] * depth, True: [None] * depth}   # EMA of "drafts 1..j all kept"
        self.said = {False: [None] * depth, True: [None] * depth}   # EMA of the head's product for those drafts
        self.sampled = False
        self.products: list[float] = []

    def adds(self, j: int) -> float:
        """ms that verifying draft j (0-based) adds to the round: one more verify row and its MTP step."""

        return self.verify[j + 1] - self.verify[j] + self.draft_ms

    def chance(self, j: int, product: float) -> float:
        kept, said = self.kept[self.sampled][j], self.said[self.sampled][j]
        if kept is None or said <= 0:
            return product
        return min(1.0, product * kept / said)

    def pays(self, j: int, product: float) -> bool:
        """Whether draft j, at head product ``product``, repays the ms it adds at ``cost`` tokens per ms."""

        return self.chance(j, product) >= self.cost * self.adds(j)

    def begin(self, sampled: bool) -> None:
        self.sampled, self.products = sampled, []

    def observe(self, verified: int, kept: int) -> None:
        """One verified round: of its ``verified`` drafts the first ``kept`` were kept."""

        k, s = self.kept[self.sampled], self.said[self.sampled]
        for j, p in enumerate(self.products[:verified]):
            if k[j] is None:                        # seeded by the head's product: scale 1 until rounds say more
                k[j] = s[j] = p
            k[j] += RATE * (float(j < kept) - k[j])
            s[j] += RATE * (p - s[j])

    def describe(self) -> str:
        v = self.verify
        return (f"verify {v[0]:.1f}-{v[self.depth]:.1f} ms for 1-{self.depth + 1} rows, a draft step "
                f"{self.draft_ms:.2f} ms")


@torch.no_grad()
def measure_round_costs(e, rows: int, reps: int = 5) -> tuple[tuple[float, ...], float]:
    """This engine's verify windows of 1 .. ``rows`` rows and one MTP draft step, ms (medians, after a short prompt),
    on distinct tokens (a window's cost grows with the distinct experts its rows route to); leaves the state empty."""

    from .decode import prefill

    gen = torch.Generator().manual_seed(0)
    lo = min(1000, e.w.cfg.vocab // 4)
    ids = torch.randint(lo, max(lo + 1, e.w.cfg.vocab // 2), (64 + rows,), generator=gen).tolist()
    prefill(e, ids[:64], None)
    tokens = ids[64:]

    def timed(fn) -> float:
        times = []
        for _ in range(reps + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times[1:]))

    verify = [timed(lambda r=r: e.forward(tokens[:r])) for r in range(1, rows + 1)]
    for r in range(1, rows):                       # a wider window never costs less: noise must not reorder bars
        verify[r] = max(verify[r], verify[r - 1])
    step = timed(lambda: e.mtp_forward(tokens[:1], e.last_streams)) if e.mbuf is not None else 0.0
    e.kept = None
    e.reset()
    return tuple(round(v, 3) for v in verify), round(step, 3)
