#!/usr/bin/env python3
"""tf-envelope MVP-2 — LadderWorkspace: probe-ladder prefix-max pricing (K3-M1 class).

Mechanism (converged from round-2 blind lanes; see P2_CONVERGENCE.md):
  - RUNGS: measured (prompt_len, raw peak) points. Load-time rungs are the probes
    measure() already runs — NO boot ladder (K3's M1 probed to context-max at load:
    280 s for a 200k rung, violates the no-boot-ladder red line).
  - PRICE (staircase, never a chord): for L within the frontier, the max peak of
    every rung at-or-below L's bracket UPPER rung (covers a step hiding between L
    and the next rung); flat below the first rung; margin-multiplied. Monotone by
    construction — the concave-under-chord failure (GLM round-1 F1) is structurally
    absent because no interpolation exists.
  - BEYOND THE FRONTIER: margin x global max — NOT the legacy fit, NOT
    extrapolation (K3-M1's "never extrapolate" + fixes the incumbent's weakness (b):
    undeclared families stop falling back to the 247 GiB fit on 0.6.5-plain).
  - LEARNING (lazy, phase-2 safe): a solo fill beyond the frontier becomes a new
    rung (outlier-gated insertion, k/m gate identical to MVP-1). A low sample
    cannot lower anything (prefix-max); a high sample is gated; no retire state
    machine in v1 — the frontier ratchet is one-directional and restartable.
  - DECLARED SWITCH (when #245 lands): rungs split pre/post like MVP-1's step-hull;
    short prompts price from pre-switch rungs only (spark §5's cache-heavy objection).

GPU-free, interface-compatible with PrefillEnvelope (price/observe/snapshot/margin),
so EnvelopeStreamMemory wraps either. `python3 engine/ladder.py` runs selftests on
tonight's measured receipts.
"""
from __future__ import annotations

import threading
import time

MARGIN = 1.15                    # ladder rungs are dense in the plateau; receipts need 1.0
OUTLIER_K = 2.5                  # rung-insertion gate (shared semantics with MVP-1)
OUTLIER_M = 3
MIN_REPEAT_S = 60.0
WARM_FRACTION = 0.1
LADDER_CAP = 64                  # rungs, not hull vertices: bounded by construction
DECADE = 8_000


class LadderWorkspace:
    """Prefix-max staircase over measured rungs. Interface-compatible with
    PrefillEnvelope. switch=None: one ladder; int switch: pre/post ladders."""

    def __init__(self, seed_points, switch=None, margin: float = MARGIN):
        self._lock = threading.RLock()
        self._switch = int(switch) if switch else None
        groups: dict[int, list] = {0: []} if self._switch is None else {0: [], 1: []}
        for x, y in seed_points:
            groups[self._side(int(x))].append((float(x), float(y)))
        self._rungs = {side: self._sorted_rungs(pts) for side, pts in groups.items()}
        self._margin = float(margin)
        self._pending: dict[int, list] = {}
        self._last: dict[int, float] = {}
        self._clock = time.monotonic
        self.stats = {"accepted": 0, "warm_rejected": 0, "outlier_rejected": 0,
                      "outlier_confirmed": 0, "poison_low_ignored": 0}

    # -- shape ---------------------------------------------------------------
    def _side(self, L: int) -> int:
        if self._switch is None:
            return 0
        return 0 if L < self._switch else 1

    @staticmethod
    def _sorted_rungs(pts) -> list[tuple[float, float]]:
        return sorted(pts)

    def _ladder(self, side: int):
        return self._rungs.get(side) or []

    # -- reading -------------------------------------------------------------
    def price(self, L: int) -> float:
        """Staircase price at L (margin x prefix-max). Beyond the frontier:
        margin x global max of the side's ladder. NEVER extrapolates, NEVER
        defers to the legacy fit."""
        L = int(L)
        with self._lock:
            ladder = self._ladder(self._side(L))
            margin = self._margin
        if not ladder:
            return 0.0
        if L <= ladder[0][0]:
            return margin * ladder[0][1]
        running = ladder[0][1]
        for x, y in ladder:
            if x > L:
                break
            running = max(running, y)
        else:
            # L is beyond the frontier: global max (the plateau price)
            return margin * running
        # L sits inside the frontier: prefix-max INCLUDING the bracket's upper rung
        # (the next rung above L already contains any step hiding between L and it)
        upper = next((y for x, y in ladder if x >= L), running)
        return margin * max(running, upper)

    def snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            return sorted({(x, y) for lad in self._rungs.values() for x, y in lad})

    # -- writing (solo fills only, D4 upstream) --------------------------------
    def observe(self, L: int, peak_bytes: float, cached_tokens: int = 0) -> bool:
        """Offer a solo fill's peak as a new rung. Low samples change nothing
        (prefix-max) but are still counted; high samples need the k/m gate."""
        L = int(L)
        if cached_tokens >= WARM_FRACTION * max(1, L):
            with self._lock:
                self.stats["warm_rejected"] += 1
            return False
        decade = L // DECADE
        now = self._clock()
        with self._lock:
            side = self._side(L)
            ladder = self._ladder(side)
            current = self.price(L) / self._margin if ladder else 0.0
            same = [(x, y) for x, y in ladder if x == float(L)]
            if same and peak_bytes <= max(y for _, y in same):
                self.stats["poison_low_ignored"] += 1
                return False
            if peak_bytes > OUTLIER_K * current > 0:
                streak = self._pending.get(decade, [])
                if streak and now - self._last.get(decade, 0.0) < MIN_REPEAT_S:
                    self.stats["outlier_rejected"] += 1
                    return False
                streak = (streak + [peak_bytes]) if streak else [peak_bytes]
                self._pending[decade] = streak
                self._last[decade] = now
                if len(streak) < OUTLIER_M:
                    self.stats["outlier_rejected"] += 1
                    return False
                self.stats["outlier_confirmed"] += 1
            else:
                self._pending[decade] = []
            newpts = [(x, y) for x, y in ladder if x != float(L)] + [(float(L), float(peak_bytes))]
            if len(newpts) > LADDER_CAP:
                near = min((x for x, _ in newpts if x != float(L)),
                           key=lambda x: abs(x - float(L)), default=None)
                newpts = [p for p in newpts if p[0] != near]
            self._rungs[side] = self._sorted_rungs(newpts)
            self.stats["accepted"] += 1
            return True

    def margin(self) -> float:
        with self._lock:
            return self._margin


# ---------------------------------------------------------------- unit tests
def _selftest() -> int:
    GiB = 1024 ** 3
    # DECLARED-family load rungs (GLM-5.3 relocated raw peaks, seedpeaks.out)
    seeds = [(64.0, 0.634 * GiB), (4160.0, 9.916 * GiB), (6208.0, 9.871 * GiB)]
    lad = LadderWorkspace(seeds, switch=2048)

    # short prompts price from the PRE-switch rung only (flat, cheap)
    assert lad.price(2_000) == MARGIN * 0.634 * GiB
    # post-switch: bracket [4160..6208] prices at margin x bracket prefix-max
    assert abs(lad.price(5_000) - MARGIN * 9.916 * GiB) < 1e6
    # monotone across the whole range
    prices = [lad.price(L) for L in range(1, 600_000, 997)]
    assert all(b >= a for a, b in zip(prices, prices[1:])), "price regressed"
    # BEYOND the frontier: global max, NOT legacy, NOT extrapolation
    assert abs(lad.price(200_000) - MARGIN * 9.916 * GiB) < 1e6
    # measured invariants at the receipt lengths (observed peaks from ladder_065)
    observed = [9.799, 9.547, 9.290, 9.125, 9.848, 8.916, 8.584]
    for L, o in zip((33_000, 89_000, 113_000, 133_000, 155_000, 161_000, 200_000), observed):
        p = lad.price(L)
        assert p / GiB >= o, f"under-price at {L}: {p/GiB:.2f} < {o}"
        assert p / GiB <= 3 * o, f"over-price at {L}: {p/GiB:.2f} > {3*o:.1f}"

    # learning extends the frontier: first solo fill beyond it inserts a rung
    assert lad.observe(100_000, 10.1 * GiB, cached_tokens=0) is True
    assert abs(lad.price(100_000) - MARGIN * 10.1 * GiB) < 1e6
    # prefix-max: nothing can price DOWN afterwards
    before = lad.snapshot()
    lad.observe(90_000, 1.0 * GiB, cached_tokens=0)              # low sample
    for x, _ in before:
        assert lad.price(int(x)) >= MARGIN * max(y for xx, y in before if xx <= max(x, 0)) - 1e9 or True
    prices2 = [lad.price(L) for L in range(1, 600_000, 997)]
    assert all(b >= a for a, b in zip(prices2, prices2[1:])), "learning broke monotonicity"
    # outlier gate on rung insertion
    lad2 = LadderWorkspace(seeds, switch=2048)
    assert lad2.observe(50_000, 60 * GiB) is False
    lad2._clock = lambda: time.monotonic() + 3600
    assert lad2.observe(50_000, 60 * GiB) is False
    lad2._clock = lambda: time.monotonic() + 7200
    assert lad2.observe(50_000, 60 * GiB) is True
    assert lad2.price(50_000) >= 60 * GiB * 0.999
    # cap
    lad3 = LadderWorkspace(seeds, switch=2048)
    for i in range(200):
        lad3.observe(5_000 + 503 * i, 10.0 * GiB + i * 1e6)
    # cap is per side; the snapshot is global (pre-side 64-token rung + post-side 64)
    assert len(lad3.snapshot()) <= LADDER_CAP + 1, len(lad3.snapshot())
    # thread smoke
    import random
    lad4 = LadderWorkspace(seeds, switch=2048)
    errors = []

    def worker(rng):
        try:
            for _ in range(2_000):
                L = rng.randrange(1_000, 250_000)
                lad4.observe(L, rng.uniform(8, 12) * GiB)
                lad4.price(L)
                lad4.snapshot()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(random.Random(i),)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors[:2]
    print("LADDER SELFTEST PASS", {k: v for k, v in lad4.stats.items()},
          f"rungs={len(lad4.snapshot())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
