#!/usr/bin/env python3
"""tf-envelope P2 core v2 — measured monotonic envelope for TensorFold admission pricing.

Spec: ENVELOPE_SPEC v1 as amended by P1_SYNTHESIS (round-1 reviews: GLM-5.3-high,
M3.1, spark-Flash). v2 changes adopted:
  D1  MARGIN = 1.5 (n=1 residual needs one-sided headroom; ratchet-down on evidence)
  D2  prices are NEVER clamped to budget (a clamp mutes admission_refused_total);
      refusal = unclamped price + live terms > budget, upstream in Admission
  D3  step-hull: with a declared regime switch, pre/post-switch points price on
      separate hulls (no convex smear across the step)
  D5  outlier gate k=2.5, m=3 consecutive same-decade, >=60 s between accepted
      repeats; suspect vertices RETIRE after 3 same-decade samples price below them
  D7  seeds must be RAW probe peaks (caller's obligation; measure() mutates s1/s2)
  D8  beyond the last observed point: declared family -> slope extension;
      undeclared -> caller falls back to the legacy fit (see spec; a steep
      short-seed extension silently under-prices the post-switch step)
  D4/D6 live server-side (sampling eligibility + decade-bucketed concurrent work
      term); not in this module.

GPU-free: pure policy, no MLX imports. `python3 tf_envelope_hull.py` runs selftests.
"""
from __future__ import annotations

import threading
import time

MARGIN = 1.5                    # D1: one-sided risk (refusal visible, OOM fatal)
OUTLIER_K = 2.5                 # D5: reject > k x current price at that x...
OUTLIER_M = 3                   # ...unless m consecutive same-decade samples...
MIN_REPEAT_S = 60.0             # ...spaced >= 60 s apart (co-tenant windows span fills)
RETIRE_N = 3                    # a suspect vertex retires after 3 same-decade belows
WARM_FRACTION = 0.1             # cached_tokens >= 0.1*L => warm, discard (poison-low)
HULL_CAP = 256
DECADE = 8_000                  # bucket width for "same size class" / consecutive rules


def upper_hull(points):
    """Andrew's monotone chain, upper hull, x strictly increasing.
    y is FORCED to its running max first: real observations are non-monotone
    (measured: relocated peaks 9.916 then 9.871 GiB) and a decreasing final chord
    would make price() non-monotone — the envelope must never price down."""
    pts = sorted(set(points))
    running: list[tuple[float, float]] = []
    best = float("-inf")
    for x, y in pts:
        best = max(best, y)
        if running and running[-1][0] == x:
            running[-1] = (x, max(running[-1][1], y))
        else:
            running.append((x, best))
    hull = []
    for p in running:
        while len(hull) >= 2 and (
                (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])
                <= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0])):
            hull.pop()
        hull.append(p)
    return hull


def hull_price(hull, L: int, extend: bool = True) -> float:
    """Price at L: interpolation on the hull; past the last point, last-segment slope
    (UNCLAMPED — D2); at or below the first point, the first point's y (flat floor)."""
    if not hull:
        raise ValueError("empty hull")
    if L <= hull[0][0]:
        return hull[0][1]
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if L <= x2:
            return y1 + (y2 - y1) * (L - x1) / (x2 - x1)
    if not extend or len(hull) < 2:
        return hull[-1][1]
    (x1, y1), (x2, y2) = hull[-2], hull[-1]
    slope = (y2 - y1) / (x2 - x1) if x2 > x1 else 0.0
    return max(0.0, y2 + slope * (L - x2))


class PrefillEnvelope:
    """Thread-safe monotonic envelope over measured (prompt_len, peak_bytes) points.
    switch=None prices one hull; an int switch prices pre/post on separate hulls (D3)."""

    def __init__(self, seed_points, legacy_fit=None, switch=None, margin: float = MARGIN):
        self._lock = threading.RLock()
        self._switch = int(switch) if switch else None
        groups = self._split(seed_points)
        self._hulls = {side: upper_hull(pts) for side, pts in groups.items()}
        self._margin = float(margin)
        self._legacy = legacy_fit
        self._pending: dict[int, list] = {}       # decade -> consecutive high samples
        self._suspects: dict[int, tuple] = {}     # vertex x -> (side, accepted_at)
        self._below: dict[int, int] = {}          # suspect x -> consecutive belows
        self._last: dict[int, float] = {}         # decade -> last high-sample time
        self._clock = time.monotonic
        self.stats = {"accepted": 0, "warm_rejected": 0, "outlier_rejected": 0,
                      "outlier_confirmed": 0, "poison_low_ignored": 0, "retired": 0}

    def _side(self, L: int) -> int:
        if self._switch is None:
            return 0
        return 0 if L < self._switch else 1

    def _split(self, points):
        groups: dict[int, list] = {0: []} if self._switch is None else {0: [], 1: []}
        for x, y in points:
            groups[self._side(int(x))].append((x, y))
        return groups

    # -- reading -------------------------------------------------------------
    def _hull(self, side: int):
        return self._hulls.get(side) or []

    def price(self, L: int, beyond: str = "extend") -> float:
        """Workspace price at prompt length L (bytes, UNCLAMPED, margin included).

        beyond="extend": past the last observed point, last-segment slope extension
        (for families that DECLARE their regime switch — the seeds sit past it).
        beyond="legacy": past the last observed point, delegate to the legacy fit
        (undeclared families — extension from possibly wrong-regime seeds silently
        under-prices; the shipped fit's over-price is at least known and counted).
        """
        L = int(L)
        with self._lock:
            hull = self._hull(self._side(L))
            margin = self._margin
        if not hull:
            return float(self._legacy(L)) if self._legacy else 0.0
        if L > hull[-1][0] and beyond == "legacy" and self._legacy is not None:
            return float(self._legacy(L))
        return margin * hull_price(hull, L)

    def snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            pts = sorted({(x, y) for h in self._hulls.values() for x, y in h})
        return pts

    # -- writing (admission-time learning; eligible SOLO fills only — D4 upstream) --
    def observe(self, L: int, peak_bytes: float, cached_tokens: int = 0) -> bool:
        """Offer one solo-fill sampled peak. Returns True if it became a hull point."""
        L = int(L)
        if cached_tokens >= WARM_FRACTION * max(1, L):
            with self._lock:
                self.stats["warm_rejected"] += 1
            return False
        decade = L // DECADE
        now = self._clock()
        with self._lock:
            side = self._side(L)
            hull = self._hull(side)
            current = hull_price(hull, L) if hull else 0.0

            # retirement first: a suspect vertex with RETIRE_N same-decade belows drops
            for sx in [x for x in list(self._suspects) if abs(x - L) <= DECADE]:
                if peak_bytes < self._margin * hull_price(self._hull(self._suspects[sx][0]), sx):
                    # "prices below" includes the margin: D5's letter (GLM r2 found the
                    # /OUTLIER_K reading leaves a 2.5x-camouflaged scar in place)
                    self._below[sx] = self._below.get(sx, 0) + 1
                    if self._below[sx] >= RETIRE_N:
                        s_side = self._suspects[sx][0]
                        self._hulls[s_side] = upper_hull(
                            [p for p in self._hull(s_side) if p[0] != float(sx)])
                        del self._suspects[sx]
                        del self._below[sx]
                        self.stats["retired"] += 1
                else:
                    self._below[sx] = 0

            if current > 0 and peak_bytes > OUTLIER_K * current:
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
            point = (float(L), float(peak_bytes))
            same = [y for x, y in hull if x == point[0]]
            if same and point[1] <= max(same):
                self.stats["poison_low_ignored"] += 1
                return False
            if not same and point[1] < hull_price(hull, L) - 1e-9:
                # below the price the hull already charges at this x (chord or
                # extension): accepting it would slope the hull DOWN and lower
                # prices at every larger length — poison-low, record and keep pricing
                self.stats["poison_low_ignored"] += 1
                return False
            candidate = upper_hull(hull + [point])
            for x, _ in hull:                      # hull only grows
                if hull_price(candidate, int(x)) < hull_price(hull, int(x)) - 1e-9:
                    self.stats["poison_low_ignored"] += 1
                    return False
            if len(candidate) > HULL_CAP:
                near = min((x for x, _ in candidate if x != point[0]),
                           key=lambda x: abs(x - point[0]), default=None)
                candidate = upper_hull([p for p in candidate if p[0] != near])
            if current > 0 and point[1] > OUTLIER_K * current:
                self._suspects[int(point[0])] = (side, now)
                self._below[int(point[0])] = 0
            self._hulls[side] = candidate          # copy-on-write swap
            self.stats["accepted"] += 1
            return True

    def margin(self) -> float:
        with self._lock:
            return self._margin


# ---------------------------------------------------------------- unit tests
def _selftest() -> int:
    GiB = 1024 ** 3
    # DECLARED-family seeds (GLM-5.3, probes relocated past the 2048 switch)
    seeds = [(64.0, 0.5 * GiB), (4160.0, 9.9 * GiB), (6208.0, 9.9 * GiB)]
    env = PrefillEnvelope(seeds, switch=2048)

    assert abs(env.price(4160) - MARGIN * 9.9 * GiB) < 1e6, env.price(4160) / GiB
    prices = [env.price(L) for L in range(1, 300_000, 997)]
    assert all(b >= a for a, b in zip(prices, prices[1:])), "price regressed"
    assert env.price(10) == env.price(64)                    # flat floor below seeds
    assert env.price(10_000_000) > 0                         # D2: unclamped extension
    assert env.price(2_000) < env.price(33_000)              # D3: short prompt pre-side
    truth = lambda L: 9.8 * GiB if L > 2048 else 0.3 * GiB
    for L in (33_000, 89_000, 113_000, 133_000, 155_000, 161_000, 200_000):
        p = env.price(L)
        assert p >= truth(L) * 0.999, f"under-price at {L}: {p/GiB:.2f}"
        assert p <= 3 * truth(L) * 1.55, f"over-price at {L}: {p/GiB:.2f}"
    # D8 motivation: steep short seeds + extension blows up — caller falls back instead
    env_bad = PrefillEnvelope([(64.0, 0.5 * GiB), (2112.0, 4.0 * GiB), (4160.0, 9.9 * GiB)])
    assert env_bad.price(200_000) > 100 * GiB, "steep-seed extension blows up (why D8)"
    # D2, stronger: only the steep-seed environment discriminates a budget clamp —
    # its extension must exceed the 202 GiB budget class (14.85 GiB would not).
    assert env_bad.price(10_000_000) > 202 * GiB, "a budget clamp would mute this (D2)"

    # D1 fixture: margin derives from the receipt residuals (recompute, don't vibe).
    # Receipt pairs (observed GiB, priced-flat GiB) at the seven lengths, relocated
    # flat price 9.89: worst observed/price = 9.799/9.89. MARGIN must dominate it.
    residuals = [(9.799, 9.89), (9.547, 9.89), (9.290, 9.89), (9.125, 9.89),
                 (9.848, 9.89), (8.916, 9.89), (8.584, 9.89)]
    need = max(o / p for o, p in residuals)
    assert MARGIN > need, f"MARGIN {MARGIN} must exceed worst receipt residual {need:.3f}"
    # ratchet-down bound: evidence threshold stays above the residual too
    assert 1.35 > need

    # eviction path keeps the hull-grows invariant (GLM r2: untested bypass)
    env_e = PrefillEnvelope(seeds, switch=2048)
    base_pts = env_e.snapshot()
    for i in range(300):
        env_e.observe(5_000 + 131 * i, 10.2 * GiB)   # > current price: accepts, evicts at cap
        after = env_e.snapshot()
        for x, _ in base_pts:
            assert hull_price(upper_hull(after), int(x)) >= hull_price(upper_hull(base_pts), int(x)) - 1e-9, \
                f"eviction lowered the price at previously-observed x={x}"
    print("eviction-path invariant held across", 300, "evicting accepts")

    # MEASURED regression (seedpeaks.out, 2026-10-03): relocated raw peaks are
    # NON-MONOTONE (9.916 then 9.871 GiB); the running-max pre-pass must keep
    # pricing monotone and never below any observed point.
    env_m = PrefillEnvelope([(64.0, 0.634 * GiB), (4160.0, 9.916 * GiB), (6208.0, 9.871 * GiB)],
                            switch=2048)
    mprices = [env_m.price(L) for L in range(1, 600_000, 997)]
    assert all(b >= a for a, b in zip(mprices, mprices[1:])), "measured seeds broke monotonicity"
    assert abs(env_m.price(6208) - MARGIN * 9.916 * GiB) < 10**6
    assert env_m.price(200_000) >= MARGIN * 9.916 * GiB

    # learning: above-hull solo observation accepted, prices rise; poison-low ignored
    before = env.price(100_000)
    assert env.observe(100_000, 10.1 * GiB, cached_tokens=0) is True
    assert env.price(100_000) >= before
    assert env.observe(100_000, 1.0 * GiB, cached_tokens=0) is False
    assert env.price(100_000) >= 10.1 * GiB * env.margin() * 0.999
    n = env.stats["warm_rejected"]
    env.observe(100_000, 50 * GiB, cached_tokens=50_000)
    assert env.stats["warm_rejected"] == n + 1

    # D5: outlier needs m=3 consecutive, spaced >= MIN_REPEAT_S
    env2 = PrefillEnvelope(seeds, switch=2048)
    assert env2.observe(50_000, 60 * GiB) is False           # 1st: rejected
    assert env2.price(50_000) < 60 * GiB
    env2._clock = lambda: time.monotonic() + 3600
    assert env2.observe(50_000, 60 * GiB) is False           # 2nd: rejected
    env2._clock = lambda: time.monotonic() + 7200
    assert env2.observe(50_000, 60 * GiB) is True            # 3rd: confirmed, suspect
    assert env2.price(50_000) >= 60 * GiB * 0.999
    # retirement: 3 same-decade samples below k x price drop the suspect vertex
    env2._clock = lambda: time.monotonic() + 10800
    for _ in range(RETIRE_N):
        env2.observe(50_500, 10.0 * GiB)
    assert env2.price(50_000) < 60 * GiB, "suspect vertex should have retired"
    assert env2.stats["retired"] == 1

    env3 = PrefillEnvelope(seeds, switch=2048)
    for i in range(400):
        env3.observe(5_000 + 97 * i, 9.9 * GiB + i * 1e6)
    assert len(env3.snapshot()) <= HULL_CAP + 8, len(env3.snapshot())
    env4 = PrefillEnvelope([], legacy_fit=lambda L: 6.8 * GiB)
    assert env4.price(200_000) == 6.8 * GiB                  # empty hull -> legacy, never 0

    import random
    env5 = PrefillEnvelope(seeds, switch=2048)
    errors = []

    def worker(rng):
        try:
            for _ in range(2_000):
                L = rng.randrange(1_000, 250_000)
                env5.observe(L, rng.uniform(8, 12) * GiB)
                env5.price(L)
                env5.snapshot()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    rngs = [random.Random(i) for i in range(8)]
    threads = [threading.Thread(target=worker, args=(r,)) for r in rngs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors[:3]
    pts = env5.snapshot()
    assert all(pts[i][0] < pts[i + 1][0] for i in range(len(pts) - 1)), "hull corrupted"
    print("SELFTEST PASS", {k: v for k, v in env5.stats.items()}, f"points={len(pts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
