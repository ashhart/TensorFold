"""Independent contract tests for tensorfold.engine.ladder.LadderWorkspace.

Derived ONLY from the written contract (rules 1-7); nothing here is taken from
the module's internal _selftest(). Constants (MARGIN_DENSE, MARGIN_SPARSE,
SPARSE_RATIO, LADDER_CAP) are imported from the module under test, as the
contract quotes them.

Contract summary under test:
  1. LadderWorkspace(seed_points, switch=None|int, margin=floor). Rungs are
     (prompt_tokens:int, peak_bytes:float). switch splits rungs into pre
     (x < switch) / post (x >= switch) sides; None = one ladder.
  2. price(L) = margin x prefix-max staircase. Within the frontier: max of
     (every rung peak <= L, the covering bracket's UPPER rung peak -- the next
     rung above L, covering a step hiding between L and that rung). Beyond the
     frontier (L > last rung x): global max of that side's ladder. Below the
     first rung: that rung's y. Empty ladder (or side): 0.0.
  3. Density-aware margin. The covering bracket (nearest rung <= L, nearest
     rung >= L) sets it: (upper_x - lower_x)/lower_x > SPARSE_RATIO ->
     MARGIN_SPARSE, else MARGIN_DENSE. Frontier (no next rung above L) or
     single-rung ladder -> MARGIN_SPARSE. At L <= first rung the constructor
     floor margin applies. The constructor margin is the FLOOR only: it never
     overrides the density rule elsewhere.
  4. Monotone WITHIN each bracket; may step DOWN only exactly at a measured
     rung x. Soundness floor: price(L) >= MARGIN_DENSE x every measured peak
     covered by L's bracket (prefix peaks and the bracket's upper rung).
  5. observe(): warm reject when cached_tokens >= 0.1*max(1, L). Outlier gate:
     peak > 2.5 x current unmargin'd price at L (the raw staircase coverage,
     no margin) needs 3 consecutive same-bucket (L // 8000) samples spaced
     >= 60 s (monotonic clock) to insert; otherwise rejected. Duplicate x with
     lower-or-equal y: poison-low, rejected (stats only). Insertion otherwise
     always allowed and never lowers any price (prefix-max). Cap 64 rungs per
     side, nearest-x eviction.
  6. note_underprice() increments stats['underpriced_events'] under lock.
  7. RLock; snapshot() returns a merged, sorted list across sides.

Synthetic families (all shapes/rung placements differ from
[64, 2112, 4160, 17000, 70000, 240000]):
  step_flat_then_jump : flat dense plateau, one sparse jump, exact-ratio-2.0 bracket
  peak_burden         : rises to a measured peak then falls off it
  falling_staircase   : strictly falling peaks (down-steps at every rung)
  linear_dense_ramp   : 40 evenly spaced rungs, all brackets dense
  two_step_plateau    : flat plateau -> sparse jump -> falling step, exact-ratio-2.0 bracket
  switch_split        : declared switch with seeded pre/post sides
"""
from __future__ import annotations

import itertools
import random
import threading
import time

import pytest

from tensorfold.engine.ladder import (
    LADDER_CAP,
    MARGIN_DENSE,
    MARGIN_SPARSE,
    SPARSE_RATIO,
    LadderWorkspace,
)

EPS = 1e-9


# --------------------------------------------------------------------------- helpers
def _split_sides(seed_points, switch):
    """Contract rule 1: side 0 is x < switch, side 1 is x >= switch."""
    if switch is None:
        return sorted((float(x), float(y)) for x, y in seed_points)
    s = int(switch)
    return {
        0: sorted((float(x), float(y)) for x, y in seed_points if int(x) < s),
        1: sorted((float(x), float(y)) for x, y in seed_points if int(x) >= s),
    }


def contract_covered(rungs, L):
    """Contract rule 2 coverage (no margin): max(prefix-max of rungs <= L,
    next strictly-above rung's peak). L <= first rung -> that rung's y."""
    if not rungs:
        return 0.0
    if L <= rungs[0][0]:
        return rungs[0][1]
    running = rungs[0][1]
    upper = None
    for x, y in rungs:
        if x > L:
            upper = y
            break
        running = max(running, y)
    return running if upper is None else max(running, upper)


def expected_price(seed_points, L, switch=None, floor_margin=MARGIN_SPARSE):
    """Exact contract price (rules 2+3) for an unlearned workspace."""
    sides = _split_sides(seed_points, switch)
    rungs = sides if switch is None else sides[0 if L < int(switch) else 1]
    if not rungs:
        return 0.0
    if L <= rungs[0][0]:
        return floor_margin * rungs[0][1]
    running = rungs[0][1]
    upper = None
    for x, y in rungs:
        if x > L:
            upper = y
            break
        running = max(running, y)
    frontier = upper is None
    covered = running if frontier else max(running, upper)
    if frontier or len(rungs) < 2:
        margin = MARGIN_SPARSE
    else:
        lower = max(x for x, _ in rungs if x <= L)
        upper_x = min(x for x, _ in rungs if x >= L)
        margin = MARGIN_SPARSE if (upper_x - lower) / lower > SPARSE_RATIO else MARGIN_DENSE
    return margin * covered


def soundness_floor(side_rungs, L):
    """Contract rule 4: MARGIN_DENSE x every measured peak covered by L's
    bracket. Coverage below/at the first rung is that rung's own peak (rule 2:
    'below the first rung: that rung's y'; rule 3: 'at L <= first rung the
    floor margin applies' -- the pre-first-rung zone has its own simple
    pricing). Above it, coverage = max(prefix peaks, the bracket's upper-rung
    peak covering a step hiding between L and the next rung)."""
    if L <= side_rungs[0][0]:
        return MARGIN_DENSE * side_rungs[0][1]
    base = max(y for x, y in side_rungs if x <= L)
    above = [x for x, _ in side_rungs if x > L]
    if above:
        base = max(base, next(y for x, y in side_rungs if x == min(above)))
    return MARGIN_DENSE * base


class FakeClock:
    """Replaces the workspace's monotonic clock for gate-spacing tests."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# --------------------------------------------------------------------------- families
def _grid(*parts):
    return sorted(set(itertools.chain(*parts)))


FAMILIES = {
    "step_flat_then_jump": {
        "seeds": [(500, 2.0), (1000, 2.0), (1500, 2.0), (2000, 2.0), (2500, 2.0),
                  (3000, 2.0), (3500, 2.0), (4000, 2.0), (4500, 2.0),
                  (20000, 16.0), (60000, 20.0)],
        "switch": None,
        "grid": _grid(range(0, 80_001, 7),
                      range(249, 252), range(499, 502), range(999, 1002),
                      range(4499, 4502), range(19999, 20002), range(59999, 60002)),
    },
    "peak_burden": {
        "seeds": [(1000, 1.0), (2000, 4.0), (3000, 9.0), (4000, 4.0)],
        "switch": None,
        "grid": _grid(range(0, 12_001, 3),
                      range(999, 1002), range(1999, 2002),
                      range(2999, 3002), range(3999, 4002)),
    },
    "falling_staircase": {
        "seeds": [(100, 9.0), (200, 7.0), (300, 5.0), (400, 3.0), (500, 1.0)],
        "switch": None,
        "grid": _grid(range(0, 3_000, 2), range(95, 506)),
    },
    "linear_dense_ramp": {
        "seeds": [(1000 * k, 1.0 + 0.5 * (k - 1)) for k in range(1, 41)],
        "switch": None,
        "grid": _grid(range(0, 100_001, 13),
                      range(999, 1002), range(1999, 2002),
                      range(38999, 39002), range(39999, 40002)),
    },
    "two_step_plateau": {
        "seeds": [(250, 8.0), (750, 8.0), (1250, 8.0), (1750, 8.0), (2250, 8.0),
                  (40000, 64.0), (120000, 60.0)],
        "switch": None,
        "grid": _grid(range(0, 200_001, 17),
                      range(249, 252), range(2249, 2252),
                      range(39999, 40002), range(119999, 120002)),
    },
    "switch_split": {
        "seeds": [(100, 1.0), (1500, 2.0), (2048, 3.0), (99999, 4.0)],
        "switch": 2048,
        "grid": _grid(range(0, 120_001, 23),
                      range(99, 102), range(1499, 1502), range(2046, 2050)),
    },
}


# --------------------------------------------------------------------------- rule 1
class TestRule1Construction:
    def test_snapshot_sorted_merged_across_sides(self):
        ws = LadderWorkspace([(5000.0, 3.0), (100.0, 1.0), (99999.0, 4.0), (1500.0, 2.0)],
                             switch=2048)
        assert ws.snapshot() == [(100.0, 1.0), (1500.0, 2.0), (5000.0, 3.0), (99999.0, 4.0)]

    def test_snapshot_single_ladder_sorted(self):
        ws = LadderWorkspace([(300.0, 2.0), (100.0, 1.0), (200.0, 1.5)])
        assert ws.snapshot() == [(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]

    def test_unsorted_and_duplicate_free_snapshot_is_a_copy(self):
        ws = LadderWorkspace([(200.0, 2.0), (100.0, 1.0)])
        snap = ws.snapshot()
        snap.append((999.0, 9.0))
        assert ws.snapshot() == [(100.0, 1.0), (200.0, 2.0)]

    def test_int_seeds_coerced_to_float_rungs(self):
        ws = LadderWorkspace([(100, 1), (2000, 2)])
        snap = ws.snapshot()
        assert snap == [(100.0, 1.0), (2000.0, 2.0)]
        assert all(isinstance(x, float) and isinstance(y, float) for x, y in snap)
        assert isinstance(ws.price(150), float)

    def test_margin_accessor_returns_constructor_floor(self):
        assert LadderWorkspace([(100.0, 1.0)], margin=2.5).margin() == 2.5
        assert LadderWorkspace([(100.0, 1.0)]).margin() == MARGIN_SPARSE

    def test_switch_assigns_sides_by_threshold(self):
        ws = LadderWorkspace([(100.0, 1.0), (1500.0, 2.0), (2048.0, 3.0), (99999.0, 4.0)],
                             switch=2048)
        # 2047 prices from the PRE side (bracket 100..1500), 2048 from POST (first rung).
        assert ws.price(2047) == pytest.approx(MARGIN_SPARSE * 2.0, abs=EPS)
        assert ws.price(2048) == pytest.approx(MARGIN_SPARSE * 3.0, abs=EPS)


# --------------------------------------------------------------------------- rule 2
class TestRule2PriceStaircase:
    def test_empty_ladder_prices_zero(self):
        ws = LadderWorkspace([])
        assert ws.price(123) == 0.0
        assert ws.price(0) == 0.0
        assert ws.snapshot() == []

    def test_empty_side_with_switch_prices_zero(self):
        ws = LadderWorkspace([(5000.0, 1.0)], switch=1000)
        assert ws.price(999) == 0.0
        assert ws.price(5000) == pytest.approx(MARGIN_SPARSE * 1.0, abs=EPS)
        assert ws.snapshot() == [(5000.0, 1.0)]

    def test_below_first_rung_flat_at_floor(self):
        ws = LadderWorkspace([(500.0, 3.0), (5000.0, 30.0)], margin=2.0)
        for L in (0, 1, 250, 499, 500):
            assert ws.price(L) == pytest.approx(2.0 * 3.0, abs=EPS)

    def test_bracket_price_includes_upper_rung_peak(self):
        # peak_burden: at L=1500 the prefix-max is 1.0 but the bracket's upper
        # rung (2000) carries 4.0 -> price = MARGIN_DENSE x 4.0.
        ws = LadderWorkspace(FAMILIES["peak_burden"]["seeds"])
        assert ws.price(1500) == pytest.approx(MARGIN_DENSE * 4.0, abs=EPS)

    def test_sparse_bracket_price_includes_hidden_step_headroom(self):
        # step family: L=10000 sits between 4500 (peak 2) and 20000 (peak 16);
        # the unseen step is covered by the upper rung, sparse margin applied.
        ws = LadderWorkspace(FAMILIES["step_flat_then_jump"]["seeds"])
        assert ws.price(10000) == pytest.approx(MARGIN_SPARSE * 16.0, abs=EPS)

    def test_beyond_frontier_is_global_max_no_extrapolation(self):
        ws = LadderWorkspace(FAMILIES["step_flat_then_jump"]["seeds"])
        for L in (60_001, 100_000, 10**9, 10**12):
            assert ws.price(L) == pytest.approx(MARGIN_SPARSE * 20.0, abs=EPS)

    def test_beyond_frontier_is_global_max_not_last_rung(self):
        # Rule 2: beyond the frontier the price is the GLOBAL MAX of the side's
        # ladder. two_step_plateau's max is the 64.0 rung at x=40000, so a
        # falling last rung (60.0 at 120000) must NOT pull the price down.
        ws = LadderWorkspace(FAMILIES["two_step_plateau"]["seeds"])
        assert ws.price(120_001) == pytest.approx(MARGIN_SPARSE * 64.0, abs=EPS)
        assert ws.price(500_000) == pytest.approx(MARGIN_SPARSE * 64.0, abs=EPS)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_exact_price_matches_contract_sweep(name):
    fam = FAMILIES[name]
    ws = LadderWorkspace(fam["seeds"], switch=fam["switch"])
    for L in fam["grid"]:
        want = expected_price(fam["seeds"], L, switch=fam["switch"])
        got = ws.price(L)
        assert got == pytest.approx(want, abs=1e-6, rel=1e-12), f"price({L}) {got} != {want}"


# --------------------------------------------------------------------------- rule 3
class TestRule3DensityMargin:
    def test_dense_bracket_uses_dense_margin(self):
        ws = LadderWorkspace([(1000.0, 10.0), (2000.0, 10.0)])
        assert ws.price(1500) == pytest.approx(MARGIN_DENSE * 10.0, abs=EPS)
        assert ws.price(1999) == pytest.approx(MARGIN_DENSE * 10.0, abs=EPS)

    def test_sparse_bracket_uses_sparse_margin(self):
        ws = LadderWorkspace([(1000.0, 10.0), (4000.0, 10.0)])
        assert ws.price(2500) == pytest.approx(MARGIN_SPARSE * 10.0, abs=EPS)

    def test_ratio_exactly_two_is_dense(self):
        ws = LadderWorkspace([(1000.0, 4.0), (3000.0, 4.0)])
        assert ws.price(2000) == pytest.approx(MARGIN_DENSE * 4.0, abs=EPS)

    def test_ratio_just_over_two_is_sparse(self):
        ws = LadderWorkspace([(1000.0, 4.0), (3001.0, 4.0)])
        assert ws.price(2000) == pytest.approx(MARGIN_SPARSE * 4.0, abs=EPS)

    def test_frontier_margin_is_sparse(self):
        ws = LadderWorkspace([(1000.0, 10.0), (2000.0, 10.0)])
        assert ws.price(5000) == pytest.approx(MARGIN_SPARSE * 10.0, abs=EPS)

    def test_single_rung_frontier_is_sparse(self):
        ws = LadderWorkspace([(500.0, 3.0)])
        assert ws.price(900) == pytest.approx(MARGIN_SPARSE * 3.0, abs=EPS)
        assert ws.price(10**7) == pytest.approx(MARGIN_SPARSE * 3.0, abs=EPS)

    def test_constructor_margin_is_floor_only(self):
        # margin=2.5 applies only at L <= first rung; density rule governs
        # brackets (1.15) and frontier (1.5) regardless of the floor value.
        ws = LadderWorkspace([(1000.0, 10.0), (2000.0, 10.0)], margin=2.5)
        assert ws.price(999) == pytest.approx(2.5 * 10.0, abs=EPS)
        assert ws.price(1000) == pytest.approx(2.5 * 10.0, abs=EPS)
        assert ws.price(1500) == pytest.approx(MARGIN_DENSE * 10.0, abs=EPS)
        assert ws.price(5000) == pytest.approx(MARGIN_SPARSE * 10.0, abs=EPS)

    def test_covering_bracket_sets_margin_around_a_rung(self):
        ws = LadderWorkspace([(1000.0, 10.0), (4000.0, 10.0), (5000.0, 10.0)])
        # just below the 4000 rung: bracket (1000, 4000) is sparse
        assert ws.price(3999) == pytest.approx(MARGIN_SPARSE * 10.0, abs=EPS)
        # at the 4000 rung: degenerate bracket (4000, 4000) is dense, step
        # down at a measured rung is allowed (rule 4)
        assert ws.price(4000) == pytest.approx(MARGIN_DENSE * 10.0, abs=EPS)
        # just above: bracket (4000, 5000) is dense
        assert ws.price(4001) == pytest.approx(MARGIN_DENSE * 10.0, abs=EPS)


# --------------------------------------------------------------------------- rule 4
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_soundness_floor_holds_everywhere(name):
    fam = FAMILIES[name]
    ws = LadderWorkspace(fam["seeds"], switch=fam["switch"])
    snap = ws.snapshot()
    for L in fam["grid"]:
        sides = _split_sides(fam["seeds"], fam["switch"])
        side_rungs = sides if fam["switch"] is None else sides[0 if L < fam["switch"] else 1]
        if not side_rungs:
            assert ws.price(L) == 0.0
            continue
        assert ws.price(L) >= soundness_floor(side_rungs, L) - 1e-6, f"floor broke at {L}"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_price_never_exceeds_sparse_insurance_of_covered(name):
    fam = FAMILIES[name]
    ws = LadderWorkspace(fam["seeds"], switch=fam["switch"])
    for L in fam["grid"]:
        sides = _split_sides(fam["seeds"], fam["switch"])
        side_rungs = sides if fam["switch"] is None else sides[0 if L < fam["switch"] else 1]
        cap = MARGIN_SPARSE * contract_covered(side_rungs, L)
        assert ws.price(L) <= cap + 1e-6, f"insurance cap exceeded at {L}"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_step_downs_occur_only_at_measured_rungs(name):
    fam = FAMILIES[name]
    ws = LadderWorkspace(fam["seeds"], switch=fam["switch"])
    snap = ws.snapshot()
    grid = fam["grid"]
    downs = []
    for a, b in zip(grid, grid[1:]):
        if ws.price(b) < ws.price(a) - 1e-9:
            # A down-step between consecutive sampled L must sit AT a measured
            # rung: either x in [a, b) (density rises as L passes the rung) or
            # x == b (the degenerate bracket [b, b] is dense at the rung). Both
            # are the contract's "exactly at a measured rung x" allowance.
            side_b = 0 if (fam["switch"] is None or b < fam["switch"]) else 1
            xs = [x for x, _ in snap if (fam["switch"] is None or (x < fam["switch"]) == (side_b == 0))]
            assert any(a <= x <= b for x in xs), f"down-step between rungs: {a} -> {b}"
            downs.append(b)
    if name in ("falling_staircase", "two_step_plateau", "step_flat_then_jump"):
        assert downs, "family expected to contain at least one legitimate down-step"


def test_falling_family_absorbs_falling_rungs():
    # Prefix-max: a falling rung cannot lower the running peak, so the price
    # stays flat across the falling rungs; the only drop is the margin step at
    # the FIRST rung's right edge (floor margin -> dense bracket), and the
    # frontier restores the sparse margin (price steps back UP at the last rung).
    ws = LadderWorkspace(FAMILIES["falling_staircase"]["seeds"])
    assert ws.price(99) == pytest.approx(MARGIN_SPARSE * 9.0, abs=EPS)    # floor zone
    assert ws.price(100) == pytest.approx(MARGIN_SPARSE * 9.0, abs=EPS)   # at the rung
    assert ws.price(101) == pytest.approx(MARGIN_DENSE * 9.0, abs=EPS)    # dense bracket
    for L in (200, 201, 300, 301, 400, 401, 499):
        assert ws.price(L) == pytest.approx(MARGIN_DENSE * 9.0, abs=EPS), f"price({L})"
    assert ws.price(500) == pytest.approx(MARGIN_SPARSE * 9.0, abs=EPS)   # frontier


def test_within_bracket_monotone_step_family():
    ws = LadderWorkspace(FAMILIES["step_flat_then_jump"]["seeds"])
    # Strict bracket interiors: the lower endpoint belongs to the previous zone
    # (floor margin below the first rung, degenerate bracket at a rung) and the
    # upper endpoint is a rung where a down-step is explicitly allowed (rule 4).
    brackets = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
                (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500),
                (4500, 20000), (20000, 60000)]
    for lo, hi in brackets:
        ps = [ws.price(L) for L in range(lo + 1, hi, 97)]
        assert all(b >= a - EPS for a, b in zip(ps, ps[1:])), f"regression in bracket {lo}-{hi}"


# --------------------------------------------------------------------------- rule 5
class TestRule5WarmReject:
    def test_warm_reject_at_exact_boundary(self):
        ws = LadderWorkspace([(100.0, 5.0), (200.0, 7.0)])
        assert ws.observe(100, 6.0, cached_tokens=10) is False      # 0.1*100 == 10
        assert ws.stats["warm_rejected"] == 1
        assert ws.snapshot() == [(100.0, 5.0), (200.0, 7.0)]
        assert ws.observe(100, 6.0, cached_tokens=9) is True        # 9 < 10

    def test_warm_reject_tiny_l_clamps_denominator_to_one(self):
        ws = LadderWorkspace([(100.0, 5.0), (200.0, 7.0)])
        assert ws.observe(1, 6.0, cached_tokens=1) is False          # 1 >= 0.1*max(1,1)
        assert ws.observe(1, 6.0, cached_tokens=0) is True
        assert (1.0, 6.0) in ws.snapshot()

    def test_zero_cached_tokens_never_warm_rejects(self):
        ws = LadderWorkspace([(1000.0, 1.0)])
        assert ws.observe(5000, 2.0, cached_tokens=0) is True


class TestRule5OutlierGate:
    def test_three_spaced_samples_insert_on_the_third(self):
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        clock = FakeClock()
        ws._clock = clock
        assert ws.observe(8000, 100.0) is False                      # streak 1
        clock.advance(60.0)                                          # exactly 60 s: allowed
        assert ws.observe(8000, 100.0) is False                      # streak 2
        clock.advance(60.0)
        assert ws.observe(8000, 100.0) is True                       # streak 3 -> insert
        assert ws.price(8000) == pytest.approx(MARGIN_SPARSE * 100.0, abs=EPS)
        assert ws.stats["outlier_confirmed"] == 1
        assert ws.stats["outlier_rejected"] == 2

    def test_spacing_just_under_60s_blocks_the_streak(self):
        # A too-soon sample is rejected WITHOUT touching the streak: the 60 s
        # spacing is measured from the last sample that was accepted INTO the
        # streak, so a flood of early retries cannot fast-forward the gate.
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        clock = FakeClock()
        ws._clock = clock
        assert ws.observe(8000, 100.0) is False          # t=0: streak 1, anchor 0
        clock.advance(59.999)
        assert ws.observe(8000, 100.0) is False          # too soon (59.999 < 60)
        clock.advance(0.001)
        assert ws.observe(8000, 100.0) is False          # t=60: gap 60 ok -> streak 2
        clock.advance(60.0)
        assert ws.observe(8000, 100.0) is True           # t=120: streak 3 -> insert
        assert ws.price(8000) == pytest.approx(MARGIN_SPARSE * 100.0, abs=EPS)
        assert ws.stats["outlier_confirmed"] == 1
        assert ws.stats["accepted"] == 1

    def test_modest_sample_resets_the_streak(self):
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        clock = FakeClock()
        ws._clock = clock
        assert ws.observe(8000, 100.0) is False                      # streak 1
        clock.advance(120.0)
        assert ws.observe(8000, 1.0) is True                         # modest: inserts, resets
        clock.advance(120.0)
        assert ws.observe(8000, 100.0) is False                      # streak restarted
        clock.advance(120.0)
        assert ws.observe(8000, 100.0) is False                      # streak 2
        clock.advance(120.0)
        assert ws.observe(8000, 100.0) is True                       # 3rd consecutive
        assert ws.price(8000) == pytest.approx(MARGIN_SPARSE * 100.0, abs=EPS)

    def test_streaks_are_isolated_per_8000_bucket(self):
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        clock = FakeClock()
        ws._clock = clock
        assert ws.observe(8000, 100.0) is False                      # bucket 1, streak 1
        clock.advance(120.0)
        assert ws.observe(16000, 100.0) is False                     # bucket 2: own streak
        clock.advance(120.0)
        assert ws.observe(8000, 100.0) is False                      # bucket 1, streak 2
        clock.advance(120.0)
        assert ws.observe(8000, 100.0) is True                       # bucket 1 confirmed
        assert sorted(x for x, _ in ws.snapshot()) == [1000.0, 2000.0, 3000.0, 8000.0]

    def test_peak_exactly_at_2_5x_is_not_an_outlier(self):
        ws = LadderWorkspace([(10000.0, 10.0)])
        assert ws.observe(8000, 25.0) is True                        # 25 > 2.5*10 is false
        assert ws.price(8000) >= 25.0 - EPS

    def test_peak_just_over_2_5x_is_gated(self):
        ws = LadderWorkspace([(10000.0, 10.0)])
        assert ws.observe(8000, 25.000001) is False
        assert ws.snapshot() == [(10000.0, 10.0)]

    def test_extreme_outlier_rapid_flood_never_inserts(self):
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        for _ in range(60):                                          # real clock: all < 60 s apart
            assert ws.observe(80000, 1e12) is False
        assert ws.snapshot() == [(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)]
        assert ws.stats["outlier_rejected"] == 60
        assert ws.stats["accepted"] == 0


class TestRule5PoisonLow:
    def test_duplicate_x_equal_y_is_poison_low(self):
        ws = LadderWorkspace([(100.0, 5.0), (200.0, 7.0)])
        assert ws.observe(100, 5.0) is False
        assert ws.stats["poison_low_ignored"] == 1
        assert ws.snapshot() == [(100.0, 5.0), (200.0, 7.0)]

    def test_duplicate_x_lower_y_is_poison_low(self):
        ws = LadderWorkspace([(100.0, 5.0), (200.0, 7.0)])
        assert ws.observe(100, 4.999) is False
        assert ws.stats["poison_low_ignored"] == 1
        assert ws.snapshot() == [(100.0, 5.0), (200.0, 7.0)]

    def test_poison_low_does_not_block_a_later_higher_y(self):
        ws = LadderWorkspace([(100.0, 5.0), (200.0, 7.0)])
        assert ws.observe(200, 7.0) is False
        assert ws.observe(200, 7.5) is True
        assert (200.0, 7.5) in ws.snapshot()
        assert ws.price(200) >= 7.5 - EPS


class TestRule5Insertion:
    def test_first_rung_into_empty_ladder(self):
        ws = LadderWorkspace([])
        assert ws.observe(5000, 100.0) is True
        assert ws.snapshot() == [(5000.0, 100.0)]
        assert ws.price(4000) == pytest.approx(MARGIN_SPARSE * 100.0, abs=EPS)

    def test_frontier_extension_with_modest_peak_inserts_immediately(self):
        ws = LadderWorkspace([(10000.0, 10.0)])
        assert ws.observe(50000, 12.0) is True                       # 12 <= 2.5*10
        assert ws.price(50000) == pytest.approx(MARGIN_SPARSE * 12.0, abs=EPS)
        assert ws.price(30000) == pytest.approx(MARGIN_SPARSE * 12.0, abs=EPS)

    def test_inserted_rung_raises_prefix_prices(self):
        ws = LadderWorkspace([(1000.0, 1.0), (2000.0, 1.0), (3000.0, 1.0)])
        before = {L: ws.price(L) for L in (4000, 8000, 20000)}
        assert ws.observe(10000, 2.0) is True       # 2.0 <= 2.5x1: no gate
        for L, old in before.items():
            assert ws.price(L) >= old + EPS

    def test_insertion_never_violates_soundness_floor(self):
        ws = LadderWorkspace([(10000.0, 10.0), (40000.0, 10.0)])
        assert ws.observe(30000, 2.0) is True
        snap = ws.snapshot()
        for L in range(10001, 40000, 977):
            assert ws.price(L) >= soundness_floor(snap, L) - 1e-6


class TestRule5CapAndEviction:
    def test_cap_evicts_nearest_x_rung(self):
        ws = LadderWorkspace([(1000.0 * k, 1.0) for k in range(1, 65)])
        assert len(ws.snapshot()) == 64
        assert ws.observe(64500, 1.0) is True                        # nearest is 64000
        xs = [x for x, _ in ws.snapshot()]
        assert len(xs) == 64
        assert 64000.0 not in xs and 63000.0 in xs and 64500.0 in xs
        assert ws.observe(200, 1.0) is True                          # nearest is 1000
        xs = [x for x, _ in ws.snapshot()]
        assert len(xs) == 64
        assert 1000.0 not in xs and 2000.0 in xs and 200.0 in xs

    def test_cap_is_per_side(self):
        ws = LadderWorkspace([(100.0 * k, 1.0) for k in range(1, 65)], switch=10**9)
        for i in range(70):
            assert ws.observe(2 * 10**9 + 1000 * i, 1.0) is True
        snap = ws.snapshot()
        assert len(snap) == 128                                      # 64 pre + 64 post
        assert sum(1 for x, _ in snap if x >= 10**9) == 64
        assert sum(1 for x, _ in snap if x < 10**9) == 64

    def test_cap_holds_under_sustained_inserts(self):
        ws = LadderWorkspace([(1000.0 * k, 1.0) for k in range(1, 65)])
        for i in range(100):
            ws.observe(70000 + 503 * i, 1.0 + i * 1e-9)
        assert len(ws.snapshot()) == 64


def test_adversarial_all_equal_rungs():
    seeds = [(500, 2.0), (1000, 2.0), (2000, 2.0), (4000, 2.0)]
    ws = LadderWorkspace(seeds)
    for L in _grid(range(0, 20001, 7), range(499, 502), range(999, 1002),
                   range(1999, 2002), range(3999, 4002)):
        want = expected_price(seeds, L)
        assert ws.price(L) == pytest.approx(want, abs=1e-6, rel=1e-12), f"price({L})"


def test_adversarial_single_rung_all_regions():
    ws = LadderWorkspace([(500.0, 3.0)], margin=2.0)
    assert ws.price(0) == pytest.approx(2.0 * 3.0, abs=EPS)          # floor below
    assert ws.price(500) == pytest.approx(2.0 * 3.0, abs=EPS)        # floor at the rung
    assert ws.price(501) == pytest.approx(MARGIN_SPARSE * 3.0, abs=EPS)   # frontier (single rung)
    assert ws.observe(500, 3.0) is False                             # poison-low tie
    assert ws.observe(500, 4.0) is True                              # higher y replaces
    assert ws.price(10**6) == pytest.approx(MARGIN_SPARSE * 4.0, abs=EPS)


def test_switch_boundary_prices_from_correct_side():
    ws = LadderWorkspace([(100.0, 1.0), (1500.0, 2.0), (2048.0, 3.0), (5000.0, 8.0)],
                         switch=2048)
    assert ws.price(2047) == pytest.approx(MARGIN_SPARSE * 2.0, abs=EPS)   # pre side
    assert ws.price(2048) == pytest.approx(MARGIN_SPARSE * 3.0, abs=EPS)   # post: first rung
    assert ws.price(2049) == pytest.approx(MARGIN_DENSE * 8.0, abs=EPS)    # post: dense bracket
    assert ws.price(2048) != ws.price(2047)                                # discontinuity at switch


# --------------------------------------------------------------------------- rule 6
def test_note_underprice_increments_counter():
    ws = LadderWorkspace([(1000.0, 1.0)])
    assert ws.stats["underpriced_events"] == 0
    ws.note_underprice(1500, admitted_price=1.2, observed=2.0)
    ws.note_underprice(1600, admitted_price=1.2, observed=3.0)
    ws.note_underprice(1700, admitted_price=1.2, observed=4.0)
    assert ws.stats["underpriced_events"] == 3


def test_note_underprice_thread_safe():
    ws = LadderWorkspace([(1000.0, 1.0)])
    def bump():
        for _ in range(25):
            ws.note_underprice(1500, admitted_price=1.2, observed=2.0)
    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert ws.stats["underpriced_events"] == 100


# --------------------------------------------------------------------------- rule 7
def test_lock_is_reentrant():
    ws = LadderWorkspace([(1000.0, 1.0)])
    lock = getattr(ws, "_lock", None)
    assert lock is not None
    assert isinstance(lock, type(threading.RLock()))
    # Re-enter while holding the lock: an RLock allows it, a plain Lock hangs.
    with lock:
        assert ws.price(1500) == pytest.approx(MARGIN_SPARSE * 1.0, abs=EPS)
        assert ws.snapshot() == [(1000.0, 1.0)]


def test_thread_stress_writers_and_readers():
    ws = LadderWorkspace([(250.0, 1.0), (750.0, 2.0), (9000.0, 4.0), (25000.0, 6.0)],
                         switch=4000)
    errors = []
    stop = threading.Event()

    def writer(k):
        try:
            rng = random.Random(k)
            for _ in range(400):
                L = rng.randrange(200, 40000)
                cached = 0 if rng.random() < 0.9 else 10**9
                ws.observe(L, rng.uniform(3.0, 8.0), cached_tokens=cached)
        except Exception as exc:  # noqa: BLE001
            errors.append(("writer", exc))

    def reader():
        try:
            while not stop.is_set():
                ws.price(2000)
                ws.price(30000)
                ws.snapshot()
        except Exception as exc:  # noqa: BLE001
            errors.append(("reader", exc))

    writers = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(2)]
    for t in writers + readers:
        t.start()
    for t in writers:
        t.join(120)
    stop.set()
    for t in readers:
        t.join(30)
    assert not errors, errors[:3]

    snap = ws.snapshot()
    assert snap == sorted(snap), "final ladder not sorted"
    xs = [x for x, _ in snap]
    assert len(set(xs)) == len(xs), "duplicate x leaked into the ladder"
    pre = [x for x in xs if x < 4000]
    post = [x for x in xs if x >= 4000]
    assert len(pre) <= LADDER_CAP and len(post) <= LADDER_CAP

    # soundness floor over the final learned ladder, on both sides
    for L in range(1, 45000, 331):
        side = [p for p in snap if (p[0] < 4000) == (L < 4000)]
        if not side:
            continue
        assert ws.price(L) >= soundness_floor(side, L) - 1e-6, f"floor broke at {L}"


# --------------------------------------------------------------------------- violations
# Contract-derived tests. VIOLATION 1 (a+b) was found by this suite's first run and
# FIXED (the gate now uses the unmargin'd staircase); they are promoted to real
# assertions. VIOLATION 2 remains open (semantic decision deferred to v1.1) and
# stays xfail(strict) so a partial "fix" fails loudly.
class TestFixedViolations:
    # were xfail(strict) at the independent suite's first run; FIXED (gate now uses
    # the unmargin'd staircase) and promoted to real assertions per the suite's own
    # protocol ("if the module is ever fixed they flip to failures and must be
    # promoted into the main suite").
    def test_gate_threshold_uses_unmargind_price_sparse_bracket(self):
        ws = LadderWorkspace([(10000.0, 10.0), (50000.0, 10.0)], margin=1.15)
        # unmargin'd staircase price at 30000 = 10.0; 30.0 > 2.5*10 -> gated
        assert ws.observe(30000, 30.0, cached_tokens=0) is False
        assert ws.snapshot() == [(10000.0, 10.0), (50000.0, 10.0)]

    def test_gate_accepts_2x_peak_with_default_floor_margin(self):
        ws = LadderWorkspace([(1000.0, 10.0), (2000.0, 10.0)])
        assert ws.observe(1500, 20.0, cached_tokens=0) is True

    @pytest.mark.xfail(strict=True, reason=(
        "VIOLATION 2: inserting a rung at x=30000 (y=2.0) between rungs "
        "(10000,10) and (40000,10) drops price(25000) from 15.0 to 11.5. "
        "Rule 5 says a new rung can never LOWER any price (prefix-max), and "
        "rule 4 allows down-steps only exactly at a measured rung x; 25000 is "
        "not a rung. (The 1.15x soundness floor itself still holds.)"))
    def test_inserted_rung_never_lowers_existing_price(self):
        ws = LadderWorkspace([(10000.0, 10.0), (40000.0, 10.0)])
        before = ws.price(25000)
        assert ws.observe(30000, 2.0, cached_tokens=0) is True
        assert ws.price(25000) >= before - EPS
