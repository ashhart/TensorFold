"""A lone stream's copy windows: the first at the family's first width, twice as wide after a copy lands whole, back
to the first after one breaks, never past the exact width, and every token the fake target's own serial decode."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneStream  # noqa: E402
from tests.lane_fakes import VOCAB, FakeEngine, FakeFamily, fake_next, fake_serial  # noqa: E402


class BreakAt:
    """Proposes the fake target's own continuation, wrong at the given positions; every proposal is backed."""

    last_match = 1 << 30

    def __init__(self, breaks: set[int]) -> None:
        self.breaks = set(breaks)
        self.observed: list[tuple[int, int]] = []

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        history, out = list(context), []
        for _ in range(max_draft):
            token = fake_next(history)
            if len(history) in self.breaks:
                token = (token + 1) % VOCAB
            out.append(token)
            history.append(token)
        return out

    def observe(self, proposed: int, accepted: int) -> None:
        self.observed.append((proposed, accepted))


class RampFamily(FakeFamily):
    first_copy_rows = 4


def decode(family: FakeFamily, breaks: set[int], limit: int = 90) -> tuple[LaneStream, list[tuple[int, int]]]:
    proposer = BreakAt(breaks)
    stream = LaneStream("a", [1, 2, 3], limit, proposer=proposer)
    engine = FakeEngine(family)
    engine.add_stream(stream)
    engine.run()
    assert stream.emitted == fake_serial([1, 2, 3], limit, set())
    return stream, proposer.observed


def test_copies_double_while_they_land_whole_and_start_over_after_a_break() -> None:
    _, observed = decode(RampFamily(), {40})
    widths = [proposed for proposed, _ in observed]
    assert widths[:3] == [3, 7, 15]                   # 4, 8 and 16 rows, then held at the exact width
    width = 3
    for proposed, accepted in observed:
        assert proposed <= width                      # never wider than earned (the reply's end may cut it)
        width = min(15, 2 * width + 1) if accepted == proposed else 3
    broke = next(i for i, (p, a) in enumerate(observed) if a < p)
    assert widths[broke + 1] == 3                     # the break starts over at the first width


def test_without_a_first_width_copies_take_the_exact_width_at_once() -> None:
    _, observed = decode(FakeFamily(), {40})
    widths = [proposed for proposed, _ in observed]
    assert widths[0] == 15 and widths.count(15) >= len(widths) - 1


class Asked(BreakAt):
    """BreakAt that also records each proposal's width with how many streams were live when it was asked."""

    def __init__(self, engine: FakeEngine) -> None:
        super().__init__(set())
        self.engine = engine
        self.asked: list[tuple[int, int]] = []

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        self.asked.append((max_draft, self.engine.active_count))
        return super().propose(context, max_draft)


def test_copies_ramp_only_while_one_stream_is_live() -> None:
    engine = FakeEngine(RampFamily())
    first, second = Asked(engine), Asked(engine)
    streams = [LaneStream("a", [1, 2, 3], 30, proposer=first), LaneStream("b", [4, 5, 6], 120, proposer=second)]
    for stream in streams:
        engine.add_stream(stream)
    engine.run()
    assert [s.emitted for s in streams] == [fake_serial([1, 2, 3], 30, set()), fake_serial([4, 5, 6], 120, set())]
    assert any(r.streams > 1 for r in engine.round_stats)            # shared rounds ran
    shared = [w for p in (first, second) for w, live in p.asked if live > 1]
    alone = [w for w, live in second.asked if live == 1]
    assert shared and max(shared) <= 3                               # shared rounds keep the first width
    assert alone[0] == 3 and 15 in alone                             # alone, the ramp starts over and widens
