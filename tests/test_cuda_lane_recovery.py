"""Two fake ranks when a step fails: admissions undone on every rank, lower tiers that fail, rounds that fail."""

from __future__ import annotations

import pytest
from cuda_lane_fakes import Pattern, drive
from test_cuda_lanes import Pair, disk_tiers, prompts, serial, stream


def test_a_failed_resume_frees_the_lane_on_that_rank(tmp_path):
    p = prompts(1)[0]
    pair = Pair(lanes=1, keep=1)
    pair.run([stream(p, 4)])
    pair.follower.cache.entries.clear()  # rank 1 lost the entry rank 0 resumes from
    with pytest.raises(RuntimeError, match="another rank failed"):  # rank 0 learns it from rank 1's vote
        pair.decoder.admit(stream(p + [1], 4))
    assert pair.decoder.free == [0] and pair.follower.lanes == [None] and pair.follower.tables[0].pages == []
    assert pair.decoder.local.lanes == [None] and pair.decoder.local.tables[0].pages == []


def test_a_lower_tier_state_one_rank_lacks_starts_the_prompt_fresh(tmp_path):
    p = prompts(1)[0]
    Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path)).run([stream(p, 6), stream(prompts(2)[1], 6)])
    for f in (tmp_path / "states").rglob("rank1/*.tfs"):  # one rank's disk lost its entries
        f.unlink()
    pair = Pair(lanes=1, keep=1, tiers=disk_tiers(tmp_path))
    longer = p + [8, 8, 8]
    s = stream(longer, 15)
    assert pair.run([s]) == [serial(longer, 15)] and s.cached == 0
    again = stream(p + [9, 9], 6)  # the same prefix: rank 0 no longer offers the state rank 1 lacks
    assert pair.run([again]) == [serial(p + [9, 9], 6)]
    from tensorfold.cuda.lanes.link import ADMIT

    tried = [m for kind, m in pair.link.sent if kind == ADMIT and m[3] >= 0]
    assert len(tried) == 1


def test_a_failed_admission_on_rank_0_is_undone_on_every_rank(monkeypatch):
    pair = Pair(lanes=1, keep=0)
    real, calls = pair.decoder.local._admit, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("rank 0 fails after the broadcast")
        return real(*a, **k)

    monkeypatch.setattr(pair.decoder.local, "_admit", flaky)
    with pytest.raises(RuntimeError, match="after the broadcast"):
        pair.decoder.admit(stream(prompts(1)[0], 4))
    assert pair.follower.lanes == [None] and pair.follower.tables[0].pages == [] and pair.decoder.free == [0]
    assert pair.run([stream(prompts(1)[0], 4)]) == [serial(prompts(1)[0], 4)]


def test_a_follower_does_not_vote_on_a_round():
    from tensorfold.cuda.lanes.link import ROUND, STOP

    pair = Pair(lanes=1)
    votes = []

    class Script:
        def __init__(self):
            self.msgs = [(ROUND, [7]), (STOP, [])]  # a malformed round: the follower's apply raises

        def recv(self):
            return self.msgs.pop(0)

        def agree(self, ok):
            votes.append(ok)
            return False

    with pytest.raises(RuntimeError):  # out of the loop, as before: no vote for rank 0's next collective to meet
        pair.follower.follow(Script())
    assert votes == []


def test_a_failed_round_fails_every_admitted_request_and_one_rank_goes_on():
    from tensorfold.cuda.lanes.link import LocalLink

    p = prompts(2)
    pair = Pair(lanes=2, prefill_rows=512)
    pair.decoder.link = LocalLink()
    a, b = stream(p[0], 5), stream(p[1], 5)
    pair.decoder.admit(a)
    pair.decoder.admit(b)
    real = pair.decoder.forward.window
    pair.decoder.forward.window = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        pair.decoder.round()
    assert pair.decoder.drop() == [a, b] and pair.decoder.free == [0, 1] and pair.decoder.live() == 0
    pair.decoder.forward.window = real
    s = stream(p[0], 5)
    drive(pair.decoder, [s], lanes=2)
    assert s.out == serial(p[0], 5)


def test_a_failed_round_on_two_ranks_refuses_further_work():
    pair = Pair(lanes=1)
    pair.decoder.admit(stream(prompts(1)[0], 5))
    pair.decoder.drop()
    with pytest.raises(RuntimeError, match="two ranks"):
        pair.decoder.admit(stream(prompts(1)[0], 5))


@pytest.mark.parametrize("where", ["forward.reset", "_resume", "drafter.reset", "depth.reset"])
@pytest.mark.parametrize("rank", [0, 1])
def test_an_admission_failing_anywhere_on_either_rank_leaves_both_as_they_were(where, rank, monkeypatch):
    from tensorfold.cuda.drafting import StaticDepth

    p = prompts(1)[0]
    pair = Pair(lanes=1, keep=1, drafter=Pattern, depth=lambda: StaticDepth(2))
    pair.run([stream(p, 4)])  # a kept state, so a resume runs
    side = pair.decoder.local if rank == 0 else pair.follower
    owner, name = (side, "_resume") if where == "_resume" else (getattr(side, where.split(".")[0]), where.split(".")[1])
    real = getattr(owner, name)

    def fail(*a, **k):
        if where == "_resume":
            real(*a, **k)  # its pages are mapped and written first
        raise RuntimeError(f"{where} fails on rank {rank}")

    monkeypatch.setattr(owner, name, fail)
    def snap(lanes):
        tables, pool = lanes.tables, lanes.pool
        return lanes.lanes, [t.quota for t in tables], [list(t.pages) for t in tables], dict(pool.refs), pool.free

    before = [snap(lanes) for lanes in (pair.decoder.local, pair.follower)]
    with pytest.raises(RuntimeError):
        pair.decoder.admit(stream(p + [1], 4))
    after = [snap(lanes) for lanes in (pair.decoder.local, pair.follower)]
    assert after == before and pair.decoder.free == [0]
    a, b = pair.decoder.local, pair.follower
    assert a.state() == b.state()  # the device cache's order too: EVICT names entries by index
    for plane in a.pool.buffers:
        assert (a.pool.buffers[plane] == b.pool.buffers[plane]).all()
    monkeypatch.setattr(owner, name, real)
    s = stream(p + [1], 4)
    assert pair.run([s]) == [serial(p + [1], 4)]


def test_a_failing_disk_never_stops_serving_and_every_page_comes_back(tmp_path):
    from cuda_lane_fakes import FailingTier

    p = prompts(4)
    pair = Pair(lanes=1, keep=1, tiers=[[FailingTier()], [FailingTier()]])
    outs = pair.run([stream(q, 5) for q in p])  # each new state pushes the last one down, into a failing tier
    assert outs == [serial(q, 5) for q in p]
    cache = pair.decoder.cache
    assert cache.dropped == len(p) - 1 and len(cache.entries) == 1
    pages = pair.decoder.pool.pages - pair.decoder.pool.free
    assert pages == len(cache.entries[0][1].pages)  # only the one kept state holds pages
