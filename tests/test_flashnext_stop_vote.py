"""Both Flash Next ranks stop a reply after the same round once rank 0's caller asks (the client left, a stop string),
as GLM's ranks do with #194's vote."""

from __future__ import annotations

import threading

from tensorfold.families.qwen4_exp.cuda.engine import StopVote


def _two_ranks():
    """Two ranks' gathers in lockstep: each returns both ranks' values, rank 0's first."""

    barrier, votes = threading.Barrier(2), [None, None]

    def gather(rank):
        def run(values):
            votes[rank] = list(values)
            barrier.wait()
            both = [list(votes[0]), list(votes[1])]
            barrier.wait()
            return both
        return run

    return gather(0), gather(1)


def test_both_ranks_stop_after_the_round_rank_zero_asks():
    g0, g1 = _two_ranks()
    asks = iter([False, None, False, True, False])     # the caller asks at the fourth round, then answers anything
    seen = {0: [], 1: []}

    def run(rank, vote):
        for round_ in range(5):
            if vote([round_]):
                seen[rank].append(round_)
                return

    ranks = [StopVote(lambda tokens: next(asks), g0), StopVote(lambda tokens: None, g1)]
    threads = [threading.Thread(target=run, args=(r, v)) for r, v in enumerate(ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen == {0: [3], 1: [3]}


def test_a_vote_stays_asked():
    asks = iter([True, False])
    vote = StopVote(lambda tokens: next(asks), lambda values: [values, [0]])
    assert vote([1]) and vote([2])
