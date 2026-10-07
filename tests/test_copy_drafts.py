"""Copy drafts: the latest earlier copy's continuation, at least COPY_MATCH long, never the needle; the env switch."""

import os

from tensorfold.families.qwen4_exp.cuda.copy_drafts import COPY_MATCH, CopyIndex, enabled

A = list(range(100, 140))                 # 40 distinct tokens
B = list(range(200, 212))                 # 12 more


def test_no_copy_no_chain():
    assert CopyIndex(A).chain(15) == []
    assert CopyIndex(A + A[:COPY_MATCH - 1]).chain(15) == []      # the tail repeats fewer than COPY_MATCH tokens


def test_a_repeated_tail_drafts_its_continuation():
    ctx = A + B + A[:COPY_MATCH]                                    # the tail is A's first 8 tokens again
    assert CopyIndex(ctx).chain(15) == A[COPY_MATCH:COPY_MATCH + 15]
    assert CopyIndex(ctx).chain(40) == A[COPY_MATCH:] + B[:8]       # up to max_nodes, across what followed


def test_a_round_without_room_for_a_chain_gets_none():
    ctx = A + B + A[:COPY_MATCH]
    assert CopyIndex(ctx).chain(COPY_MATCH - 1) == []               # the MTP chain takes such a round
    assert CopyIndex(ctx).chain(COPY_MATCH) == A[COPY_MATCH:2 * COPY_MATCH]


def test_the_latest_copy_wins_and_the_needle_is_not_a_copy():
    first = A[:COPY_MATCH] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    second = A[:COPY_MATCH] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ctx = first + second + A[:COPY_MATCH]
    assert CopyIndex(ctx).chain(10) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # a tail repeated back to back: the earlier copy is followed by the needle, which is drafted (the pattern repeats)
    assert CopyIndex(A[:COPY_MATCH] + A[:COPY_MATCH]).chain(15) == A[:COPY_MATCH]


def test_extend_is_the_same_as_rebuilding():
    ctx = A + B + A[:COPY_MATCH + 4]
    grown = CopyIndex(ctx[:30])
    for t in ctx[30:]:
        grown.extend([t])
    assert grown.chain(15) == CopyIndex(ctx).chain(15) == A[COPY_MATCH + 4:COPY_MATCH + 4 + 15]
    assert grown.ctx == ctx


def test_switch(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_COPY_DRAFTS", raising=False)
    assert enabled() and not enabled(False)
    for off in ("0", "off", "False", " no "):
        monkeypatch.setenv("TENSORFOLD_COPY_DRAFTS", off)
        assert not enabled()
    monkeypatch.setenv("TENSORFOLD_COPY_DRAFTS", "1")
    assert enabled(False)
