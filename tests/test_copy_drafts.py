"""Copy drafts propose what followed the latest earlier occurrence of the context's last tokens."""

import pytest

from tensorfold.cuda.copy_drafts import CopyDrafts, CopySettings

S = CopySettings(match=3, most=4)


def test_no_match_no_drafts():
    assert CopyDrafts([1, 2, 3, 4, 5, 6], S).propose() == []


def test_latest_occurrence_with_room():
    ctx = [9, 1, 2, 3, 10, 11, 12, 13, 7, 1, 2, 3, 20, 21, 22, 23, 8, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [20, 21, 22, 23]
    assert CopyDrafts(ctx, S).propose(room=2) == [20, 21]


def test_short_tail_takes_the_longest_continuation():
    ctx = [1, 2, 3, 10, 11, 12, 13, 14, 5, 1, 2, 3, 30, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [30, 1, 2, 3]               # both have 4 after them: the latest wins
    ctx = [1, 2, 3, 10, 11, 1, 2, 3]
    assert CopyDrafts(ctx, S).propose() == [10, 11, 1, 2]


def test_extend_and_truncate():
    c = CopyDrafts([5, 1, 2, 3, 6, 7, 8, 9], S)
    c.extend([1, 2])
    assert c.propose() == []
    c.extend([3])
    assert c.propose() == [6, 7, 8, 9]
    c.truncate(9)
    assert len(c) == 9 and c.propose() == []


def test_settings_from_env():
    assert CopySettings.from_env(5, env={"TF_COPY_DRAFTS": "0"}) is None
    assert CopySettings.from_env(5, env={}).most == 5
    s = CopySettings.from_env(5, env={"TF_COPY_MAX": "15"})
    assert s.match == 8 and s.most == 5
    with pytest.raises(ValueError):
        CopySettings.from_env(5, env={"TF_COPY_DRAFTS": "1", "TF_COPY_MATCH": "1"})
