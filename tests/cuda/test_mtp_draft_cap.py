"""CPU unit tests for adaptive MTP draft depth (no GPU)."""

from tensorfold.families.qwen4_exp.cuda import draft_cap


def test_draft_cap_zero_room_or_depth():
    assert draft_cap(6, 0, 3.0) == 0
    assert draft_cap(0, 10, 3.0) == 0


def test_draft_cap_floors_at_one_when_accept_is_low():
    assert draft_cap(10, 64, 0.0) == 1
    assert draft_cap(10, 64, 0.4) == 1


def test_draft_cap_tracks_ema_plus_one_up_to_depth():
    assert draft_cap(10, 64, 2.0) == 3
    assert draft_cap(10, 64, 9.0) == 10
    assert draft_cap(6, 64, 9.0) == 6


def test_draft_cap_respects_remaining_room():
    assert draft_cap(10, 2, 8.0) == 2
