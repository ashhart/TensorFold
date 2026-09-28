"""CUDA snapshot points: the second message's start and the last assistant start, spaced by MIN_GAP."""

from tensorfold.cuda.markers import MIN_GAP, snapshot_points

OPEN, ASSISTANT = 900, 901


def _prompt(system: int, user: int, history: int = 0) -> list[int]:
    ids = [OPEN] + [5] * system + [OPEN] + [6] * user
    if history:
        ids += [OPEN, ASSISTANT] + [7] * history + [OPEN] + [6] * user
    return ids + [OPEN, ASSISTANT, 8]


def test_system_block_end_and_last_reply_start():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    ids = _prompt(3 * MIN_GAP, 2 * MIN_GAP)
    assert points(ids) == [3 * MIN_GAP + 1, len(ids) - 3]
    chat = _prompt(3 * MIN_GAP, MIN_GAP, history=2 * MIN_GAP)
    assert points(chat) == [3 * MIN_GAP + 1, len(chat) - 3]


def test_points_too_close_are_dropped():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    assert points(_prompt(10, 3 * MIN_GAP)) == [len(_prompt(10, 3 * MIN_GAP)) - 3]      # system block under MIN_GAP
    ids = _prompt(3 * MIN_GAP, 10)
    assert points(ids) == [3 * MIN_GAP + 1]                                              # reply start too close
    assert points([5] * 4 * MIN_GAP) == []
