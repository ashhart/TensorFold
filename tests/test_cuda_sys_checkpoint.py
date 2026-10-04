"""System-block checkpoints (TF_SYS_CHECKPOINT): a kept state at the last prompt-pass boundary before a system
block's end, so a block that differs only near its end (a date line, a memory section) resumes there."""

import pytest

from tensorfold.cuda import markers
from tensorfold.cuda.markers import MIN_GAP, snapshot_points
from tensorfold.families.qwen4_exp.cuda import prefixes

OPEN, ASSISTANT = 900, 901


def _prompt(system: int, user: int, tail: int = 0, tail_token: int = 9) -> list[int]:
    return [OPEN] + [5] * system + [tail_token] * tail + [OPEN] + [6] * user + [OPEN, ASSISTANT, 8]


def test_the_checkpoint_is_the_last_granule_at_least_min_gap_before_the_block_end():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT), checkpoint=2048)
    ids = _prompt(13123, 600)                        # the second message starts at 13124
    assert points.checkpoint(ids) == 12288
    assert points(ids) == [12288, 13124, len(ids) - 3]
    near = _prompt(12288 + MIN_GAP - 2, 600)         # block end 12288 + 255: too close, the granule before it
    assert points.checkpoint(near) == 10240
    assert points.checkpoint(_prompt(2000, 600)) is None         # shorter than a granule + MIN_GAP: none
    assert points.checkpoint([5] * 30000) is None                # no second message
    assert snapshot_points((OPEN,), (OPEN, ASSISTANT), checkpoint=4096).checkpoint(ids) == 12288


def test_without_a_granule_the_points_are_unchanged():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT))
    ids = _prompt(13123, 600)
    assert points.checkpoint(ids) is None and points(ids) == [13124, len(ids) - 3]


def test_the_granule_comes_from_tf_sys_checkpoint(monkeypatch):
    monkeypatch.delenv("TF_SYS_CHECKPOINT", raising=False)
    assert markers.checkpoint_rows() == 2048
    monkeypatch.setenv("TF_SYS_CHECKPOINT", "0")
    assert markers.checkpoint_rows() == 0
    monkeypatch.setenv("TF_SYS_CHECKPOINT", "4096")
    assert markers.checkpoint_rows() == 4096
    for bad in ("100", "-1", "lots"):
        monkeypatch.setenv("TF_SYS_CHECKPOINT", bad)
        with pytest.raises(ValueError):
            markers.checkpoint_rows()


class Slot:
    def __init__(self, name):
        self.name, self.copied = name, None

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)


class Owner:
    def __init__(self, keep, free=()):
        self.kept, self.free, self.keep, self.depth = [], list(free), keep, 3

    def _busy(self):
        return set()

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, **kwargs):
        return True

    def _shrink(self, st, **kwargs):
        pass


def test_a_block_with_a_new_tail_resumes_from_the_checkpoint_on_a_copy():
    points = snapshot_points((OPEN,), (OPEN, ASSISTANT), checkpoint=2048)
    a, b = _prompt(12600, 300, tail=500), _prompt(12600, 300, tail=500, tail_token=7)
    cp = points.checkpoint(a)
    assert cp == points.checkpoint(b) == 12288 and a[:cp] == b[:cp] and a[:points(a)[1]] != b[:points(b)[1]]
    owner, slot, spare = Owner(keep=8), Slot("a"), Slot("spare")
    for p in points(a):                                  # the kept states a's prefill leaves, as the engine keeps them
        prefixes.remember(owner, a[:p], slot, {"pos": p, "mtp_len": p - 1}, p)
    owner.free = [spare]
    st, resume, cached = prefixes.slot_for(owner, b, True)
    assert cached == cp and resume["tail"] == cp and st is spare and spare.copied == (slot, cp, cp - 1)
    assert [len(k[0]) for k in owner.kept] == points(a)  # a's own states stay for a's next turn
