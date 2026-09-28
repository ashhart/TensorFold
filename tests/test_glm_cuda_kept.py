"""GLM's kept conversations on CUDA, without a GPU: saved rows never exceed the budget at any moment, a dropped
snapshot frees its rows at once, and the resume point is never dropped to make room."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda.engine import GlmEngine

pytestmark = pytest.mark.torch


class Snap:
    def __init__(self, ids, need, states=5):
        self.ids, self.need, self.states, self.rows, self.nbytes = list(ids), need, states, None, 0


@pytest.fixture
def kept(monkeypatch):
    """An engine with only its kept-conversation state; save_rows records the saved bytes of every snapshot."""
    made, peaks = [], []

    def save_rows(e, snap):
        snap.rows, snap.nbytes = object(), snap.need
        peaks.append(sum(s.nbytes for s in made if s.rows is not None) + sum(s.states for s in made if s in e.cache))

    decode = SimpleNamespace(row_bytes=lambda e, s: s.need, save_rows=save_rows,
                             snapshot_bytes=lambda s: s.states + (s.nbytes if s.rows is not None else 0))
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", decode)
    engine = GlmEngine.__new__(GlmEngine)
    engine.cache, engine.live, engine.cache_bytes, engine.cache_entries = [], [], 100, 8
    engine.e = engine

    def snap(ids, need, states=5):
        s = Snap(ids, need, states)
        made.append(s)
        return s
    return engine, snap, peaks


def test_a_dropped_snapshot_frees_its_rows_before_the_next_is_saved(kept):
    engine, snap, peaks = kept
    first = snap(range(50), 60)
    follow = snap(range(52), 60)                     # the same conversation's next prompt, whose rows are live
    engine.cache, engine.live = [first, follow], list(range(52))
    engine._take_over([])                             # another conversation takes the caches
    assert first not in engine.cache and first.rows is None and follow.rows is not None
    assert max(peaks) <= engine.cache_bytes           # 0.3.6.1's #54 held both (120 of 100) at the second save


def test_the_resume_point_is_never_dropped_to_save_another_conversation(kept):
    engine, snap, peaks = kept
    hit = snap(range(40), 60)
    hit.rows, hit.nbytes = object(), 60               # saved earlier; the next prompt resumes from it
    live = snap(range(100, 150), 60)
    engine.cache, engine.live = [hit, live], list(range(100, 150))
    engine._take_over(list(range(40)) + [7, 8])
    assert hit in engine.cache and hit.rows is not None
    assert live not in engine.cache and live.rows is None and not peaks    # no room: not saved, dropped


def test_remember_keeps_the_newest_and_frees_what_it_drops(kept):
    engine, snap, _ = kept
    old = snap(range(10), 80)
    old.rows, old.nbytes = object(), 80
    engine.cache = [old]
    new = snap(range(20, 30), 0, states=30)
    engine._remember(new)
    assert engine.cache == [new] and old.rows is None
    again = snap(range(20, 30), 0)
    engine._remember(again)                           # the same prompt again replaces its entry
    assert engine.cache == [again] and new.rows is None
