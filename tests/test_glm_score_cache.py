"""A decision prefill must not leave its attention rows named as the previous conversation."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from tensorfold.families.glm5_next.cuda.engine import GlmEngine


class _Drafter:
    def __init__(self):
        self.context_end = 40
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.context_end = 0


class _Snap:
    def __init__(self, ids, need):
        self.ids, self.need, self.states = list(ids), need, 5
        self.rows, self.nbytes = None, 0
        self.mtp_len, self.drafter_end = len(self.ids), len(self.ids)


def _engine(monkeypatch, cells, logits):
    def save_rows(e, snap):
        snap.rows, snap.nbytes, snap.drafter_end = list(cells), snap.need, -1

    def prompt_logits(e, prompt):
        cells[:] = ["decision", *prompt]
        return logits

    decode = SimpleNamespace(row_bytes=lambda e, s: s.need, save_rows=save_rows,
                             snapshot_bytes=lambda s: s.states + (s.nbytes if s.rows is not None else 0))
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", decode)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.score",
                        SimpleNamespace(prompt_logits=prompt_logits))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    engine = GlmEngine.__new__(GlmEngine)
    engine.cache, engine.live, engine.cache_entries = [], [], 8
    engine.kept_counts = dict.fromkeys(("hits", "misses", "evictions"), 0)
    engine.e = engine
    engine.comm = None
    engine.rank = 0
    engine.limit = 1000
    engine._share = lambda values: list(values)
    return engine


def test_scoring_saves_the_live_rows_before_the_decision_overwrites_them(monkeypatch):
    cells = ["conversation"]
    engine = _engine(monkeypatch, cells, [0.0, 1.0])
    engine.cache_bytes = 100
    engine.drafter = _Drafter()
    conversation = _Snap(range(40), 40)
    engine.cache, engine.live = [conversation], list(range(40))
    values, logsumexp = engine.score_labels([9, 9], [0, 1])
    assert values == [0.0, 1.0] and logsumexp > 0
    assert cells == ["decision", 9, 9]
    assert conversation.rows == ["conversation"]
    assert conversation in engine.cache and engine.live == []
    assert engine.drafter.context_end == 0 and engine.drafter.resets == 1
    assert engine._resume([9, 9, 1], [0]) is None
    assert engine._resume(list(range(40)) + [7], [0]) is conversation


def test_scoring_drops_a_live_conversation_it_cannot_save(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.cache_bytes = 0
    engine.drafter = _Drafter()
    conversation = _Snap(range(40), 80)
    engine.cache, engine.live = [conversation], list(range(40))
    engine.score_labels([3], [0])
    assert conversation not in engine.cache and conversation.rows is None
    assert engine.live == [] and engine.drafter.context_end == 0
    assert engine._resume(list(range(40)) + [7], [0]) is None


def test_rank_one_scores_through_the_same_release(monkeypatch):
    cells = ["conversation"]
    engine = _engine(monkeypatch, cells, [0.0])
    engine.cache_bytes = 100
    engine.rank = 1
    engine.drafter = None
    conversation = _Snap(range(8), 10)
    engine.cache, engine.live = [conversation], list(range(8))
    assert engine._score_local([4, 5], [0]) == ([], 0.0)
    assert conversation.rows == ["conversation"] and engine.live == []
