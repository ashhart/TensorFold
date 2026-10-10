"""A decision prefill must not leave its attention rows named as the previous conversation."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from tensorfold.families.glm5_next.cuda.engine import GlmEngine, _user_token_id


class _Drafter:
    def __init__(self):
        self.context_end = 40
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.context_end = 0


class _Snap:
    def __init__(self, ids, need, *, rewind=False):
        self.ids, self.need, self.states = list(ids), need, 5
        self.rows, self.nbytes = None, 0
        self.mtp_len, self.drafter_end = len(self.ids), len(self.ids)
        self.rewind = rewind


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


def test_rewind_survives_repeated_replacements_and_short_replies(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.cache_bytes = 100
    engine.user_token = 900
    history = [1, 2, 3]
    first = history + [900, 10, 901, 20]
    rewind_at = engine._rewind_point(first)
    assert rewind_at == len(history)
    rewind = _Snap(first[:rewind_at], 20, rewind=True)
    first_end = _Snap(first[:-1], 20)
    engine._remember(rewind)
    engine._remember(first_end)
    assert engine._resume(first + [42, 900, 30], [0]) is first_end
    for message in (11, 12):
        replacement = history + [900, message, 901, 20]
        assert engine._resume(replacement, [0]) is rewind
        engine._remember(_Snap(replacement[:-1], 20))
    assert engine._resume(history + [900, 13, 901, 20], [0]) is rewind
    assert engine._resume([8, 2, 3, 900, 13, 901, 20], [0]) is None
    assert engine._rewind_point(history + [900, 14, 901]) == len(history)


def test_rewind_takes_priority_within_entry_and_byte_limits(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.cache_entries = 1
    engine.cache_bytes = 10
    rewind = _Snap([1, 2], 5, rewind=True)
    engine._remember(rewind)
    end = _Snap([1, 2, 3], 5)
    engine._remember(end)
    assert engine.cache == [end]

    engine.cache_entries = 2
    engine._remember(rewind)
    engine._remember(end)
    assert engine.cache == [rewind, end]
    next_end = _Snap([1, 2, 4], 5)
    engine._remember(next_end)
    assert engine.cache == [rewind, next_end]
    newer = _Snap([1, 2, 3, 4], 5, rewind=True)
    engine._remember(newer)
    assert engine.cache == [rewind, newer]
    latest_end = _Snap([1, 2, 3, 4, 7], 5)
    engine._remember(latest_end)
    assert engine.cache == [newer, latest_end]

    engine.cache_entries = 8
    engine._remember(_Snap([1, 2, 3, 4, 5], 5))
    engine._remember(_Snap([1, 2, 3, 4, 5, 6], 5))
    assert newer in engine.cache and engine._held_bytes() <= engine.cache_bytes


def test_rewind_rows_survive_when_an_old_prefix_must_be_evicted(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.cache_bytes = 45
    rewind = _Snap([1, 2], 40, rewind=True)
    ordinary = _Snap([1, 2, 3], 40)
    engine.cache, engine.live = [rewind, ordinary], [1, 2, 3]
    engine._take_over([])
    assert engine.cache == [rewind] and rewind.rows == ["conversation"]
    assert engine._held_bytes() == engine.cache_bytes


def test_rewind_point_uses_only_the_configured_user_token(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.user_token = 900
    assert engine._rewind_point([900, 1, 901, 2]) is None
    assert engine._rewind_point([901, 2, 900, 3, 901]) == 2
    assert engine._rewind_point([901, 2, 900]) is None


def test_rewind_token_comes_from_added_special_tokens(tmp_path):
    from tokenizers import Tokenizer, models

    tok = Tokenizer(models.WordLevel({"x": 0}, unk_token="x"))
    user = tok.add_special_tokens(["<|user|>"])
    assert user == 1
    tok.save(str(tmp_path / "tokenizer.json"))
    assert _user_token_id(tmp_path) == tok.token_to_id("<|user|>")


def test_rewind_point_is_sent_to_the_other_rank(monkeypatch):
    engine = _engine(monkeypatch, ["conversation"], [0.0])
    engine.user_token = 900
    engine.policy = "0"
    engine.serial_only = False
    engine.request = SimpleNamespace(policy=None, stop_eos=True)
    engine._effective = lambda code: code
    engine._ring = lambda: None
    sent = []
    engine._share = lambda values: sent.append(values) or values
    seen = []
    engine._run = lambda *args: seen.append(args[-1]) or {}
    prompt = [1, 2, 900, 3, 901, 4]
    engine.generate(prompt, 1, None, lambda new: None)
    assert sent[0][15] == 2 and seen == [2]

    calls = iter((None, StopIteration))

    def bell():
        if next(calls) is StopIteration:
            raise StopIteration

    engine._await_bell = bell
    incoming = iter(sent)
    engine._share = lambda values: next(incoming)
    engine.rank = 1
    import pytest

    with pytest.raises(StopIteration):
        engine.follow()
    assert seen == [2, 2]
