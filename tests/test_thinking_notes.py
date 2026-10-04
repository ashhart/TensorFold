"""Thinking's two notes: a startup line when replies think by default, a warning when one ran out of tokens thinking."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tensorfold.server import thinking_notes
from tests.test_think_call import served_post

EOS = 3


def test_the_startup_line_names_the_switch_only_when_the_template_has_one(tmp_path):
    assert thinking_notes.startup(tmp_path, True) is None                     # no template
    (tmp_path / "chat_template.jinja").write_text("{%- if enable_thinking is defined %}x{% endif %}")
    line = thinking_notes.startup(tmp_path, True)
    assert "thinking on" in line and "--no-thinking" in line and '"enable_thinking": false' in line
    assert thinking_notes.startup(tmp_path, False) is None
    other = tmp_path / "other"
    other.mkdir()
    (other / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ enable_thinking }}"}))
    assert thinking_notes.template_thinks(other)
    (other / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    assert not thinking_notes.template_thinks(other)


@pytest.mark.parametrize("finish, thinking, content, calls, warned", [
    ("length", True, "", None, True), ("length", True, "  \n", None, True), ("length", True, "Hi", None, False),
    ("stop", True, "", None, False), ("length", False, "", None, False), ("length", True, "", [{"id": 1}], False)])
def test_the_warning_is_for_a_reply_cut_while_thinking(finish, thinking, content, calls, warned):
    line = thinking_notes.unanswered(finish, thinking, content, calls)
    assert (line is not None) == warned
    if warned:
        assert "max_tokens" in line and "reasoning_content" in line and "--no-thinking" in line


def _app(script: list[int]):
    """A thinking ChatApp whose model writes ``script`` (indexes into PIECES) after a three-token prompt."""

    pytest.importorskip("mlx.core")
    from tensorfold.server.app import ChatApp
    from tests.lane_fakes import FakeEngine, FakeFamily

    class Tokenizer:
        eos_token_ids = {EOS}

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
            return [0, 1, 2]

        def decode(self, ids: list[int], **_: Any) -> str:
            return "".join(PIECES[int(t)] for t in ids)

        def encode(self, text: str, **_: Any) -> list[int]:
            return [PIECES.index(text)] if text in PIECES else []

        def convert_tokens_to_ids(self, token: str) -> int | None:
            return PIECES.index(token) if token in PIECES else None

    class Family(FakeFamily):
        def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
            import mlx.core as mx
            import numpy as np

            history, out = cache[0].rows[0], []
            for token in np.array(inputs).reshape(-1).tolist():
                history.append(int(token))
                out.append(script[min(len(history) - 3, len(script) - 1)])
            return mx.array(out, dtype=mx.float32).reshape(1, -1, 1)

    family = Family()
    return ChatApp(None, Tokenizer(), served_name="fake", lanes=1, max_rows=16, max_draft=4, default_max_tokens=6,
                   checkpoint_slots=0, use_proposer=False, enable_thinking=True,
                   engine_factory=lambda model, **kw: FakeEngine(family, **kw))


PIECES = ["<p>", "<q>", "<a>", "<eos>", "Let me think. ", "</think>", "The answer is 4."]


@pytest.mark.parametrize("script, warned", [([4, 4, 4, 4, 4, 4, 4, 4], True), ([4, 5, 6, EOS], False)])
def test_the_server_warns_once_for_a_reply_that_never_left_its_think_block(capsys, script, warned):
    app = _app(script)
    try:
        status, raw = served_post(app, {"messages": [{"role": "user", "content": "2 + 2?"}], "max_tokens": 6})
        assert status == 200
        choice = json.loads(raw)["choices"][0]
        assert (choice["finish_reason"], bool(choice["message"].get("content"))) == (
            ("length", False) if warned else ("stop", True))
        printed = capsys.readouterr().out
        assert printed.count("reached max_tokens while still thinking") == (1 if warned else 0)
    finally:
        app.close()
