"""With thinking off, Gemma channel markup stays out of content. Harmony keeps its own parser."""

# Streamed and finished replies both go through ChatApp. A channel closer selects split_thinking.

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.test_think_call import events, served_post

EOS = 3


def _app(pieces: list[str], script: list[int]):
    """A ChatApp whose model writes ``script`` (indexes into ``pieces``) after a three-token prompt, thinking off."""

    pytest.importorskip("mlx.core")
    from tensorfold.server.app import ChatApp
    from tests.lane_fakes import FakeEngine, FakeFamily

    class Tokenizer:
        eos_token_ids = {EOS}

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
            return [0, 1, 2]

        def decode(self, ids: list[int], **_: Any) -> str:
            return "".join(pieces[int(t)] for t in ids)

        def encode(self, text: str, **_: Any) -> list[int]:
            return [pieces.index(text)] if text in pieces else []

        def convert_tokens_to_ids(self, token: str) -> int | None:
            return pieces.index(token) if token in pieces else None

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
    return ChatApp(None, Tokenizer(), served_name="fake", lanes=1, max_rows=16, max_draft=4, default_max_tokens=32,
                   checkpoint_slots=0, use_proposer=False, enable_thinking=False,
                   engine_factory=lambda model, **kw: FakeEngine(family, **kw))


def _reply(app: Any) -> tuple[dict[str, Any], str, str]:
    """(non-streamed message, streamed content, streamed reasoning) for one user turn."""

    body = {"messages": [{"role": "user", "content": "List the files."}]}
    status, raw = served_post(app, body)
    assert status == 200
    message = json.loads(raw)["choices"][0]["message"]
    status, text = served_post(app, {**body, "stream": True})
    assert status == 200
    deltas = [c["choices"][0]["delta"] for c in events(text) if c.get("choices")]
    return (message, "".join(d.get("content") or "" for d in deltas),
            "".join(d.get("reasoning_content") or "" for d in deltas))


GEMMA = ["<p>", "<q>", "<a>", "<turn|>", "<|channel>", "thought\n", "<channel|>", "The files are a.py and b.py."]


def test_gemma_thought_channel_with_thinking_off_never_reaches_content():
    app = _app(GEMMA, [4, 5, 6, 7, EOS])
    try:
        from tensorfold.server.text import CHANNEL_MARKERS

        assert app.think_markers == CHANNEL_MARKERS
        message, streamed, streamed_reasoning = _reply(app)
        assert message["content"] == "The files are a.py and b.py."
        assert not message.get("reasoning_content")                    # the block was empty
        assert streamed == "The files are a.py and b.py." and streamed_reasoning == ""
    finally:
        app.close()


HARMONY = ["<p>", "<q>", "<a>", "<|return|>", "<|channel|>", "analysis", "<|message|>", "Think it over.", "<|end|>",
           "<|start|>", "assistant", "final", "Hello there."]
STREAMED_HARMONY = ("Hello there.", "Think it over.")     # what v0.6.0 streams for it, measured before the change


def test_harmony_reply_with_thinking_off_is_unchanged():
    """gpt-oss replies with thinking off come out exactly as before the Gemma change: parse_harmony_output."""

    from tensorfold.server.text import CHANNEL_MARKERS, parse_harmony_output

    script = [4, 5, 6, 7, 8, 9, 10, 4, 11, 6, 12, EOS]
    app = _app(HARMONY, script)
    try:
        assert app.think_markers != CHANNEL_MARKERS                     # no <channel|> token: the Harmony path
        text = "".join(HARMONY[t] for t in script[:-1])
        assert parse_harmony_output(text) == ("Hello there.", "Think it over.")
        message, streamed, streamed_reasoning = _reply(app)
        assert (message["content"], message.get("reasoning_content")) == ("Hello there.", "Think it over.")
        assert (streamed, streamed_reasoning) == STREAMED_HARMONY
    finally:
        app.close()


def test_plain_reply_without_channels_is_unchanged_for_both_families():
    for pieces in (GEMMA, HARMONY):
        app = _app(pieces + ["Just an answer."], [len(pieces), EOS])
        try:
            message, streamed, _ = _reply(app)
            assert message["content"] == "Just an answer." and streamed == "Just an answer."
            assert not message.get("reasoning_content")
        finally:
            app.close()

