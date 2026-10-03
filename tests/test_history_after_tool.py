"""A prompt with no generation suffix still gets a history boundary one token short of its end."""

from __future__ import annotations

import threading
from typing import Any

from tensorfold.server.checkpoints import choose_checkpoints
from tensorfold.server.prompt_blocks import PromptBlocks


class GemmaLikeTokenizer:
    """One id per message; a generation suffix [9, 8] only after a user message, none after a tool result."""

    chat_template = "fake"

    def apply_chat_template(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **_: Any,
    ) -> list[int]:
        ids = [1] + [10 + i for i, _ in enumerate(messages)]
        if add_generation_prompt and messages[-1]["role"] == "user":
            ids += [9, 8]
        return ids


class Renderer(PromptBlocks):
    def __init__(self) -> None:
        self.tokenizer, self.tokenizer_lock = GemmaLikeTokenizer(), threading.Lock()
        self.enable_thinking, self.late_system = False, ""

    def effort_for(self, explicit: str | None) -> str | None:
        return None


def test_a_prompt_ending_in_a_user_message_keeps_its_history_boundary():
    prompt, history_len = Renderer().render([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert prompt == [1, 10, 11, 9, 8] and history_len == 3


def test_a_prompt_continuing_after_a_tool_result_gets_a_boundary_one_short_of_its_end():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                                                      "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c", "content": "ok"}]
    prompt, history_len = Renderer().render(messages)
    assert history_len == len(prompt) - 1
    assert choose_checkpoints(history_len, 0, None, prompt) == [len(prompt) - 1]


def test_a_continued_turn_resumes_from_the_boundary_and_matches_a_fresh_reply() -> None:
    """The next step after a tool result reuses that boundary and matches a fresh decode."""

    from tests.test_lane_server import FakeTokenizer, expected_reply, make_app

    class ContinuationTokenizer(FakeTokenizer):
        """A generation marker only when the last message is from the user."""

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
            add = bool(kwargs.get("add_generation_prompt", True))
            add = add and bool(messages) and messages[-1].get("role") == "user"
            return super().apply_chat_template(messages, **{**kwargs, "add_generation_prompt": add})

    app = make_app(tokenizer=ContinuationTokenizer(), lanes=1, checkpoint_slots=4)
    try:
        call = {"id": "c", "type": "function", "function": {"name": "list", "arguments": "{}"}}
        first = [
            {"role": "user", "content": "list the project"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c", "content": "readme"},
        ]
        prompt, history_len = app.render(first)
        assert history_len == len(prompt) - 1
        reply = app.chat(first, max_tokens=6)
        assert reply["cached_tokens"] == 0
        assert reply["content"] == expected_reply(app, first, 6)[1]
        assert any(len(entry.tokens) == history_len for entry in app.checkpoints._entries)
        nxt = [*first, {"role": "tool", "tool_call_id": "c2", "content": "src"}]
        reply2 = app.chat(nxt, max_tokens=5)
        assert reply2["cached_tokens"] == history_len
        assert reply2["content"] == expected_reply(app, nxt, 5)[1]
        assert app.engine.prefill_calls[-1][1] == history_len
    finally:
        app.close()
