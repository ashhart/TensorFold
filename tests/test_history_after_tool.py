"""A prompt that is all history (the template adds no generation suffix, e.g. Gemma 4 continuing its own turn
after a tool result) still gets a history boundary, so a checkpoint lands near its end and the next agentic
step resumes from there instead of re-reading the conversation from the last turn start."""

from __future__ import annotations

import threading
from typing import Any

from tensorfold.server.checkpoints import choose_checkpoints
from tensorfold.server.prompt_blocks import PromptBlocks


class GemmaLikeTokenizer:
    """One id per message; a generation suffix [9, 8] only after a user message, none after a tool result."""

    chat_template = "fake"

    def apply_chat_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **_: Any) -> list[int]:
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
    assert choose_checkpoints(history_len, 0, None, prompt) == [len(prompt) - 1]     # a checkpoint near the end
