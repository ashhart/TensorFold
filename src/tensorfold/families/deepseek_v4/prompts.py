"""Chat prompts through DeepSeek's own encoder: the checkpoint's template repeats </think> and cannot render tools."""

from __future__ import annotations

import copy
from typing import Any

from tensorfold.families.deepseek_v4.vendor.encoding_dsv4 import ASSISTANT_SP_TOKEN as ASSISTANT
from tensorfold.families.deepseek_v4.vendor.encoding_dsv4 import encode_messages

EFFORTS = {"max": "max", "high": "high", "xhigh": "max"}


def render(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, thinking: bool = False,
           reasoning_effort: str | None = None) -> str:
    """The prompt text for OpenAI-style messages; tools join the first system message (one is added if needed)."""

    msgs = [copy.deepcopy(m) for m in messages]
    for m in msgs:
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
    if tools:
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    mode = "thinking" if thinking else "chat"
    effort = EFFORTS.get(str(reasoning_effort)) if thinking and reasoning_effort else None
    return encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)


class DeepSeekTokenizer:
    """The checkpoint's tokenizer with ``apply_chat_template`` rendered by DeepSeek's encoder."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def message_markers(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Replies start at <｜Assistant｜>, an added token the checkpoint leaves unmarked; no message opener."""

        return (), tuple(int(i) for i in self._inner.encode(ASSISTANT, add_special_tokens=False))

    def apply_chat_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                            add_generation_prompt: bool = True, tokenize: bool = True, enable_thinking: bool = False,
                            thinking_mode: str | None = None, reasoning_effort: str | None = None,
                            **_: Any) -> Any:
        """The encoder adds the reply prefix after a last user turn itself; without a generation prompt it goes."""

        thinking = thinking_mode == "thinking" if thinking_mode is not None else bool(enable_thinking)
        text = render(messages, tools=tools, thinking=thinking, reasoning_effort=reasoning_effort)
        prefix = ASSISTANT + ("<think>" if thinking else "</think>")
        last = messages[-1].get("role") if messages else None
        if add_generation_prompt and last == "assistant":
            text += prefix                                                   # as the checkpoint's template does
        elif not add_generation_prompt and last != "assistant" and text.endswith(prefix):
            text = text[:-len(prefix)]                   # the history alone, so its reply starts a prompt chunk
        if not tokenize:
            return text
        return list(self._inner.encode(text, add_special_tokens=False))
