"""GLM-5.3's prompts with thinking off, the same on both servers: no effort line, no opened think block."""

from __future__ import annotations

from typing import Any

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
OPENED = "<|assistant|><think>"


def thinking_off(text: str) -> str:
    """The checkpoint text as the thinking-off template renders it: no effort line, an empty think block."""

    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


class GlmTokenizer:
    """The checkpoint's tokenizer (the Mac's), its chat template read through ``thinking_off`` when thinking is off."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def apply_chat_template(self, messages: Any, *args: Any, tokenize: bool = True, **kwargs: Any) -> Any:
        if kwargs.get("enable_thinking", True) is not False:
            return self._inner.apply_chat_template(messages, *args, tokenize=tokenize, **kwargs)
        text = thinking_off(self._inner.apply_chat_template(messages, *args, tokenize=False, **kwargs))
        return list(self._inner.encode(text, add_special_tokens=False)) if tokenize else text
