"""GLM-5.3's prompts, the same on both servers: earlier turns keep their reasoning, and with thinking off no effort
line and no opened think block."""

from __future__ import annotations

import os
from typing import Any

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
OPENED = "<|assistant|><think>"
CLEAR_THINKING = "TF_GLM_CLEAR_THINKING"


def clear_thinking() -> bool:
    """The template's ``clear_thinking`` when a request leaves it unset: False unless ``TF_GLM_CLEAR_THINKING=1``.

    zai-org's GLM-5.3-Flash template defaults it to false, keeping every assistant turn's reasoning in the prompt
    (its model card: pass ``clear_thinking=true`` for chat). Checkpoints that carry the template's first revision
    (``Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`` among them) clear it unless told otherwise, so the reasoning before the
    last user message renders as ``<think></think>``: each new user message changes the earlier turns' tokens, and
    an agent's whole previous tool loop is prefilled again. Passing false renders those checkpoints as the current
    template does; on the current template it changes nothing."""

    value = os.environ.get(CLEAR_THINKING, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{CLEAR_THINKING}={value!r}: expected 0 or 1")
    return value == "1"


def thinking_off(text: str) -> str:
    """The checkpoint text as the thinking-off template renders it: no effort line, an empty think block."""

    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


class GlmTokenizer:
    """The checkpoint's tokenizer (the Mac's), its chat template read through ``thinking_off`` when thinking is off
    and given ``clear_thinking`` (``clear_thinking()``) when the call leaves it unset."""

    def __init__(self, inner: Any, clear: bool | None = None) -> None:
        self._inner = inner
        self._clear = clear_thinking() if clear is None else clear

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def apply_chat_template(self, messages: Any, *args: Any, tokenize: bool = True, **kwargs: Any) -> Any:
        kwargs.setdefault("clear_thinking", self._clear)
        if kwargs.get("enable_thinking", True) is not False:
            return self._inner.apply_chat_template(messages, *args, tokenize=tokenize, **kwargs)
        text = thinking_off(self._inner.apply_chat_template(messages, *args, tokenize=False, **kwargs))
        return list(self._inner.encode(text, add_special_tokens=False)) if tokenize else text
