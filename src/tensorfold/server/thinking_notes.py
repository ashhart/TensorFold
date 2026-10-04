"""The server's two thinking notes: thinking is on at startup, and a reply that ran out of tokens while thinking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OFF = 'chat_template_kwargs {"enable_thinking": false}'


def template_thinks(model_dir: Path) -> bool:
    """Whether the checkpoint's chat template reads ``enable_thinking`` (its default decides a silent request)."""

    template = Path(model_dir) / "chat_template.jinja"
    if template.is_file() and "enable_thinking" in template.read_text(errors="ignore"):
        return True
    config = Path(model_dir) / "tokenizer_config.json"
    try:
        return "enable_thinking" in json.dumps(json.loads(config.read_text()).get("chat_template") or "")
    except (OSError, ValueError):
        return False


def startup(model_dir: Path, thinking: bool) -> str | None:
    """The startup line when replies think by default, or None."""

    if not thinking or not template_thinks(model_dir):
        return None
    return ("[tensorfold] thinking on (the chat template's default): replies reason in reasoning_content before the "
            f"answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send {OFF}")


def unanswered(finish: str, thinking: bool, content: Any, calls: Any = None) -> str | None:
    """A warning when a reply ended at max_tokens with nothing visible: its tokens all went to thinking."""

    if finish != "length" or not thinking or calls or str(content or "").strip():
        return None
    return ("[tensorfold] warning: a reply reached max_tokens while still thinking, so its content is empty and its "
            f"text is all in reasoning_content; raise max_tokens, or send {OFF} (server: --no-thinking)")


__all__ = ["startup", "template_thinks", "unanswered"]
