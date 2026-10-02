"""Text chat messages shared by the HTTP and template paths."""

import json
from typing import Any, Callable

from tensorfold.server.errors import RequestError

_MEDIA = ("image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url")


def validate_modalities(body: dict[str, Any]) -> None:
    if any(body.get(k) for k in _MEDIA) or body.get("modalities") not in (None, ["text"]):
        raise RequestError("this server accepts and produces text only; image, audio and video are unsupported")


_PROBE = "tensorfold-late-system-probe"


def late_system_role(render: Callable[[list[dict[str, Any]]], Any]) -> str:
    """``system`` when ``render`` (a chat template, as text) keeps a later system message in place, else ``user``."""

    probe = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
             {"role": "assistant", "content": "a"}, {"role": "system", "content": _PROBE},
             {"role": "user", "content": "v"}]
    try:
        return "system" if _PROBE in str(render(probe)) else "user"
    except Exception:  # noqa: BLE001 - a template that rejects the probe cannot render later system messages
        return "user"


_VISUAL = ("image_url", "image", "video_url", "video")      # a video part reaches here only where videos are on


def normalize_messages(messages: list[dict[str, Any]], *, late_system: str = "system",
                       allow_images: bool = False) -> list[dict[str, Any]]:
    """Merge leading instructions as system text and retain later instructions as ``late_system`` so earlier conversation tokens stay unchanged."""

    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out, instructions = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError("message role must be system, developer, user, assistant or tool")
        if any(message.get(k) for k in _MEDIA):
            raise RequestError("this server accepts text only; image, audio and video inputs are unsupported")
        content = message.get("content")
        if isinstance(content, list) and allow_images and any(
                isinstance(p, dict) and p.get("type") in _VISUAL for p in content):
            if role != "user":
                raise RequestError("images and videos are supported only in user messages")
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("text", *_VISUAL):
                    raise RequestError("image messages may contain text and image_url (or video_url) parts only")
                if part["type"] == "text" and not isinstance(part.get("text"), str):
                    raise RequestError("a text content part must contain a text string")
            out.append({**message, "content": list(content)})
            continue
        if isinstance(content, list):
            text = []
            for part in content:
                if (not isinstance(part, dict) or part.get("type") != "text"
                        or any(part.get(k) for k in _MEDIA)):
                    raise RequestError("this server accepts text parts only; image, audio and video inputs are unsupported")
                if not isinstance(part.get("text"), str):
                    raise RequestError("a text content part must contain a text string")
                text.append(part["text"])
            content = "".join(text)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            raise RequestError("message content must be text or an array of text parts")
        item = message if content is message.get("content") else {**message, "content": content}
        if role in ("system", "developer"):
            if not out:
                instructions.append(item)
                continue
            if role != late_system:
                item = {**item, "role": late_system}
        out.append(item)
    if instructions:
        out.insert(0, {**instructions[0], "role": "system",
                       "content": "\n\n".join(m["content"] for m in instructions)})
    return messages if out == messages else out


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy tool arguments into mappings for templates; invalid ones go under ``_invalid_arguments``, not a crash."""

    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    changed = False
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not calls:
            out.append(message)
            continue
        new_calls = []
        touched = False
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(fn, dict) and "arguments" in fn and not isinstance(args, dict):
                parsed = args
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except (ValueError, TypeError):
                        parsed = None
                if not isinstance(parsed, dict):
                    parsed = {"_invalid_arguments": args}
                call = {**call, "function": {**fn, "arguments": parsed}}
                touched = True
            new_calls.append(call)
        if touched:
            out.append({**message, "tool_calls": new_calls})
            changed = True
        else:
            out.append(message)
    return out if changed else messages
