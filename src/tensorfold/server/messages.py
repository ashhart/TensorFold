"""Text chat messages shared by the HTTP and template paths."""

import json
from typing import Any, Callable

from tensorfold.server.errors import RequestError

_MEDIA = ("image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url")


def validate_modalities(body: dict[str, Any], *, allow_images: bool = False) -> None:
    if any(body.get(k) for k in _MEDIA):
        raise RequestError("top-level image, audio and video inputs are unsupported")
    modalities = body.get("modalities")
    if modalities is None:
        return
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(modality != "text" for modality in modalities)
    ):
        raise RequestError(
            "only text output is supported; image, audio and video output are unsupported"
        )


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


def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    late_system: str = "system",
    allow_images: bool = False,
) -> list[dict[str, Any]]:
    """Merge leading instructions as system text and retain later instructions as ``late_system`` so earlier conversation tokens stay unchanged."""

    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    out, instructions = [], []
    image_count = 0
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError("message role must be system, developer, user, assistant or tool")
        if any(message.get(k) for k in _MEDIA):
            raise RequestError("this server accepts text only; image, audio and video inputs are unsupported")
        content = message.get("content")
        if isinstance(content, list):
            text = []
            has_image = False
            for part in content:
                if not isinstance(part, dict):
                    raise RequestError("message content parts must be objects")
                part_type = part.get("type")
                if part_type == "text" and not any(part.get(k) for k in _MEDIA):
                    if not isinstance(part.get("text"), str):
                        raise RequestError("a text content part must contain a text string")
                    text.append(part["text"])
                    continue
                if part_type == "image_url":
                    if not allow_images:
                        raise RequestError(
                            "this server accepts text parts only; image inputs are unsupported"
                        )
                    if role != "user":
                        raise RequestError("image inputs are allowed only in user messages")
                    image_spec = part.get("image_url")
                    if not isinstance(image_spec, dict) or not isinstance(
                        image_spec.get("url"), str
                    ):
                        raise RequestError("an image_url part must contain an image URL string")
                    image_count += 1
                    if image_count > 1:
                        raise RequestError("only one image is supported per request")
                    has_image = True
                    continue
                if part_type in ("audio", "input_audio", "video", "video_url"):
                    prefix = "" if allow_images else "this server accepts text only; "
                    raise RequestError(f"{prefix}audio and video inputs are unsupported")
                raise RequestError(
                    "message content may contain text and one image_url part only"
                )
            if not has_image:
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
    """Copy assistant argument strings into mappings for templates, preserving caller messages."""

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
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    call = {**call, "function": {**fn, "arguments": parsed}}
                    touched = True
            new_calls.append(call)
        if touched:
            out.append({**message, "tool_calls": new_calls})
            changed = True
        else:
            out.append(message)
    return out if changed else messages
