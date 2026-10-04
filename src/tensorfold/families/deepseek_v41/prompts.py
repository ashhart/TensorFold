"""DeepSeek-V4.1 chat prompts through DeepSeek's own encoder: the checkpoint ships no Jinja template."""

from __future__ import annotations

from typing import Any

from tensorfold.families.deepseek_v41.vendor import encoding_dsv41 as enc
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import thinking_switch

EFFORTS = frozenset(enc.REASONING_EFFORT_MAPPINGS)   # low, high and max: the names TF's effort ladder maps onto
_ALIASES = {"xhigh": "max"}                          # TF keeps xhigh; V4.1's highest name is max
ROLES = ("system", "developer", "user", "assistant", "tool", "latest_reminder")
OPEN = enc.ASSISTANT_SP_TOKEN + enc.thinking_start_token
CLOSE = enc.ASSISTANT_SP_TOKEN + enc.thinking_end_token


def _has_image(blocks: Any) -> bool:
    """Whether content blocks (a tool result's included) hold an image the encoder would render as a placeholder."""

    return isinstance(blocks, list) and any(
        isinstance(b, dict) and (b.get("type") in ("image", "image_url")
                                 or (b.get("type") == "tool_result" and _has_image(b.get("content"))))
        for b in blocks)


def render(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, thinking: bool = False,
           reasoning_effort: str | int | None = None, add_generation_prompt: bool = True,
           drop_thinking: bool = True) -> str:
    """The prompt text for OpenAI-style messages; tools join the first system message (an empty one leads if none)."""

    msgs = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ROLES:
            raise ValueError(f"message role must be one of {', '.join(ROLES)}")
        if _has_image(m.get("content")) or _has_image(m.get("content_blocks")):
            raise ValueError("image input requires a supported vision checkpoint")
        m = {**m, "role": "system" if role == "developer" else role}
        if "reasoning_content" not in m and "reasoning" in m:
            m["reasoning_content"] = m.pop("reasoning")
        msgs.append(m)
    if tools:
        system = next((m for m in msgs if m["role"] == "system"), None)
        if system is None:
            msgs.insert(0, system := {"role": "system", "content": ""})
        system["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    effort = _ALIASES.get(reasoning_effort, reasoning_effort) if thinking else None    # checked in chat mode too
    text = enc.encode_messages(msgs, thinking_mode="thinking" if thinking else "chat", drop_thinking=drop_thinking,
                               reasoning_effort=effort)
    prefix = OPEN if thinking else CLOSE
    last = msgs[-1]["role"] if msgs else None
    if add_generation_prompt and last == "assistant":
        text += prefix                                   # a reply after a finished one, as V4's template does
    elif not add_generation_prompt and last != "assistant" and text.endswith(prefix):
        text = text[:-len(prefix)]                       # the history alone, so its reply starts a prompt chunk
    return text


class Template:
    """The CUDA App's template: the efforts it names, and render() raising TemplateError (HTTP 400) on a refusal."""

    efforts = EFFORTS

    def __init__(self, model_dir: Any = None) -> None:
        pass

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, enable_thinking: bool,
               extra: dict[str, Any] | None = None, allow_images: bool = False) -> str:
        from jinja2.exceptions import TemplateError

        extra = extra or {}
        effort = extra.get("reasoning_budget", extra.get("reasoning_effort"))   # an integer budget wins over a name
        try:
            return render(messages, tools=tools, thinking=enable_thinking, reasoning_effort=effort,
                          add_generation_prompt=bool(extra.get("add_generation_prompt", True)),
                          drop_thinking=bool(extra.get("drop_thinking", True)))
        except (AssertionError, ValueError, KeyError, TypeError, NotImplementedError) as exc:
            raise TemplateError(str(exc) or type(exc).__name__) from exc


def numeric_effort(body: dict[str, Any]) -> dict[str, Any]:
    """A request whose reasoning_effort is an integer 1..100 (V4.1's budget), rewritten for the shared reader."""

    kwargs = body.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        return body                                      # the shared path refuses it
    value = body.get("reasoning_effort")
    if value is None and kwargs:
        value = kwargs.get("reasoning_effort")
    if type(value) is not int:
        return body                                      # names (and bools) go through the shared reader
    if not 1 <= value <= 100:
        raise RequestError("reasoning_effort must be none, minimal, low, medium, high, xhigh, max or 1..100")
    kw = {k: v for k, v in (kwargs or {}).items() if k != "reasoning_effort"}
    kw["reasoning_budget"] = value
    if "enable_thinking" not in kw and thinking_switch(kw.get("thinking")) is None:
        kw["enable_thinking"] = True                     # an effort turns thinking on, as a named one does
    return {**{k: v for k, v in body.items() if k != "reasoning_effort"}, "chat_template_kwargs": kw}
