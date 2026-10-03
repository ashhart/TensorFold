"""vLLM's ``/tokenize`` and ``/detokenize`` on the Mac server: the ids the chat and completion routes would run.

``POST /tokenize`` takes ``prompt`` (``add_special_tokens``, default true, as vLLM's) or ``messages`` with the chat
fields that shape the prompt (``tools``, ``reasoning_effort``, ``chat_template_kwargs``) and ``add_generation_prompt``
(default true); it answers ``count``, ``max_model_len`` and ``tokens``, plus ``token_strs`` with
``return_token_strs``. ``POST /detokenize`` turns ``tokens`` back into ``prompt``, special tokens included. The CUDA
server answers the same fields (``cuda.server.App.tokenize``)."""

from __future__ import annotations

from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.messages import normalize_messages
from tensorfold.server.prompts import has_images, prepare_prompt
from tensorfold.server.request_options import thinking_fields
from tensorfold.server.text import render_prompt_ids
from tensorfold.server.tools import active_tool_specs

ROUTES = ("/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize")


def flag(body: dict[str, Any], name: str, default: bool) -> bool:
    """A boolean request field; RequestError when it is there and not a boolean."""

    value = body.get(name, default)
    if not isinstance(value, bool):
        raise RequestError(f"{name} must be a boolean")
    return value


def _vocab(tokenizer: Any) -> int | None:
    """The vocabulary's size, added tokens included, through the wrappers families put around a tokenizer (GLM's and
    DeepSeek's ``_inner``, mlx-lm's ``_tokenizer``); None when none of them tells."""

    for _ in range(4):
        if hasattr(type(tokenizer), "__len__"):
            return len(tokenizer)
        if hasattr(type(tokenizer), "get_vocab_size"):              # a tokenizers.Tokenizer
            return int(tokenizer.get_vocab_size(with_added_tokens=True))
        attributes = getattr(tokenizer, "__dict__", {})
        tokenizer = next((attributes[k] for k in ("_inner", "inner", "_tokenizer") if k in attributes), None)
        if tokenizer is None:
            return None
    return None


def token_ids(value: Any, vocab: int | None, field: str = "tokens") -> list[int]:
    """Token ids as a request gives them (a list, or a list holding one list); RequestError outside the vocabulary."""

    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(t) is not int for t in value):
        raise RequestError(f"{field} must be a list of integer token ids (one prompt a request)")
    if any(t < 0 or (vocab is not None and t >= vocab) for t in value):
        top = "" if vocab is None else f" to {vocab - 1}"
        raise RequestError(f"{field} token ids must be in the vocabulary's range 0{top}")
    return list(value)


def tokenize(app: Any, body: Any) -> dict[str, Any]:
    """A prompt's or a chat request's ids, rendered as ``ChatApp.chat`` renders them."""

    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    strings = flag(body, "return_token_strs", False)
    if "messages" in body:
        generation = flag(body, "add_generation_prompt", True)
        messages = normalize_messages(body["messages"], allow_images=getattr(app, "vision", None) is not None)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        fields = thinking_fields(body, getattr(app, "effort_levels", frozenset()))
        requested = fields.get("enable_thinking")
        thinking = app.enable_thinking if requested is None else bool(requested)
        if has_images(messages):
            if not generation:
                raise RequestError("add_generation_prompt false is not supported with images")
            ids = prepare_prompt(app, messages, tools, thinking, None, fields).tokens
        else:
            with app.tokenizer_lock:
                ids = render_prompt_ids(app.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                        reasoning_effort=app.effort_for(fields.get("reasoning_effort")),
                                        add_generation_prompt=generation, late_system=app.late_system)
    else:
        text = body.get("prompt")
        if not isinstance(text, str):
            raise RequestError("prompt must be a string (or send messages)")
        with app.tokenizer_lock:
            ids = list(app.tokenizer.encode(text, add_special_tokens=flag(body, "add_special_tokens", True)))
    reply: dict[str, Any] = {"count": len(ids), "max_model_len": int(getattr(app, "context_window", 0)) or None,
                             "tokens": [int(t) for t in ids]}
    if strings:
        with app.tokenizer_lock:
            reply["token_strs"] = list(app.tokenizer.convert_ids_to_tokens(reply["tokens"]))
    return reply


def detokenize(app: Any, body: Any) -> dict[str, Any]:
    """The text of ``tokens``, special tokens included."""

    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    with app.tokenizer_lock:
        ids = token_ids(body.get("tokens"), _vocab(app.tokenizer))
        return {"prompt": app.tokenizer.decode(ids, skip_special_tokens=False)}
