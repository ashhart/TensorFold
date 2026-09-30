"""Responses requests as the chat completions that run them, and chat replies as Responses (items, events, usage)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from tensorfold.server.errors import RequestError
from tensorfold.server.probabilities import probability_options

if TYPE_CHECKING:
    from tensorfold.server.responses import Store

# fields a chat completion reads as they are: OpenAI's and this server's own (top_k, min_p, seed, draft, ...)
PASSED = ("model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop", "draft",
          "thinking_budget", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs")
REFUSED = {"background": "background responses are not supported: send the request and wait for it",
           "conversation": "conversations are not supported: send previous_response_id or the items",
           "prompt": "prompt templates are not supported: send input and instructions",
           "context_management": "context management is not supported"}


@dataclass
class Request:
    chat: dict[str, Any]                 # the chat completion that runs it
    echo: dict[str, Any]                 # the request fields its Response repeats
    added: list[dict[str, Any]]          # the chat messages it adds to its conversation (instructions not carried)
    store: bool = True
    stream: bool = False


def _content(content: Any) -> str | list[dict[str, Any]]:
    """A message's content as chat content: text parts as text, images as image_url parts."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be a string or a list of content parts")
    parts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text") and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            parts.append({"type": "text", "text": part["refusal"]})
        elif kind == "input_image":
            if not isinstance(part.get("image_url"), str):
                raise RequestError("input_image needs an image_url (a URL or data URL); file ids are not supported")
            url = {"url": part["image_url"], **({"detail": part["detail"]} if part.get("detail") else {})}
            parts.append({"type": "image_url", "image_url": url})
        else:
            raise RequestError(f"content parts of type {kind!r} are not supported: send input_text or input_image")
    return parts


def _output(output: Any) -> str:
    """A function_call_output's output as a tool message's text."""

    if isinstance(output, str):
        return output
    if isinstance(output, list) and all(isinstance(p, dict) and p.get("type") == "input_text" for p in output):
        return "".join(str(p.get("text") or "") for p in output)
    raise RequestError("a function_call_output's output must be a string or input_text parts")


def messages(items: list[Any]) -> list[dict[str, Any]]:
    """Input items (output items too) as chat messages: a turn's reasoning, text and calls are one assistant message."""

    out: list[dict[str, Any]] = []
    thought: str | None = None                    # reasoning for the assistant message that follows it
    for item in items:
        if not isinstance(item, dict):
            raise RequestError("each input item must be an object")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise RequestError("a message's role must be user, assistant, system or developer")
            message = {"role": role, "content": _content(item.get("content"))}
            if role == "assistant" and thought is not None:
                message["reasoning_content"], thought = thought, None
            out.append(message)
        elif kind == "function_call":
            if not isinstance(item.get("call_id"), str) or not isinstance(item.get("name"), str):
                raise RequestError("a function_call item needs a call_id and a name")
            call = {"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}}
            if not out or out[-1]["role"] != "assistant":          # a turn that opens with its call
                out.append({"role": "assistant", "content": ""})
            if thought is not None:
                out[-1]["reasoning_content"], thought = thought, None
            out[-1].setdefault("tool_calls", []).append(call)
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise RequestError("a function_call_output item needs the call_id of its call")
            out.append({"role": "tool", "tool_call_id": item["call_id"], "content": _output(item.get("output"))})
        elif kind == "reasoning":
            texts = [p.get("text") for p in item.get("content") or [] if isinstance(p, dict)]
            if item.get("encrypted_content") and not texts:
                raise RequestError("encrypted reasoning is not supported: send the reasoning item's content")
            thought = "".join(t for t in texts if isinstance(t, str)) or None
        else:
            raise RequestError(f"input items of type {kind!r} are not supported: send messages, function_call, "
                               "function_call_output and reasoning items")
    return out


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError("tools must be a list")
    out = []
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise RequestError(f"tools of type {kind!r} are not supported: this server runs function tools only")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise RequestError("a function tool needs a name")
        out.append({"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters", "strict")
                                                     if tool.get(k) is not None}})
    return out


def _tool_choice(choice: Any, tools: list[dict[str, Any]] | None) -> tuple[Any, list[dict[str, Any]] | None]:
    if choice is None or choice in ("none", "auto", "required"):
        return choice, tools
    kind = choice.get("type") if isinstance(choice, dict) else None
    if kind == "function" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}, tools
    if kind == "allowed_tools" and choice.get("mode", "auto") in ("auto", "required"):
        allowed = choice.get("tools") or []
        if not all(isinstance(t, dict) and t.get("type") == "function" for t in allowed):
            raise RequestError("allowed_tools may name function tools only")
        names = {t.get("name") for t in allowed}
        return choice.get("mode", "auto"), [t for t in tools or [] if t["function"]["name"] in names]
    raise RequestError("tool_choice must be none, auto, required, a function or allowed_tools of functions")


def _format(text: Any) -> dict[str, Any] | None:
    """``text.format`` as a chat ``response_format`` (None for plain text)."""

    fmt = text.get("format") if isinstance(text, dict) else None
    if text is not None and not isinstance(text, dict):
        raise RequestError("text must be an object")
    if fmt is None or fmt.get("type") == "text":
        return None
    if fmt.get("type") == "json_object":
        return {"type": "json_object"}
    if fmt.get("type") == "json_schema":
        return {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict", "description")
                                                       if fmt.get(k) is not None}}
    raise RequestError("text.format must be text, json_object or json_schema")


def translate(body: Any, store: Store) -> Request:
    """A Responses request as the chat completion that runs it; RequestError where it asks what this server lacks."""

    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    probability_options(body)
    for name, reason in REFUSED.items():
        if body.get(name):
            raise RequestError(reason)
    if body.get("include"):
        raise RequestError(f"include is not supported ({', '.join(map(str, body['include']))}): reasoning comes as "
                           "text in its reasoning item, and nothing is encrypted")
    if body.get("truncation") not in (None, "disabled"):
        raise RequestError('truncation must be "disabled": a prompt too long for the context is refused')
    if body.get("top_logprobs"):
        raise RequestError("top_logprobs is not supported")
    metadata = body.get("metadata") or {}
    if not isinstance(metadata, dict) or len(metadata) > 16 or not all(
            isinstance(k, str) and len(k) <= 64 and isinstance(v, str) and len(v) <= 512 for k, v in metadata.items()):
        raise RequestError("metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)")
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise RequestError("instructions must be a string")
    given = body.get("input")
    if isinstance(given, str):
        given = [{"role": "user", "content": given}]
    if not isinstance(given, list) or not given:
        raise RequestError("input must be a string or a non-empty list of items")
    added = messages(given)
    parent = body.get("previous_response_id")
    history = store.conversation(parent) if parent is not None else []
    tools = _tools(body.get("tools"))
    choice, offered = _tool_choice(body.get("tool_choice"), tools)
    chat: dict[str, Any] = {k: body[k] for k in PASSED if k in body}
    chat["messages"] = ([{"role": "system", "content": instructions}] if instructions else []) + history + added
    if offered is not None:
        chat["tools"] = offered
    if choice is not None:
        chat["tool_choice"] = choice
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    reasoning = body.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        raise RequestError("reasoning must be an object")
    if reasoning.get("effort") is not None:
        chat["reasoning_effort"] = reasoning["effort"]   # the effort rule of chat completions: unset is the template's
    fmt = _format(body.get("text"))
    if fmt is not None:
        chat["response_format"] = fmt
    echo = {"instructions": instructions, "max_output_tokens": body.get("max_output_tokens"), "metadata": metadata,
            "parallel_tool_calls": body.get("parallel_tool_calls", True), "previous_response_id": parent,
            "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")},
            "store": body.get("store") is not False, "temperature": body.get("temperature"),
            "text": {"format": (body.get("text") or {}).get("format") or {"type": "text"}},
            "tool_choice": body.get("tool_choice") or "auto", "tools": body.get("tools") or [],
            "top_p": body.get("top_p"), "truncation": "disabled", "user": body.get("user"), "background": False}
    return Request(chat, echo, added, store=echo["store"], stream=bool(body.get("stream")))


# -- replies ----------------------------------------------------------------------------------


def usage(chat: dict[str, Any] | None) -> dict[str, Any] | None:
    """A chat completion's usage as a Response's."""

    if not chat:
        return None
    cached = (chat.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    thought = (chat.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return {"input_tokens": chat["prompt_tokens"], "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": chat["completion_tokens"], "output_tokens_details": {"reasoning_tokens": thought},
            "total_tokens": chat["prompt_tokens"] + chat["completion_tokens"]}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Reply:
    """A Response built from chat-completion deltas, in order: reasoning, text, calls; ``emit`` hears its events."""

    def __init__(self, base: dict[str, Any], emit: Callable[[dict[str, Any]], None] | None = None,
                 keep: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.base, self.emit, self.keep = base, emit, keep      # keep: stores a finished Response before it is sent
        self.items: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None     # the item being written
        self.calls: dict[int, dict[str, Any]] = {}     # chat tool-call index -> its function_call item
        self.seq = 0
        self.final: dict[str, Any] | None = None

    def _event(self, kind: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit({"type": kind, "sequence_number": self.seq, **fields})
            self.seq += 1

    def start(self) -> None:
        self._event("response.created", response=self.base)
        self._event("response.in_progress", response=self.base)

    def _where(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"item_id": item["id"], "output_index": next(i for i, x in enumerate(self.items) if x is item)}

    def _open(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._close("completed")
        if kind == "message":
            item = {"id": _id("msg"), "type": "message", "role": "assistant", "status": "in_progress", "content": []}
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        elif kind == "reasoning":
            item = {"id": _id("rs"), "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}
            part = {"type": "reasoning_text", "text": ""}
        else:
            item, part = {"id": _id("fc"), "type": "function_call", **fields, "status": "in_progress"}, None
        self.items.append(item)
        self.current = item
        self._event("response.output_item.added", output_index=len(self.items) - 1, item=item)
        if part is not None:
            item["content"].append(part)
            self._event("response.content_part.added", **self._where(item), content_index=0, part=part)
        return item

    def _close(self, status: str) -> None:
        item, self.current = self.current, None
        if item is None:
            return
        item["status"] = status
        where = self._where(item)
        if item["type"] == "function_call":
            self._event("response.function_call_arguments.done", **where, name=item["name"],
                        arguments=item["arguments"])
        else:
            part = item["content"][0]
            kind, extra = (("output_text", {"logprobs": []}) if item["type"] == "message" else ("reasoning_text", {}))
            self._event(f"response.{kind}.done", **where, content_index=0, text=part["text"], **extra)
            self._event("response.content_part.done", **where, content_index=0, part=part)
        self._event("response.output_item.done", output_index=where["output_index"], item=item)

    def _write(self, kind: str, text: str) -> None:
        item = self.current if self.current is not None and self.current["type"] == kind else self._open(kind)
        item["content"][0]["text"] += text
        name, extra = ("output_text", {"logprobs": []}) if kind == "message" else ("reasoning_text", {})
        self._event(f"response.{name}.delta", **self._where(item), content_index=0, delta=text, **extra)

    def delta(self, delta: dict[str, Any]) -> None:
        thought = delta.get("reasoning_content") or delta.get("reasoning")
        if thought:
            self._write("reasoning", thought)
        if delta.get("content"):
            self._write("message", delta["content"])
        for call in delta.get("tool_calls") or []:
            function = call.get("function") or {}
            item = self.calls.get(call.get("index", 0))
            if item is None:
                item = self.calls[call.get("index", 0)] = self._open(
                    "function_call", call_id=call.get("id") or _id("call"), name=function.get("name") or "",
                    arguments="")
            if function.get("arguments"):
                item["arguments"] += function["arguments"]
                self._event("response.function_call_arguments.delta", **self._where(item),
                            delta=function["arguments"])

    def finish(self, reason: str | None, chat_usage: dict[str, Any] | None, stats: Any = None) -> dict[str, Any]:
        """The reply ended (``reason`` is the chat finish_reason): its last item closes and the Response is final."""

        status = "incomplete" if reason == "length" else "completed"
        if status == "completed" and all(item["type"] == "reasoning" for item in self.items):
            self._open("message")                          # an empty answer is still a message
        self._close(status)
        final = {**self.base, "status": status, "output": self.items, "usage": usage(chat_usage),
                 "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None}
        if status == "completed":
            final["completed_at"] = int(time.time())
        if stats is not None:
            final["tensorfold"] = stats
        self.final = final
        if self.keep is not None:                  # before the client hears of it: its next request may name it
            self.keep(final)
        self._event(f"response.{status}", response=final)
        return final

    def fail(self, error: Any) -> dict[str, Any]:
        self._close("incomplete")
        error = error if isinstance(error, dict) else {"message": str(error)}
        code = "invalid_prompt" if error.get("type") == "invalid_request_error" else "server_error"
        self.final = {**self.base, "status": "failed", "output": self.items,
                      "error": {"code": code, "message": str(error.get("message") or "the reply failed")}}
        self._event("response.failed", response=self.final)
        return self.final

    def chunk(self, payload: dict[str, Any] | None) -> None:
        """One chat-completion stream event (None: its ``[DONE]``)."""

        if self.final is not None:
            return
        if payload is None:
            self.fail("the reply ended early")
        elif "error" in payload:
            self.fail(payload["error"])
        else:
            choice = (payload.get("choices") or [{}])[0]
            self.delta(choice.get("delta") or {})
            if choice.get("finish_reason"):
                self.finish(choice["finish_reason"], payload.get("usage"), payload.get("tensorfold"))

    def completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """A whole chat completion (not streamed)."""

        choice = data["choices"][0]
        message = choice.get("message") or {}
        self.delta({"reasoning_content": message.get("reasoning_content"), "content": message.get("content"),
                    "tool_calls": [{"index": i, **call} for i, call in enumerate(message.get("tool_calls") or [])]})
        return self.finish(choice.get("finish_reason"), data.get("usage"), data.get("tensorfold"))
