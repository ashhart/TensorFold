"""Anthropic Messages requests as the chat completions that run them, and chat replies back as Messages.

Mirrors ``responses_translate``: a request becomes the chat completion that runs it (this server's
own handler, via ``responses.Wire``), and the chat reply becomes Anthropic's reply objects and
stream events. Claude Code and other Anthropic-protocol clients work without an adapter.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from tensorfold.server.errors import RequestError
from tensorfold.server.probabilities import probability_options

# fields a chat completion reads as they are (this server's own names among them)
PASSED = ("model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop",
          "ignore_eos", "priority", "return_token_ids")
REFUSED = {"mcp_servers": "MCP servers are not supported: send tools directly",
           "context_management": "context management is not supported"}
EFFORTS = {"none": "none", "off": "none", "minimal": "minimal", "low": "low", "medium": "medium",
           "high": "high", "max": "xhigh"}


@dataclass
class Request:
    chat: dict[str, Any]                 # the chat completion that runs it
    stream: bool = False


def _content(content: Any) -> str | list[dict[str, Any]]:
    """A message's content as chat content: text parts as text, images as image_url parts."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be a string or a list of content blocks")
    parts = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") != "base64":
                raise RequestError("image blocks need a base64 source (url sources are not supported)")
            url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "thinking" and isinstance(block.get("thinking"), str):
            continue                                      # replayed assistant thought: the chat needs the text only
        elif kind == "tool_use":
            continue                                      # handled by the caller (a call joins its message)
        elif kind == "tool_result":
            continue                                      # handled by the caller (a result is its own message)
        else:
            raise RequestError(f"content blocks of type {kind!r} are not supported: send text, image, tool_use or "
                               "tool_result blocks")
    return parts


def messages(blocks: list[Any]) -> list[dict[str, Any]]:
    """Anthropic messages as chat messages: tool_use joins its assistant message, tool_result becomes a tool turn."""

    out: list[dict[str, Any]] = []
    thought: str | None = None                    # replayed thinking for the assistant message that follows
    pending: list[dict[str, Any]] = []            # a message's tool_use blocks, kept until its text is seen
    for block in blocks:
        if not isinstance(block, dict) or block.get("role") not in ("user", "assistant"):
            raise RequestError("each message needs a role of user or assistant (system is the top-level field)")
        thought = None
        pending = []
        message: dict[str, Any] = {"role": block["role"]}
        kinds = [b.get("type") for b in block.get("content") or [] if isinstance(b, dict)]
        if "tool_result" in kinds:
            # Anthropic carries tool results in a user turn; the chat template reads the tool turn and
            # still needs a user message after it, so text blocks beside the results become that turn
            texts = []
            for part in block["content"]:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool_result":
                    if not isinstance(part.get("tool_use_id"), str):
                        raise RequestError("a tool_result block needs the tool_use_id of its call")
                    out.append({"role": "tool", "tool_call_id": part["tool_use_id"],
                                "content": _content(part.get("content") or "")})
                elif part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            out.append({"role": "user", "content": "\n".join(texts) if texts else "(tool results above)"})
            continue
        for part in block.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                if not isinstance(part.get("id"), str) or not isinstance(part.get("name"), str):
                    raise RequestError("a tool_use block needs an id and a name")
                pending.append({"id": part["id"], "type": "function",
                                "function": {"name": part["name"],
                                             "arguments": _json_or_text(part.get("input"))}})
            elif part.get("type") == "thinking":
                thought = part.get("thinking") if isinstance(part.get("thinking"), str) else None
        content = _content(block.get("content"))
        if content:
            message["content"] = content
        if pending:
            message.setdefault("content", "")
            message["tool_calls"] = pending
        if thought and block["role"] == "assistant":
            message["reasoning_content"] = thought
        if "content" in message or pending:
            out.append(message)
    return out


def _json_or_text(value: Any) -> str:
    """A tool_use input as the argument string a chat tool call carries."""

    import json

    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        raise RequestError("a tool_use input must be a JSON object or a string") from None


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError("tools must be a list")
    out = []
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind not in ("custom", "function", None):
            raise RequestError(f"tools of type {kind!r} are not supported: this server runs function tools only")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise RequestError("a tool needs a name")
        out.append({"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters",
                                                                         "strict") if tool.get(k) is not None}})
    return out


def _tool_choice(choice: Any) -> Any:
    if choice is None:
        return None
    if choice in ("auto", "any", "none"):
        return "required" if choice == "any" else choice      # Anthropic's any is OpenAI's required
    if isinstance(choice, dict) and choice.get("type") in ("auto", "any", "tool", "function"):
        if choice.get("type") in ("tool", "function") and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
        return "required" if choice.get("type") == "any" else choice.get("type")
    raise RequestError("tool_choice must be auto, any, none or a named tool")


def translate(body: Any) -> Request:
    """A Messages request as the chat completion that runs it; RequestError where it asks what this server lacks."""

    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    probability_options(body)
    for name, reason in REFUSED.items():
        if body.get(name):
            raise RequestError(reason)
    given = body.get("messages")
    if not isinstance(given, list) or not given:
        raise RequestError("messages must be a non-empty list")
    if body.get("max_tokens") is None:
        raise RequestError("max_tokens is required")
    chat: dict[str, Any] = {k: body[k] for k in PASSED if k in body}
    chat["messages"] = ([{"role": "system", "content": body["system"]}] if isinstance(body.get("system"), str)
                        else ([{"role": "system", "content": _content(body["system"])}]
                              if isinstance(body.get("system"), list) and body["system"] else [])) + messages(given)
    tools = _tools(body.get("tools"))
    if tools is not None:
        chat["tools"] = tools
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = _tool_choice(body["tool_choice"])
    if body.get("max_tokens") is not None:
        chat["max_tokens"] = body["max_tokens"]
    thinking = body.get("thinking") or {}
    if not isinstance(thinking, dict):
        raise RequestError("thinking must be an object")
    if thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if not isinstance(budget, int) or budget < 1024:
            raise RequestError("thinking.budget_tokens must be an integer of at least 1024")
        chat["thinking_budget"] = budget
        chat.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    elif thinking.get("type") == "disabled":
        chat.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if isinstance(body.get("thinking") is None and body.get("reasoning_effort") or None, str):
        chat["reasoning_effort"] = EFFORTS.get(body["reasoning_effort"], body["reasoning_effort"])
    stop = body.get("stop_sequences")
    if stop is not None:
        if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
            raise RequestError("stop_sequences must be a list of strings")
        chat["stop"] = stop
    return Request(chat, stream=bool(body.get("stream")))


# -- replies ----------------------------------------------------------------------------------


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def usage(chat: dict[str, Any] | None) -> dict[str, Any] | None:
    """A chat completion's usage as a Messages usage."""

    if not chat:
        return None
    return {"input_tokens": chat.get("prompt_tokens", 0), "output_tokens": chat.get("completion_tokens", 0)}


class Reply:
    """A Messages reply built from chat-completion deltas, in order: thinking, text, tool_use blocks."""

    def __init__(self, base: dict[str, Any], emit: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.base, self.emit = base, emit
        self.blocks: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None     # the block being written
        self.calls: dict[int, dict[str, Any]] = {}     # chat tool-call index -> its tool_use block
        self.seq = 0
        self.stop_reason: str | None = None
        self.usage_final: dict[str, Any] | None = None

    def _event(self, kind: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit({"type": kind, **fields})
            self.seq += 1

    def start(self) -> None:
        self._event("message_start", message={**self.base, "usage": {"input_tokens": 0, "output_tokens": 0}})

    def _where(self, block: dict[str, Any]) -> dict[str, Any]:
        return {"index": next(i for i, x in enumerate(self.blocks) if x is block)}

    def _open(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._close()
        if kind == "text":
            block = {"type": "text", "text": ""}
        elif kind == "thinking":
            block = {"type": "thinking", "thinking": ""}
        else:
            block = {"type": "tool_use", "id": fields.get("id") or _id("toolu"), "name": fields.get("name") or "",
                     "input": {}}
            self._json = ""
        self.blocks.append(block)
        self.current = block
        self._event("content_block_start", **self._where(block), content_block=block)
        return block

    def _close(self) -> None:
        block, self.current = self.current, None
        if block is None:
            return
        if block["type"] == "tool_use":
            import json

            try:
                block["input"] = json.loads(self._json or "{}")
            except ValueError:
                block["input"] = {"raw": self._json}
        self._event("content_block_stop", **self._where(block))

    def _write(self, kind: str, text: str) -> None:
        if kind == "tool_use":
            block = self.current if self.current is not None and self.current["type"] == "tool_use" else None
            if block is None:
                block = self._open(kind)
            self._json += text
            self._event("content_block_delta", **self._where(block),
                        delta={"type": "input_json_delta", "partial_json": text})
            return
        block = self.current if self.current is not None and self.current["type"] == kind else self._open(kind)
        key = "text" if kind == "text" else "thinking"
        block[key] += text
        self._event("content_block_delta", **self._where(block),
                    delta={"type": "text_delta" if kind == "text" else "thinking_delta", "text": text})

    def delta(self, delta: dict[str, Any]) -> None:
        thought = delta.get("reasoning_content") or delta.get("reasoning")
        if thought:
            self._write("thinking", thought)
        if delta.get("content"):
            self._write("text", delta["content"])
        for call in delta.get("tool_calls") or []:
            function = call.get("function") or {}
            block = self.calls.get(call.get("index", 0))
            if block is None:
                block = self.calls[call.get("index", 0)] = self._open(
                    "tool_use", id=call.get("id") or _id("toolu"), name=function.get("name") or "")
            if function.get("arguments"):
                self._write("tool_use", function["arguments"])

    def finish(self, reason: str | None, chat_usage: dict[str, Any] | None) -> dict[str, Any]:
        """The reply ended (``reason`` is the chat finish_reason): the last block closes and the message is final."""

        self.stop_reason = {"tool_calls": "tool_use", "length": "max_tokens"}.get(reason or "", "end_turn")
        self._close()
        self.usage_final = usage(chat_usage)
        final = {**self.base, "content": self.blocks, "stop_reason": self.stop_reason,
                 "stop_sequence": None, "usage": self.usage_final or {"input_tokens": 0, "output_tokens": 0}}
        self._event("message_delta", delta={"stop_reason": self.stop_reason, "stop_sequence": None},
                    usage=self.usage_final or {})
        self._event("message_stop")
        return final

    def fail(self, error: Any) -> dict[str, Any]:
        self._close()
        error = error if isinstance(error, dict) else {"message": str(error)}
        self._event("error", error={"type": "api_error",
                                    "message": str(error.get("message") or "the reply failed")})
        return {"type": "error", "error": {"type": "api_error",
                                           "message": str(error.get("message") or "the reply failed")}}

    def chunk(self, payload: dict[str, Any] | None) -> None:
        """One chat-completion stream event (None: its ``[DONE]``)."""

        if self.usage_final is not None:
            return
        if payload is None:
            self.fail("the reply ended early")
        elif "error" in payload:
            self.fail(payload["error"])
        else:
            choice = (payload.get("choices") or [{}])[0]
            self.delta(choice.get("delta") or {})
            if choice.get("finish_reason"):
                self.finish(choice["finish_reason"], payload.get("usage"))

    def completion(self, data: dict[str, Any]) -> dict[str, Any]:
        """A whole chat completion (not streamed)."""

        choice = data["choices"][0]
        message = choice.get("message") or {}
        self.delta({"reasoning_content": message.get("reasoning_content"), "content": message.get("content"),
                    "tool_calls": [{"index": i, **call} for i, call in enumerate(message.get("tool_calls") or [])]})
        return self.finish(choice.get("finish_reason"), data.get("usage"))
