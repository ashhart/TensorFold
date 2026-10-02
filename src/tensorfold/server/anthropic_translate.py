"""Anthropic Messages translated to the server's existing chat request and reply contracts."""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import EFFORTS


def _string(value: Any, name: str, *, empty: bool = True) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise RequestError(f"{name} must be a {'nonempty ' if not empty else ''}string")
    return value


def _parts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list) or any(not isinstance(p, dict) for p in value):
        raise RequestError("content must be a string or an array of content blocks")
    return value


def _image(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise RequestError("image.source must be an object")
    if source.get("type") == "base64":
        media = _string(source.get("media_type"), "image.source.media_type", empty=False)
        if media not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise RequestError("unsupported image media_type")
        url = f"data:{media};base64,{_string(source.get('data'), 'image.source.data', empty=False)}"
    elif source.get("type") == "url":
        url = _string(source.get("url"), "image.source.url", empty=False)
    else:
        raise RequestError("image.source.type must be base64 or url; file IDs are unsupported")
    return {"type": "image_url", "image_url": {"url": url}}


def messages(value: Any, system: Any = None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RequestError("messages must be a nonempty array")
    out: list[dict[str, Any]] = []
    if system is not None:
        parts = _parts(system)
        if any(p.get("type") != "text" for p in parts):
            raise RequestError("system accepts text blocks only")
        out.append({"role": "system", "content": "\n\n".join(_string(p.get("text"), "system.text") for p in parts)})
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise RequestError("message role must be user or assistant")
        role = message["role"]
        content, calls, thoughts = [], [], []
        for block in _parts(message.get("content")):
            kind = block.get("type")
            if kind == "text":
                content.append({"type": "text", "text": _string(block.get("text"), "text")})
            elif kind == "image" and role == "user":
                content.append(_image(block))
            elif kind == "thinking" and role == "assistant":
                thoughts.append(_string(block.get("thinking"), "thinking"))
            elif kind == "redacted_thinking" and role == "assistant":
                raise RequestError("redacted thinking cannot be decoded by this local model; send plaintext thinking")
            elif kind == "tool_use" and role == "assistant":
                arguments = block.get("input")
                if not isinstance(arguments, dict):
                    raise RequestError("tool_use.input must be an object")
                calls.append({"id": _string(block.get("id"), "tool_use.id", empty=False), "type": "function",
                              "function": {"name": _string(block.get("name"), "tool_use.name", empty=False),
                                           "arguments": json.dumps(arguments, ensure_ascii=False)}})
            elif kind == "tool_result" and role == "user":
                # Tool results precede the following user text, including a batch of parallel results.
                texts, images = [], []
                for part in _parts(block.get("content", "")):
                    if part.get("type") == "text":
                        texts.append(_string(part.get("text"), "tool_result.text"))
                    elif part.get("type") == "image":
                        images.append(_image(part))
                    else:
                        raise RequestError(f"unsupported tool_result content type {part.get('type')!r}")
                text = "\n".join(texts)
                if block.get("is_error"):
                    text = "Tool error: " + text
                out.append({"role": "tool", "tool_call_id": _string(block.get("tool_use_id"), "tool_use_id", empty=False),
                            "content": text})
                content.extend(images)
            else:
                raise RequestError(f"unsupported {role} content block type {kind!r}")
        if content or calls or thoughts:
            item: dict[str, Any] = {"role": role, "content": content or ""}
            if calls:
                item["tool_calls"] = calls
            if thoughts:
                item["reasoning_content"] = "".join(thoughts)
            out.append(item)
        elif not _parts(message.get("content")):
            out.append({"role": role, "content": ""})
    return out


def translate(body: Any, *, count: bool = False) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise RequestError("request body must be an object")
    model = _string(body.get("model"), "model", empty=False)
    chat: dict[str, Any] = {"model": model, "messages": messages(body.get("messages"), body.get("system"))}
    if not count:
        limit = body.get("max_tokens")
        if type(limit) is not int or limit <= 0:
            raise RequestError("max_tokens must be a positive integer")
        chat["max_tokens"] = limit
    else:
        chat["max_tokens"] = 1
    if "stream" in body:
        if type(body["stream"]) is not bool:
            raise RequestError("stream must be a boolean")
        chat["stream"] = body["stream"]
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            chat[key] = body[key]
    if "stop_sequences" in body:
        stops = body["stop_sequences"]
        if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
            raise RequestError("stop_sequences must be an array of nonempty strings")
        chat["stop"] = stops
    context = body.get("context_management")
    # Claude Code keeps all thinking; the client supplies history and there is no stored context to edit.
    if context is not None and (not isinstance(context, dict) or not isinstance(context.get("edits", []), list)
            or any(edit != {"type": "clear_thinking_20251015", "keep": "all"} for edit in context.get("edits", []))):
        raise RequestError("context_management supports clear_thinking with keep: all only")
    for field in ("container", "mcp_servers", "service_tier"):
        if body.get(field) not in (None, [], "auto"):
            raise RequestError(f"{field} is not supported by this server")
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise RequestError("tools must be an array")
    translated = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "custom") != "custom":
            raise RequestError("only client-defined function tools are supported")
        name = _string(tool.get("name"), "tool.name", empty=False)
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            raise RequestError("tool.input_schema must be an object")
        fn = {"name": name, "parameters": schema}
        if "description" in tool:
            fn["description"] = _string(tool["description"], "tool.description")
        translated.append({"type": "function", "function": fn})
    if translated:
        chat["tools"] = translated
    choice = body.get("tool_choice")
    if choice is not None:
        if not isinstance(choice, dict):
            raise RequestError("tool_choice must be an object")
        kind = choice.get("type")
        if kind in ("auto", "none", "any"):
            chat["tool_choice"] = "required" if kind == "any" else kind
        elif kind == "tool":
            chat["tool_choice"] = {"type": "function", "function": {
                "name": _string(choice.get("name"), "tool_choice.name", empty=False)}}
        else:
            raise RequestError("tool_choice.type must be auto, none, any or tool")
        if "disable_parallel_tool_use" in choice:
            if type(choice["disable_parallel_tool_use"]) is not bool:
                raise RequestError("disable_parallel_tool_use must be a boolean")
            chat["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    config = body.get("output_config", {})
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise RequestError("output_config must be an object")
    if "effort" in config:
        if config["effort"] not in EFFORTS:
            raise RequestError("unsupported output_config.effort")
        chat["reasoning_effort"] = config["effort"]
    if config.get("format") is not None:
        fmt = config["format"]
        if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
            raise RequestError("output_config.format must be json_schema with a schema object")
        chat["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt["schema"],
                                                                       "strict": True}}
    thinking = body.get("thinking")
    if thinking is None:
        thinking = {"type": "disabled"}
    if thinking is not None:
        if not isinstance(thinking, dict) or thinking.get("type") not in ("disabled", "enabled", "adaptive"):
            raise RequestError("thinking.type must be disabled, enabled or adaptive")
        enabled = thinking["type"] != "disabled"
        chat["chat_template_kwargs"] = {"enable_thinking": enabled}
        if not enabled:
            chat["reasoning_effort"] = "none"
        if thinking["type"] == "enabled":
            budget = thinking.get("budget_tokens")
            if type(budget) is not int or budget <= 0 or (not count and budget >= body["max_tokens"]):
                raise RequestError("thinking.budget_tokens must be positive and less than max_tokens")
            chat["thinking_budget"] = budget
    return chat


def usage(value: dict[str, Any]) -> dict[str, Any]:
    prompt = value.get("prompt_tokens", 0)
    cached = min(prompt, max(0, (value.get("prompt_tokens_details") or {}).get("cached_tokens", 0)))
    result = {"input_tokens": prompt - cached, "output_tokens": value.get("completion_tokens", 0),
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}
    details = value.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        result["output_tokens_details"] = {"thinking_tokens": details["reasoning_tokens"]}
    return result


def error(message: str, status: int = 400) -> dict[str, Any]:
    kind = ("invalid_request_error" if status == 400 else "not_found_error" if status == 404 else
            "overloaded_error" if status == 503 else "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


class Reply:
    """Typed Messages events from chat chunks, keeping each delta on a block of the matching type."""

    def __init__(self, model: str, send: Callable[[dict[str, Any]], None]) -> None:
        self.send = send
        self.base = {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model,
                     "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage({})}
        self.index = -1
        self.kind: str | None = None
        self.calls: dict[int, dict[str, Any]] = {}
        self.pending_text: list[tuple[str, str]] = []
        self.active_call: int | None = None
        self.finished = False
        self.finish = "end_turn"
        self.stop_sequence: str | None = None
        self.tokens = usage({})

    def start(self) -> None:
        self.send({"type": "message_start", "message": self.base})

    def close(self) -> None:
        if self.kind:
            if self.kind == "thinking":
                # Local reasoning is plaintext, with no provider signature or encryption.
                self.send({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "signature_delta", "signature": ""}})
            self.send({"type": "content_block_stop", "index": self.index})
            self.kind = None

    def text(self, kind: str, value: str) -> None:
        if not value:
            return
        if self.active_call is not None:
            self.pending_text.append((kind, value))
            return
        if kind != self.kind:
            self.close()
            self.index += 1
            self.kind = kind
            block = {"type": kind, "thinking" if kind == "thinking" else "text": ""}
            self.send({"type": "content_block_start", "index": self.index, "content_block": block})
        self.send({"type": "content_block_delta", "index": self.index,
                   "delta": {"type": kind + "_delta", "thinking" if kind == "thinking" else "text": value}})

    def chunk(self, chunk: dict[str, Any] | None) -> None:
        if self.finished:
            return
        if chunk is None:
            self.drain_calls(final=True)
            self.close()
            self.send({"type": "message_delta", "delta": {"stop_reason": self.finish, "stop_sequence": self.stop_sequence},
                       "usage": self.tokens})
            self.send({"type": "message_stop"})
            self.finished = True
            return
        if "error" in chunk:
            self.close()
            problem = chunk["error"]
            self.send(error(problem.get("message", "generation failed"),
                            400 if problem.get("type") == "invalid_request_error" else 500))
            self.finished = True
            return
        if chunk.get("usage") is not None:
            self.tokens = usage(chunk["usage"])
        if chunk.get("stop_sequence") is not None:
            self.stop_sequence = chunk["stop_sequence"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                self.finish = {"tool_calls": "tool_use", "length": "max_tokens"}.get(choice["finish_reason"], "end_turn")
            if self.stop_sequence is not None:
                self.finish = "stop_sequence"
            delta = choice.get("delta") or {}
            self.text("thinking", delta.get("reasoning_content") or "")
            self.text("text", delta.get("content") or "")
            for call in delta.get("tool_calls") or []:
                at = call.get("index", 0)
                target = self.calls.setdefault(at, {"id": "", "name": "", "arguments": "", "sent": 0, "done": False})
                if call.get("id"):
                    target["id"] = call["id"]
                fn = call.get("function") or {}
                target["name"] += fn.get("name") or ""
                target["arguments"] += fn.get("arguments") or ""

        self.drain_calls()

    def drain_calls(self, *, final: bool = False) -> None:
        for at, call in self.calls.items():
            if call["done"]:
                continue
            if not call["id"] or not call["name"]:
                return
            if self.active_call is None:
                self.close()
                self.index += 1
                self.kind, self.active_call = "tool_use", at
                self.send({"type": "content_block_start", "index": self.index,
                           "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}}})
            if at != self.active_call:
                return
            delta = call["arguments"][call["sent"]:]
            if delta:
                self.send({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "input_json_delta", "partial_json": delta}})
                call["sent"] = len(call["arguments"])
            complete = False
            if call["arguments"].rstrip().endswith("}"):
                try:
                    complete = isinstance(json.loads(call["arguments"]), dict)
                except ValueError:
                    pass
            if not complete and not final:
                return
            self.close()
            call["done"], self.active_call = True, None
            pending, self.pending_text = self.pending_text, []
            for kind, text in pending:
                self.text(kind, text)

    def completion(self, chunk: dict[str, Any]) -> dict[str, Any]:
        choice = chunk["choices"][0]
        message = choice["message"]
        content = []
        if message.get("reasoning_content"):
            content.append({"type": "thinking", "thinking": message["reasoning_content"], "signature": ""})
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            if not isinstance(args, dict):
                raise TypeError("model tool arguments are not a JSON object")
            content.append({"type": "tool_use", "id": call["id"], "name": fn["name"], "input": args})
        return {**self.base, "content": content, "usage": usage(chunk.get("usage") or {}),
                "stop_sequence": chunk.get("stop_sequence"),
                "stop_reason": "stop_sequence" if chunk.get("stop_sequence") is not None else
                {"tool_calls": "tool_use", "length": "max_tokens"}.get(choice["finish_reason"], "end_turn")}
