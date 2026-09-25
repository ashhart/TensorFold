"""The OpenAI-compatible HTTP layer: ``GET /v1/models``, ``GET /health``, ``POST /v1/chat/completions`` and
``POST /v1/completions``, with optional server-sent-event streaming, tool calls and reasoning text.

``make_handler(app)`` wraps any app with ``chat(messages, max_tokens=, temperature=, on_delta=, tools=,
sampling=)``, ``served_name``, ``model_ids``, ``tokenizer`` / ``tokenizer_lock`` and ``exact_mode``
(``server.app.ChatApp`` is the one TensorFold serves).
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Any

# TENSORFOLD_REQUEST_LOG=path appends every request body (one JSON a line), for exact replays of real traffic
_REQUEST_LOG = os.environ.get("TENSORFOLD_REQUEST_LOG", "")


class RequestError(ValueError):
    """A request the server refuses: answered with HTTP 400 (in a stream, an error event)."""


def served_model_ids(served_name: str, aliases: list[str] | None = None) -> list[str]:
    """Return the OpenAI model ids this endpoint advertises."""

    ids: list[str] = []
    for value in [served_name, *(aliases or [])]:
        model_id = str(value or "").strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make assistant tool_call arguments a MAPPING for the chat template.

    The OpenAI wire format carries `function.arguments` as a JSON STRING, and
    that is what real clients send. The Qwen3.6 template does

        {%- for args_name, args_value in tool_call.arguments|items %}

    which requires a mapping, so a string raises
    "TypeError: Can only get item pairs from a mapping." inside Jinja. The
    request then dies with no response at all -- the client sits on a spinner
    forever. It fires on every turn whose history contains a prior assistant
    tool call, i.e. every agentic turn after the first.

    Parse the string into a dict when possible and leave everything else alone.
    Copies only the messages it has to touch.
    """

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


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    """How many leading tokens two prompts share."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def longest_reusable_prefix(cached: list[int], requested: list[int]) -> int:
    """Return how many cached tokens can seed the new request.

    The live cache is reusable only when every cached token is a strict prefix
    of the request, leaving at least one request token to produce logits.
    """

    if not cached or len(cached) >= len(requested):
        return 0
    if requested[: len(cached)] != cached:
        return 0
    return len(cached)


def strip_trailing_stops(tokens: list[int], stop_ids: set[int]) -> list[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] in stop_ids:
        end -= 1
    return tokens[:end]


def tool_spec_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip() if isinstance(tool, dict) else ""


def normalize_tool_specs(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        if not tool_spec_name(tool):
            raise ValueError(f"tools[{index}] must include a function name")
        normalized.append(tool)
    return normalized


def tool_choice_disables_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "none"
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        return isinstance(value, str) and value.strip().lower() == "none"
    return False


def validate_tool_choice(tools: list[dict[str, Any]], tool_choice: Any) -> None:
    if not isinstance(tool_choice, dict):
        return
    if str(tool_choice.get("type") or "").lower() != "function":
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool_choice function must include a function object")
    requested = str(function.get("name") or "").strip()
    if not requested:
        raise ValueError("tool_choice function must include a name")
    known = {tool_spec_name(tool) for tool in tools}
    if requested not in known:
        raise ValueError(f"tool_choice requested unknown tool '{requested}'")


def active_tool_specs(tools: Any, tool_choice: Any) -> list[dict[str, Any]]:
    specs = normalize_tool_specs(tools)
    if not specs or tool_choice_disables_tools(tool_choice):
        return []
    validate_tool_choice(specs, tool_choice)
    return specs


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_NAMESPACED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$",
    re.IGNORECASE | re.DOTALL,
)
# One framing newline a side, as the chat template writes (and vLLM's Qwen parser reads) them: a
# value's own leading or trailing whitespace (a file's last newline, indentation) is part of it, and
# stripping it made the resent history differ from the tokens the model wrote, so the whole reply
# was prefilled again (a 15,007-token file write: 51 s to first token on the next turn).
_TOOL_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MISSING = object()


def _tool_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        parsed: Any = {}
    elif isinstance(value, str):
        text = value.strip()
        parsed = json.loads(text) if text else {}
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("tool_call arguments must be a JSON object")
    return parsed


def _loose_tool_arguments(payload: dict[str, Any], explicit: Any) -> Any:
    if explicit is not _MISSING:
        return explicit
    return {
        key: value
        for key, value in payload.items()
        if key not in {"name", "tool", "function", "call", "type"}
    }


def _parse_tool_call_payload(block: str) -> tuple[str, dict[str, Any]] | None:
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        for item in payload:
            parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False))
            if parsed is not None:
                return parsed
        return None
    if isinstance(payload, dict):
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name") or function.get("tool") or function.get("function")
            explicit_arguments = function.get(
                "arguments",
                function.get("args", function.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(function, explicit_arguments)
        else:
            name = (
                payload.get("name")
                or payload.get("tool")
                or payload.get("function")
                or payload.get("call")
            )
            explicit_arguments = payload.get(
                "arguments",
                payload.get("args", payload.get("parameters", _MISSING)),
            )
            arguments = _loose_tool_arguments(payload, explicit_arguments)
        name_text = str(name or "").strip()
        if not name_text:
            raise ValueError("tool_call is missing a function name")
        return name_text, _tool_json_object(arguments)

    match = _TOOL_FUNCTION_BLOCK_RE.match(block)
    if match is None:
        return None
    name = match.group(1).strip()
    arguments: dict[str, Any] = {}
    body = match.group(2)
    for param_match in _TOOL_PARAMETER_BLOCK_RE.finditer(body):
        arguments[param_match.group(1).strip()] = param_match.group(2)
    if not name:
        raise ValueError("tool_call is missing a function name")
    return name, arguments


def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE_RE.match(text)
    if match is None:
        return text
    return match.group(1).strip()


def _openai_tool_call(raw_name: str, arguments: dict[str, Any], known: dict[str, str]) -> dict[str, Any]:
    name = known.get(raw_name.lower())
    if name is None:
        raise ValueError(f"unknown tool '{raw_name}'")
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _parse_bare_json_tool_calls(text: str, known: dict[str, str]) -> list[dict[str, Any]] | None:
    stripped = _strip_json_fence(text.strip())
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    payloads = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for item in payloads:
        if not isinstance(item, dict):
            return None
        parsed = _parse_tool_call_payload(json.dumps(item, ensure_ascii=False))
        if parsed is None:
            return None
        raw_name, arguments = parsed
        if raw_name.lower() not in known:
            return None
        calls.append(_openai_tool_call(raw_name, arguments, known))
    return calls or None


def parse_tool_calls_from_content(
    text: str,
    tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]] | None]:
    if not tools:
        return text, None
    known = {tool_spec_name(tool).lower(): tool_spec_name(tool) for tool in tools}
    envelopes: list[tuple[int, int, str]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(1).strip()))
    for match in _NAMESPACED_TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(2).strip()))
    if not envelopes:
        bare_calls = _parse_bare_json_tool_calls(text, known)
        if bare_calls is not None:
            return "", bare_calls
        return text, None
    envelopes.sort(key=lambda item: item[0])
    calls: list[dict[str, Any]] = []
    residue_parts: list[str] = []
    cursor = 0
    for index, (start, end, block) in enumerate(envelopes):
        residue_parts.append(text[cursor:start])
        cursor = end
        parsed = _parse_tool_call_payload(block)
        if parsed is None:
            raise ValueError("unsupported tool_call payload format")
        raw_name, arguments = parsed
        if raw_name.lower() not in known:
            # A call to a tool the client did not offer stays in the reply as text:
            # raising here ended the stream, and the agent client retried the same turn in a loop.
            residue_parts.append(text[start:end])
            continue
        calls.append(_openai_tool_call(raw_name, arguments, known))
        if index + 1 < len(envelopes) and envelopes[index + 1][0] < end:
            raise ValueError("overlapping tool_call blocks")
    residue_parts.append(text[cursor:])
    content = "".join(residue_parts).strip()
    return content, calls or None


def stream_tool_call_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            continue
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": str(tool_call.get("id") or f"call_{index}"),
                        "type": str(tool_call.get("type") or "function"),
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": "",
                        },
                    }
                ]
            }
        )
        arguments = str(function.get("arguments") or "")
        if arguments:
            deltas.append({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]})
    return deltas


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_TERMINATORS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>")


def parse_harmony_output(text: str) -> tuple[str, str | None]:
    """Split harmony-format output (GPT-OSS) into (content, reasoning).

    Non-harmony text passes through unchanged with no reasoning. When the
    final channel never arrived (token budget spent inside analysis), content
    is empty and the partial analysis is surfaced as reasoning.
    """

    if "<|channel|>" not in text:
        return text, None

    reasoning: str | None = None
    if HARMONY_ANALYSIS_MARKER in text:
        reasoning = text.split(HARMONY_ANALYSIS_MARKER, 1)[1]
        for terminator in (HARMONY_FINAL_MARKER, *HARMONY_TERMINATORS):
            reasoning = reasoning.split(terminator, 1)[0]

    if HARMONY_FINAL_MARKER not in text:
        return "", reasoning
    content = text.split(HARMONY_FINAL_MARKER, 1)[1]
    for terminator in HARMONY_TERMINATORS:
        content = content.split(terminator, 1)[0]
    return content, reasoning


def streaming_visible_text(text: str) -> str:
    """The part of partially-decoded output that should stream to the client."""

    if "<|channel|>" not in text:
        return text
    content, _ = parse_harmony_output(text)
    return content


def make_handler(app: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[tensorfold] {self.address_string()} {format % args}")

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _route(self) -> str:
            # Tolerate query strings, trailing slashes, and clients that join
            # the base URL differently (with or without the /v1 prefix).
            return self.path.split("?", 1)[0].rstrip("/")

        def do_GET(self) -> None:
            route = self._route()
            if route in {"", "/health"}:
                self._send_json(
                    {
                        "status": "ok",
                        "model": app.served_name,
                        "model_ids": app.model_ids,
                        "max_batch_size": app.max_batch_size,
                    }
                )
                return
            if route.endswith("/models") or route == "/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "tensorfold",
                            }
                            for model_id in app.model_ids
                        ],
                    }
                )
                return
            self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)

        def _legacy_prompt_to_text(self, prompt: Any) -> str:
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list):
                if all(isinstance(token_id, int) for token_id in prompt):
                    with app.tokenizer_lock:
                        return app.tokenizer.decode([int(token_id) for token_id in prompt])
                return "\n".join(self._legacy_prompt_to_text(item) for item in prompt)
            if prompt is None:
                return ""
            return str(prompt)

        def _messages_from_legacy_completion(self, body: dict[str, Any]) -> list[dict[str, Any]]:
            messages = body.get("messages")
            if isinstance(messages, list) and messages:
                return messages
            return [{"role": "user", "content": self._legacy_prompt_to_text(body.get("prompt", ""))}]

        def do_POST(self) -> None:
            route = self._route()
            is_chat_completion = route.endswith("/chat/completions")
            is_text_completion = route.endswith("/completions") and not is_chat_completion
            if not is_chat_completion and not is_text_completion:
                self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if _REQUEST_LOG and body.get("priority") != "background":   # batch jobs are not client traffic
                    with open(_REQUEST_LOG, "a") as handle:
                        handle.write(json.dumps(body) + "\n")
                if is_chat_completion:
                    messages = body.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages must be a non-empty list")
                    tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
                else:
                    messages = self._messages_from_legacy_completion(body)
                    tools = []
                max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
                temperature = float(body.get("temperature") or 0.0)
                # the raw fields, for exact sampling (an absent temperature is not temperature 0), "seed", the thinking
                # budget, "draft": false (the serial reference) and "priority": "background" (yields to every other
                # request)
                sampling_fields = {k: body[k] for k in ("temperature", "top_p", "top_k", "seed", "priority", "draft",
                                                        "thinking_budget")
                                   if k in body}
                template_kwargs = body.get("chat_template_kwargs") or {}
                if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
                    sampling_fields["enable_thinking"] = bool(template_kwargs["enable_thinking"])
                sampling_kw = ({"sampling": sampling_fields}
                               if getattr(app, "accepts_sampling", False) else {})
                stream = bool(body.get("stream", False))
            except Exception as exc:
                self._send_json({"error": {"message": str(exc)}}, status=400)
                return

            completion_id = (
                f"chatcmpl-{uuid.uuid4().hex}"
                if is_chat_completion
                else f"cmpl-{uuid.uuid4().hex}"
            )
            created = int(time.time())

            def usage_from_reply(reply: dict[str, Any]) -> dict[str, Any]:
                return {
                    "prompt_tokens": reply["prompt_tokens"],
                    "completion_tokens": reply["completion_tokens"],
                    "total_tokens": reply["prompt_tokens"] + reply["completion_tokens"],
                    "prompt_tokens_details": {"cached_tokens": reply["cached_tokens"]},
                }

            def response_extras(reply: dict[str, Any]) -> dict[str, Any]:
                extras: dict[str, Any] = {
                    "exact_mode": app.exact_mode.get("mode", "target-verified")
                }
                if reply.get("batch_size"):
                    extras["tensorfold"] = {
                        "batch_size": reply["batch_size"],
                        "seconds": reply["seconds"],
                    }
                if reply.get("runtime"):
                    extras["tensorfold"] = reply["runtime"]
                if reply.get("speculative"):
                    extras["speculative"] = reply["speculative"]
                if reply.get("pass_economics"):
                    extras["pass_economics"] = reply["pass_economics"]
                return extras

            def attach_tool_calls(reply: dict[str, Any]) -> dict[str, Any]:
                if not tools:
                    return reply
                content, tool_calls = parse_tool_calls_from_content(str(reply.get("content") or ""), tools)
                if not tool_calls:
                    return reply
                next_reply = dict(reply)
                next_reply["content"] = content
                next_reply["tool_calls"] = tool_calls
                next_reply["finish_reason"] = "tool_calls"
                return next_reply

            def stream_chunk(
                delta: str | dict[str, Any] = "",
                finish_reason: str | None = None,
            ) -> dict[str, Any]:
                if is_text_completion:
                    return {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": app.served_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "finish_reason": finish_reason,
                                "logprobs": None,
                            }
                        ],
                    }
                return {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": app.served_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta if isinstance(delta, dict) else ({"content": delta} if delta else {}),
                            "finish_reason": finish_reason,
                        }
                    ],
                }

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def emit(payload: dict[str, Any]) -> None:
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    def finish_stream(
                        finish_reason: str | None,
                        *,
                        error: BaseException | None = None,
                        extras: dict[str, Any] | None = None,
                    ) -> None:
                        payload = stream_chunk("", finish_reason or "length")
                        if extras:
                            # Telemetry rides the final chunk so streaming
                            # clients see the same extras as JSON replies.
                            payload.update(extras)
                        if error is not None:
                            payload["tensorfold_error"] = {
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        emit(payload)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()

                    def on_delta(delta: str) -> None:
                        emit(stream_chunk(delta))

                    try:
                        if tools:
                            streamed = [False]

                            def on_prose(delta: str) -> None:
                                if not streamed[0]:
                                    streamed[0] = True
                                    emit(stream_chunk({"role": "assistant"}))
                                emit(stream_chunk(delta))

                            extra = (
                                {"on_delta": on_prose}
                                if getattr(app, "streams_prose_with_tools", False) else {}
                            )
                            reply = attach_tool_calls(
                                app.chat(
                                    messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    tools=tools,
                                    **extra,
                                    **sampling_kw,
                                )
                            )
                            tool_calls = reply.get("tool_calls")
                            if tool_calls and not reply.get("tool_calls_streamed"):
                                # (calls the app already streamed as they were written are not sent twice)
                                emit(stream_chunk({"role": "assistant"}))
                                for delta in stream_tool_call_deltas(tool_calls):
                                    emit(stream_chunk(delta))
                            elif reply.get("content") and not streamed[0]:
                                emit(stream_chunk(str(reply["content"])))
                        else:
                            if is_chat_completion:
                                emit(stream_chunk({"role": "assistant"}))
                            reply = app.chat(
                                messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                on_delta=on_delta,
                                **sampling_kw,
                            )
                    except BrokenPipeError:
                        raise
                    except RequestError as exc:
                        emit({"error": {"message": str(exc), "type": "invalid_request_error"}})
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    except Exception as exc:
                        print(
                            f"[tensorfold] stream error: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()
                        try:
                            finish_stream("stop", error=exc)
                        except BrokenPipeError:
                            pass
                        return
                    extras = response_extras(reply)
                    if "prompt_tokens" in reply and "completion_tokens" in reply:
                        # Clients that time the stream count tokens from here.
                        extras["usage"] = usage_from_reply(
                            {"cached_tokens": 0, **reply})
                    finish_stream(reply.get("finish_reason") or "length", extras=extras)
                    return

                reply = attach_tool_calls(
                    app.chat(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=tools or None,
                        **sampling_kw,
                    )
                )
                if is_text_completion:
                    self._send_json(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": app.served_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": reply["content"],
                                    "finish_reason": reply["finish_reason"],
                                    "logprobs": None,
                                }
                            ],
                            "usage": usage_from_reply(reply),
                            **response_extras(reply),
                        }
                    )
                    return

                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": None if reply.get("tool_calls") else reply["content"],
                }
                if reply.get("reasoning"):
                    message["reasoning_content"] = reply["reasoning"]
                if reply.get("tool_calls"):
                    message["tool_calls"] = reply["tool_calls"]
                self._send_json(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": app.served_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": reply["finish_reason"],
                            }
                        ],
                        "usage": usage_from_reply(reply),
                        **response_extras(reply),
                    }
                )
            except BrokenPipeError:
                pass
            except RequestError as exc:
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
            except Exception as exc:  # surface runner errors to the client
                print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    self._send_json({"error": {"message": str(exc)}}, status=500)
                except Exception:
                    pass

    return Handler
