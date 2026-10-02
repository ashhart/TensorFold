"""Anthropic HTTP routes on both servers, over their own chat-completions handler and connection."""
from __future__ import annotations

import json
from typing import Any

from tensorfold.server.anthropic_translate import Reply, error, translate
from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.server.responses import LIMIT, Wire, _run_chat, _send


def route(path: str) -> bool:
    return path.split("?", 1)[0].rstrip("/") in ("/v1/messages", "/messages", "/v1/messages/count_tokens",
                                                 "/messages/count_tokens")


def count_tokens(app: Any, chat: dict[str, Any]) -> int:
    """Render with the same tokenizer, tools and thinking controls as chat, without running generation."""
    if hasattr(app, "prepare"):                   # CUDA's preparation includes vision token expansion
        return len(app.prepare(chat, True).prompt)
    from tensorfold.server.messages import normalize_messages
    from tensorfold.server.prompts import has_images, prepare_images
    from tensorfold.server.request_options import heard_effort, thinking_fields
    from tensorfold.server.text import render_prompt_ids
    from tensorfold.server.tools import active_tool_specs

    fields = thinking_fields(chat, app.effort_levels)
    thinking = fields.get("enable_thinking", app.enable_thinking)
    effort = heard_effort(fields.get("reasoning_effort"), app.reasoning_effort, app.effort_levels)
    tools = active_tool_specs(chat.get("tools"), chat.get("tool_choice"))
    msgs = normalize_messages(chat["messages"], allow_images=getattr(app, "vision", None) is not None)
    if has_images(msgs):
        from tensorfold.server.messages import _normalize_tool_call_arguments
        msgs = _normalize_tool_call_arguments(msgs)

        def render(messages):
            kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
            if tools:
                kwargs["tools"] = tools
            if thinking and effort:
                kwargs["reasoning_effort"] = effort
            with app.tokenizer_lock:
                return app.tokenizer.apply_chat_template(messages, **kwargs)

        return len(prepare_images(app.vision, msgs, render, context_limit=app.context_window or None).tokens)
    with app.tokenizer_lock:
        return len(render_prompt_ids(app.tokenizer, msgs, tools=tools, enable_thinking=thinking,
                                     reasoning_effort=effort, late_system=app.late_system))


def post(handler: Any, app: Any) -> None:
    from tensorfold.server.http import reply_model

    count = handler.path.split("?", 1)[0].rstrip("/").endswith("/count_tokens")
    try:
        try:
            length = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            handler.close_connection = True
            raise RequestError("Content-Length must be an integer") from None
        if not 0 <= length <= LIMIT:
            handler.close_connection = True
            raise RequestError("request body exceeds the 32 MiB limit")
        raw = handler.rfile.read(length)
        body = json.loads(raw or b"{}")
        chat = translate(body, count=count)
        late_system = getattr(app, "late_system", getattr(getattr(app, "template", None), "late_system", "system"))
        if late_system != "system" and any(m.get("role") == "system" for m in body["messages"]):
            raise RequestError("the model's chat template does not support mid-conversation system messages")
        if count:
            return _send(handler, 200, {"input_tokens": count_tokens(app, chat)})
    except (RequestError, ValueError, UnicodeDecodeError) as exc:
        return _send(handler, 503 if isinstance(exc, CapacityError) else 400,
                     error(str(exc), 503 if isinstance(exc, CapacityError) else 400))

    def send(event: dict[str, Any]) -> None:
        handler.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        handler.wfile.flush()

    reply = Reply(reply_model(app, body), send)

    def opened() -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        reply.start()

    wire = Wire(opened, reply.chunk)
    try:
        _run_chat(handler, chat, wire)
        if wire.stream or wire.status is None:
            return
        payload = json.loads(bytes(wire.data) or b"{}")
        if wire.status != 200:
            problem = payload.get("error") or {}
            message = problem.get("message", "request failed") if isinstance(problem, dict) else str(problem)
            return _send(handler, wire.status, error(message, wire.status))
        _send(handler, 200, reply.completion(payload))
    except OSError:
        handler.close_connection = True
    except (ValueError, KeyError, TypeError) as exc:
        if wire.stream:
            send(error(f"invalid model response: {exc}", 500))
        else:
            _send(handler, 500, error(f"invalid model response: {exc}", 500))
