"""POST /v1/messages: Anthropic's Messages API on both servers, run as this server's own chat completion.

The same shape as ``responses``: the request is translated to a chat completion, the chat handler runs
it over a ``Wire`` (a fake socket that captures its status, headers and stream events), and the reply is
translated back — non-streamed as one Messages object, streamed as Anthropic's SSE events.
"""

from __future__ import annotations

import json
import time
from typing import Any

from tensorfold.server.anthropic_translate import Reply, _id, translate
from tensorfold.server.errors import RequestError
from tensorfold.server.responses import Wire, _run_chat, _send

LIMIT = 32 * 1024**2


def route(path: str) -> bool:
    """A ``/v1/messages`` path (with or without the /v1 prefix, query strings, trailing slashes)."""

    return path.split("?", 1)[0].rstrip("/") in ("/v1/messages", "/messages")


def _refuse(handler: Any, message: str, status: int = 400) -> None:
    _send(handler, status, {"type": "error", "error": {"type": "invalid_request_error", "message": message}})


def post(handler: Any, app: Any) -> None:
    """POST /v1/messages."""

    from tensorfold.server.http import reply_model     # http imports this module

    try:
        length = int(handler.headers.get("Content-Length") or 0)
        if not 0 <= length <= LIMIT:
            handler.close_connection = True            # the unread body must not reach the next request
            raise RequestError("request body exceeds the 32 MiB limit")
        try:
            body = json.loads(handler.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise RequestError("the request body is not JSON") from None
        request = translate(body)
    except (RequestError, ValueError) as exc:
        return _refuse(handler, str(exc))
    base = {"id": _id("msg"), "type": "message", "role": "assistant", "model": reply_model(app, body),
            "content": []}

    def send(event: dict[str, Any]) -> None:
        handler.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        handler.wfile.flush()

    reply = Reply(base, send if request.stream else None)

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
        _run_chat(handler, request.chat, wire)
    except OSError:                                    # the client left while the reply was written
        handler.close_connection = True
        return
    if wire.stream:                                    # the events are written (none more if the client left)
        return
    if wire.status is None:                            # the client left before the reply
        handler.close_connection = True
        return
    if wire.status != 200:                             # refused as a chat completion: the same refusal, Anthropic's shape
        try:
            payload = json.loads(bytes(wire.data) or b"{}")
        except ValueError:
            payload = {"error": {"message": bytes(wire.data).decode("utf-8", "replace")}}
        message = (payload.get("error") or {}).get("message") or json.dumps(payload)
        return _send(handler, wire.status, {"type": "error",
                                            "error": {"type": "invalid_request_error", "message": message}})
    _send(handler, 200, reply.completion(json.loads(bytes(wire.data))))
