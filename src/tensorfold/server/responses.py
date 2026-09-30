"""OpenAI's Responses API on both servers: each request runs as its chat completion on the server's own handler."""

from __future__ import annotations

import copy
import http.client
import io
import json
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from tensorfold.server.errors import RequestError
from tensorfold.server.responses_translate import Reply, _id, messages, translate

LIMIT = 32 * 1024**2


def route(path: str) -> str | None:
    """A ``/v1/responses`` path's response id ("" for the collection), else None."""

    path = path.split("?", 1)[0].rstrip("/")
    for prefix in ("/v1/responses", "/responses"):
        if path == prefix:
            return ""
        rest = path[len(prefix) + 1:] if path.startswith(prefix + "/") else ""
        if rest and "/" not in rest:
            return rest
    return None


# -- store ------------------------------------------------------------------------------------


@dataclass
class _Entry:
    response: dict[str, Any]
    parent: str | None
    messages: list[dict[str, Any]]                 # what this response added: its input, then its output
    size: int = field(default=0)


class Store:
    """The newest stored responses (``store`` true), in memory, for GET, DELETE and ``previous_response_id``."""

    def __init__(self, limit: int = 1024, max_bytes: int = 256 * 1024**2) -> None:
        self.limit, self.max_bytes = limit, max_bytes
        self.entries: OrderedDict[str, _Entry] = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def put(self, response: dict[str, Any], added: list[dict[str, Any]]) -> None:
        entry = _Entry(copy.deepcopy(response), response.get("previous_response_id"),
                       added + messages([item for item in response["output"]]))
        entry.size = len(json.dumps(entry.response)) + len(json.dumps(entry.messages))
        with self.lock:
            self.entries[response["id"]] = entry
            self.bytes += entry.size
            while self.entries and (len(self.entries) > self.limit or self.bytes > self.max_bytes):
                self.bytes -= self.entries.popitem(last=False)[1].size

    def get(self, rid: str) -> dict[str, Any] | None:
        with self.lock:
            entry = self.entries.get(rid)
            return copy.deepcopy(entry.response) if entry is not None else None

    def delete(self, rid: str) -> bool:
        with self.lock:
            entry = self.entries.pop(rid, None)
            if entry is not None:
                self.bytes -= entry.size
            return entry is not None

    def conversation(self, rid: Any) -> list[dict[str, Any]]:
        """Every message of a stored response's chain, oldest first (the chain stays newest in the store)."""

        if not isinstance(rid, str):
            raise RequestError("previous_response_id must be a string")
        chain: list[list[dict[str, Any]]] = []
        with self.lock:
            at: str | None = rid
            while at is not None:
                entry = self.entries.get(at)
                if entry is None:
                    raise RequestError(f"previous response {at!r} is not stored here (store: false, deleted, "
                                       "or evicted): send the conversation's items instead")
                self.entries.move_to_end(at)
                chain.append(entry.messages)
                at = entry.parent
        return copy.deepcopy([m for part in reversed(chain) for m in part])


_STORES: "weakref.WeakKeyDictionary[Any, Store]" = weakref.WeakKeyDictionary()
_STORES_LOCK = threading.Lock()


def store_for(app: Any) -> Store:
    with _STORES_LOCK:
        return _STORES.setdefault(app, Store())


# -- HTTP -------------------------------------------------------------------------------------


def _send(handler: Any, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except OSError:                                  # the client has gone
        handler.close_connection = True


def _refuse(handler: Any, message: str, status: int = 400, param: str | None = None) -> None:
    _send(handler, status, {"error": {"message": message, "type": "invalid_request_error", "param": param,
                                      "code": None}})


class Wire:
    """The chat handler's side of its socket: the status and headers, then the JSON body or each stream event."""

    def __init__(self, opened: Callable[[], None], event: Callable[[dict[str, Any] | None], None]) -> None:
        self.opened, self.event = opened, event
        self.status: int | None = None
        self.stream = False
        self.data = bytearray()

    def write(self, data: bytes) -> int:
        self.data += data
        if self.status is None:
            end = self.data.find(b"\r\n\r\n")
            if end < 0:
                return len(data)
            head = bytes(self.data[:end]).decode("latin-1").split("\r\n")
            del self.data[:end + 4]
            self.status = int(head[0].split()[1])
            kinds = [v.strip() for k, _, v in (line.partition(":") for line in head[1:]) if k.lower() == "content-type"]
            self.stream = self.status == 200 and bool(kinds) and kinds[0].startswith("text/event-stream")
            if self.stream:
                self.opened()
        while self.stream and (end := self.data.find(b"\n\n")) >= 0:
            block = bytes(self.data[:end])
            del self.data[:end + 2]
            for line in block.split(b"\n"):
                if line.startswith(b"data: "):
                    try:
                        self.event(None if line[6:] == b"[DONE]" else json.loads(line[6:]))
                    except OSError as exc:           # our client has gone: the chat handler's write fails
                        raise BrokenPipeError(str(exc)) from exc
        return len(data)

    def flush(self) -> None:
        pass


def _run_chat(handler: Any, body: dict[str, Any], wire: Wire) -> None:
    """The server's own chat-completions handler on ``body``, over this connection, its response written to ``wire``."""

    data = json.dumps(body).encode()
    inner = type(handler).__new__(type(handler))
    inner.__dict__.update(handler.__dict__)
    inner.path, inner.rfile, inner.wfile, inner._headers_buffer = "/v1/chat/completions", io.BytesIO(data), wire, []
    inner.headers = http.client.HTTPMessage()
    inner.headers["Content-Type"] = "application/json"
    inner.headers["Content-Length"] = str(len(data))
    inner.log_request = lambda *args: None          # the request logs once, as itself
    inner.do_POST()


def post(handler: Any, app: Any) -> None:
    """POST /v1/responses."""

    from tensorfold.server.http import reply_model     # http imports this module

    store = store_for(app)
    try:
        length = int(handler.headers.get("Content-Length") or 0)
        if not 0 <= length <= LIMIT:
            handler.close_connection = True           # the unread body must not reach the next request
            raise RequestError("request body exceeds the 32 MiB limit")
        try:
            body = json.loads(handler.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise RequestError("the request body is not JSON") from None
        request = translate(body, store)
    except (RequestError, ValueError) as exc:
        return _refuse(handler, str(exc))
    base = {"id": _id("resp"), "object": "response", "created_at": int(time.time()), "status": "in_progress",
            "error": None, "incomplete_details": None, "model": reply_model(app, body), "output": [], "usage": None,
            **request.echo}

    def send(event: dict[str, Any]) -> None:
        handler.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        handler.wfile.flush()

    reply = Reply(base, send if request.stream else None,
                  (lambda final: store.put(final, request.added)) if request.store else None)

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
    except OSError:                                   # the client left while the reply was written
        handler.close_connection = True
        return
    if wire.stream:                                   # the events are written (none more if the client left)
        return
    if wire.status is None:                           # the client left before the reply
        handler.close_connection = True
        return
    if wire.status != 200:                            # refused as a chat completion: the same refusal
        try:
            payload = json.loads(bytes(wire.data) or b"{}")
        except ValueError:
            payload = {"error": {"message": bytes(wire.data).decode("utf-8", "replace")}}
        return _send(handler, wire.status, payload)
    _send(handler, 200, reply.completion(json.loads(bytes(wire.data))))


def get(handler: Any, app: Any, rid: str | None) -> None:
    """GET /v1/responses/{id}."""

    found = store_for(app).get(rid) if rid else None
    if found is None:
        return _refuse(handler, f"no stored response has id {rid!r}", 404, "response_id")
    _send(handler, 200, found)


def delete(handler: Any, app: Any, rid: str | None) -> None:
    """DELETE /v1/responses/{id}."""

    if not rid or not store_for(app).delete(rid):
        return _refuse(handler, f"no stored response has id {rid!r}", 404, "response_id")
    _send(handler, 200, {"id": rid, "object": "response", "deleted": True})
