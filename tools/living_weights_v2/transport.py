from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class TransportError(RuntimeError):
    pass


SSE_LINE_LIMIT = 1024 * 1024
SSE_STREAM_LIMIT = 4 * 1024 * 1024
SSE_EVENT_LIMIT = 4096
# The native graph's fact states (zig/src/server/slide_graph.zig State); anything else is an unknown schema.
GRAPH_STATES = frozenset({"queued", "learning", "learned", "missed", "replaced", "deleting", "deleted"})


def checked_graph(facts) -> dict[int, dict]:
    if not isinstance(facts, (list, tuple)):
        raise TransportError("unrecognized graph schema: missing facts array")
    result = {}
    for fact in facts:
        if not isinstance(fact, dict) or type(fact.get("id")) is not int or fact["id"] <= 0:
            raise TransportError("unrecognized graph fact")
        if type(fact.get("state")) is not str or fact["state"] not in GRAPH_STATES:
            raise TransportError(f"graph fact {fact['id']} has an unrecognized state")
        if fact["id"] in result:
            raise TransportError("duplicate graph fact ID")
        result[fact["id"]] = fact
    return result


@dataclass(frozen=True)
class Event:
    kind: str
    data: dict[str, Any]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Every request carries a key for the configured origin, so redirects must fail before another request.
        return None


class Client:
    def __init__(self, base_url: str, api_key_file: Path, allow_remote: bool = False, timeout: float = 600):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise TransportError("base URL must be an HTTP(S) origin with no path, query, or credentials")
        if not allow_remote:
            try:
                if not ipaddress.ip_address(parsed.hostname).is_loopback:
                    raise TransportError("only literal loopback addresses allowed without --allow-remote")
            except ValueError as exc:
                raise TransportError("use a literal loopback IP or explicitly allow remote access") from exc
        try:
            key = api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TransportError(f"cannot read key file: {exc.strerror}") from exc
        if not key or "\n" in key or "\r" in key:
            raise TransportError("API key file must contain one nonempty line")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
        # Loopback requests never go through an ambient http_proxy/https_proxy, which would receive the key.
        handlers = [_RejectRedirects()] if allow_remote else [urllib.request.ProxyHandler({}), _RejectRedirects()]
        self._opener = urllib.request.build_opener(*handlers)

    def _open(self, path: str, method: str = "GET", payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            return self._opener.open(request, timeout=self.timeout)
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise TransportError(f"HTTP {exc.code} from {path}") from exc
            raise TransportError(f"request to {path} failed: {type(exc).__name__}") from exc

    def json(self, path: str, method: str = "GET", payload: dict | None = None) -> dict:
        with self._open(path, method, payload) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise TransportError("JSON response exceeds 4 MiB")
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("expected JSON object")
            return result
        except (ValueError, UnicodeError) as exc:
            raise TransportError("malformed JSON response") from exc

    def events(self, path: str, method: str, payload: dict) -> Iterator[Event]:
        received_done = False
        kept = 0
        count = 0
        with self._open(path, method, payload) as response:
            kind = "message"
            lines = []
            while binary_line := response.readline(SSE_LINE_LIMIT + 1):
                if len(binary_line) > SSE_LINE_LIMIT:
                    raise TransportError("oversized SSE line")
                try:
                    line = binary_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise TransportError("invalid UTF-8 in SSE") from exc
                if not line:
                    if not lines:
                        kind = "message"
                        continue
                    content = "\n".join(lines)
                    lines = []
                    if content == "[DONE]":
                        received_done = True
                        break
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as exc:
                        raise TransportError("malformed SSE JSON") from exc
                    if not isinstance(parsed, dict):
                        raise TransportError("SSE payload must be a JSON object")
                    kept += len(content)
                    count += 1
                    if kept > SSE_STREAM_LIMIT or count > SSE_EVENT_LIMIT:
                        raise TransportError("SSE stream exceeds 4 MiB or 4096 events")
                    actual_kind = str(parsed.get("event", kind)) if kind == "message" else kind
                    yield Event(actual_kind, parsed)
                    kind = "message"
                elif line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    lines.append(line[5:].lstrip(" "))
                    if sum(map(len, lines)) > SSE_LINE_LIMIT:
                        raise TransportError("oversized SSE event")
            if lines and "\n".join(lines) == "[DONE]":
                received_done = True
        if not received_done:
            raise TransportError("SSE stream ended without [DONE]")

    def chat(self, model: str, question: str) -> str:
        result = self.json("/v1/chat/completions", "POST", {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0,
            "max_tokens": 128,
        })
        try:
            value = result["choices"][0]["message"]["content"]
            if not isinstance(value, str):
                raise TypeError("content is not a string")
            return value
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError("chat response does not contain string content") from exc

    def models(self) -> set[str]:
        response = self.json("/v1/models")
        try:
            return {str(entry["id"]) for entry in response["data"]}
        except (TypeError, ValueError, KeyError) as exc:
            raise TransportError("invalid model listing") from exc

    def graph(self) -> dict[int, dict]:
        return checked_graph(self.json("/v1/slide/graph").get("facts"))

    def teach(self, text: str) -> tuple[Event, ...]:
        return tuple(self.events("/v1/slide/learn", "POST", {"text": text}))

    def delete(self, ids: list[int], reason: str) -> tuple[Event, ...]:
        if not ids or len(ids) > 256 or len(ids) != len(set(ids)) or any(type(i) is not int or i <= 0 for i in ids):
            raise ValueError("invalid deletion IDs")
        return tuple(self.events("/v1/slide/facts", "DELETE", {"ids": ids, "reason": reason}))
