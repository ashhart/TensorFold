"""Raw HTTP exchanges for the parity tests, and the normalisation that leaves only ids and timings out."""

from __future__ import annotations

import json
import re
import socket
from typing import Any

IDS = re.compile(rb"(?:chatcmpl-|cmpl-|resp_|msg_|rs_|fc_|call_)[0-9a-f]{32}|call_[0-9a-f]{24}")
STAMPS = re.compile(rb'"(created|created_at|completed_at)": \d+')
TIMES = re.compile(rb'"(tokens_per_second|seconds|prefill_seconds|time_to_first_token)": -?[0-9.]+(?:e[-+]?\d+)?')
BUCKET = re.compile(rb'^tensorfold:\w+_bucket\{le="(?!\+Inf)[^"]*"\} \d+\n', re.M)
SUMS = re.compile(rb"^(tensorfold:\w+_sum) \S+$", re.M)
FOOTPRINT = re.compile(rb"^(tensorfold:process_footprint_bytes) \d+$", re.M)


def request(method: str, path: str, body: Any = None, headers: dict[str, str] | None = None,
            version: str = "HTTP/1.1", chunked: bool = False) -> bytes:
    """One request's bytes: a JSON body (or raw bytes), fixed length or chunked a byte at a time."""
    data = body if isinstance(body, bytes) else b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
    head = [f"{method} {path} {version}", "Host: parity"] if version else [f"{method} {path}"]
    for k, v in (headers or {}).items():
        head.append(f"{k}: {v}")
    if body is not None and chunked:
        head.append("Transfer-Encoding: chunked")
        data = b"".join(b"1\r\n" + data[i:i + 1] + b"\r\n" for i in range(len(data))) + b"0\r\n\r\n"
    elif body is not None and not any(k.lower() == "content-length" for k in (headers or {})):
        head.append(f"Content-Length: {len(data)}")
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + data


def _response(reader: Any) -> dict[str, Any]:
    first = reader.readline()
    interim = b""
    while first.startswith(b"HTTP/1.1 100"):
        interim += first + reader.readline()
        first = reader.readline()
    if not first.startswith(b"HTTP/"):
        return {"status": "", "headers": [], "body": first + reader.read(), "interim": interim}
    headers = []
    while (line := reader.readline()) not in (b"\r\n", b"\n", b""):
        name, _, value = line.decode("latin-1").rstrip("\r\n").partition(": ")
        headers.append((name, value))
    length = next((int(v) for k, v in headers if k.lower() == "content-length"), None)
    body = reader.read(length) if length is not None else reader.read()
    return {"status": first.decode("latin-1").rstrip("\r\n"), "headers": headers, "body": body, "interim": interim}


def exchange(port: int, raws: list[Any], *, half_close: bool = False, timeout: float = 15.0,
             leave_after_events: int | None = None) -> dict[str, Any]:
    """Each request in turn on one connection (a callable gets the replies so far; b"" reads one more), then whether it closed."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    reader = sock.makefile("rb")
    replies: list[dict[str, Any]] = []
    try:
        for raw in raws:
            data = raw(replies) if callable(raw) else raw
            if data:
                sock.sendall(data)
            if leave_after_events is not None:
                seen = b""
                while seen.count(b"data: ") < leave_after_events and (got := sock.recv(4096)):
                    seen += got
                return {"replies": [], "closed": "left"}
            if half_close:
                sock.shutdown(socket.SHUT_WR)
            replies.append(_response(reader))
        sock.settimeout(0.3)
        try:
            closed = "closed" if sock.recv(1) == b"" else "data"
        except OSError:
            closed = "open"
        return {"replies": replies, "closed": closed}
    finally:
        sock.close()


def normal(reply: dict[str, Any]) -> dict[str, Any]:
    """A reply with ids, timestamps and timings replaced, and Server and Date dropped."""
    seen: dict[bytes, bytes] = {}

    def name(m: re.Match[bytes]) -> bytes:
        return seen.setdefault(m.group(0), b"<id%d>" % (len(seen) + 1))

    body = STAMPS.sub(rb'"\1": <ts>', IDS.sub(name, reply["body"]))
    varied = any(p.search(body) for p in (TIMES, BUCKET, SUMS, FOOTPRINT))
    body = TIMES.sub(rb'"\1": <t>', body)
    body = SUMS.sub(rb"\1 <sum>", FOOTPRINT.sub(rb"\1 <n>", BUCKET.sub(b"", body)))
    headers = [(k, "<len>" if k.lower() == "content-length" and varied else v)
               for k, v in reply["headers"] if k.lower() not in ("server", "date")]
    return {"status": reply["status"], "headers": headers, "body": body.decode("utf-8", "replace"),
            "interim": reply["interim"].decode("latin-1")}


def framed(reply: dict[str, Any]) -> bool:
    """A fixed-length reply's Content-Length matches its body."""
    length = next((int(v) for k, v in reply["headers"] if k.lower() == "content-length"), None)
    return length is None or length == len(reply["body"])
