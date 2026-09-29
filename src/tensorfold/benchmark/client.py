"""Bounded OpenAI-compatible SSE requests, with no output or URL in results."""

from __future__ import annotations

import http.client
import json
import math
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

_MAX_EVENT_BYTES = 512 * 1024
_MAX_STREAM_BYTES = 8 * 1024 * 1024
_HASH = re.compile(r"(?:[0-9a-fA-F]{12}|[0-9a-fA-F]{64})\Z")


class StreamFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _endpoint(base_url: str, kind: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
    ):
        raise ValueError("Server must be an http or https URL without credentials, query or fragment")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    path += "/chat/completions" if kind == "chat" else "/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _socket_deadline(response: Any, deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise StreamFailure("timeout")
    # HTTPConnection can detach its socket for Connection: close replies.
    # HTTPResponse's buffered reader still owns that socket until close().
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(remaining)


def _events(response: Any, deadline: float) -> Iterator[tuple[str, float]]:
    """Join SSE data fields, handling byte boundaries, CRLF and keepalives."""

    pending = bytearray()
    data: list[str] = []
    event_bytes = 0
    total = 0
    while True:
        _socket_deadline(response, deadline)
        block = response.read1(64 * 1024)
        arrived = time.perf_counter()
        if not block:
            # An event is dispatched only by an empty line. Dropping an
            # unterminated final data field must not look like a complete run.
            return
        total += len(block)
        if total > _MAX_STREAM_BYTES:
            raise StreamFailure("invalid_stream")
        pending.extend(block)
        while b"\n" in pending:
            raw, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            if len(raw) > _MAX_EVENT_BYTES:
                raise StreamFailure("invalid_stream")
            try:
                line = raw.rstrip(b"\r").decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StreamFailure("invalid_stream") from exc
            if line == "":
                if data:
                    yield "\n".join(data), arrived
                data, event_bytes = [], 0
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data.append(value)
                event_bytes += len(raw)
                if event_bytes > _MAX_EVENT_BYTES:
                    raise StreamFailure("invalid_stream")
            # Comments and event/id/retry fields carry no completion data.
        if len(pending) > _MAX_EVENT_BYTES:
            raise StreamFailure("invalid_stream")


def _runtime_fields(runtime: dict[str, Any]) -> dict[str, Any]:
    token_sha = runtime.get("token_sha")
    return {
        "server_decode_tps": _number(runtime.get("tokens_per_second", runtime.get("decode_tps"))),
        "server_decode_seconds": _number(runtime.get("decode_seconds", runtime.get("decode_s"))),
        "prefill_seconds": _number(runtime.get("prefill_seconds", runtime.get("prefill_s"))),
        "token_sha": token_sha.lower() if isinstance(token_sha, str) and _HASH.fullmatch(token_sha) else None,
    }


def stream_request(
    base_url: str,
    model_id: str,
    fixture: dict[str, str],
    *,
    tokens: int,
    temperature: float,
    seed: int,
    repeat: int,
    serial: bool = False,
    timeout: float = 600,
) -> dict[str, Any]:
    """Retain failures as samples, and close the response on cancellation.

    A delivery rate requires two separate nonempty content/reasoning events.
    Completion counts come only from usage. Hashes come only from the server's
    token IDs, never from generated text or SSE chunk counts.
    """

    sample: dict[str, Any] = {
        "fixture_id": fixture["id"], "temperature": float(temperature), "repeat": repeat, "seed": seed,
        "status": "error", "prompt_tokens": None, "completion_tokens": None, "cached_tokens": None,
        "delivery_seconds": None, "delivery_tps": None, "ttft_seconds": None, "end_to_end_seconds": None,
        "server_decode_tps": None, "server_decode_seconds": None, "prefill_seconds": None,
        "token_sha": None, "error_code": None,
    }
    body: dict[str, Any] = {
        "model": model_id, "max_tokens": tokens, "temperature": temperature, "seed": seed,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if serial:
        body["draft"] = False
    if fixture["kind"] == "chat":
        body["messages"] = [{"role": "user", "content": fixture["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["prompt"] = fixture["prompt"]
    url = _endpoint(base_url, fixture["kind"])
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start = time.perf_counter()
    first = last = None
    deliveries = 0
    usage = None
    runtime: dict[str, Any] = {}
    done = False
    finished = False
    finish_reason = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise StreamFailure("invalid_stream")
            for event, arrived in _events(response, start + timeout):
                if event.strip() == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(event)
                except (ValueError, TypeError) as exc:
                    raise StreamFailure("invalid_stream") from exc
                if not isinstance(chunk, dict):
                    raise StreamFailure("invalid_stream")
                if chunk.get("error") is not None:
                    raise StreamFailure("server_error")
                if chunk.get("usage") is not None:
                    if not isinstance(chunk["usage"], dict):
                        raise StreamFailure("invalid_usage")
                    usage = chunk["usage"]
                reported = chunk.get("tensorfold", chunk.get("runtime"))
                if isinstance(reported, dict):
                    runtime.update(reported)
                choices = chunk.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise StreamFailure("invalid_stream")
                has_text = False
                for choice in choices:
                    if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                        raise StreamFailure("invalid_stream")
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        raise StreamFailure("invalid_stream")
                    pieces = [choice.get("text"), delta.get("content"), delta.get("reasoning_content")]
                    for piece in pieces:
                        if piece is not None and not isinstance(piece, str):
                            raise StreamFailure("invalid_stream")
                    has_text = any(pieces)
                    if finished and has_text:
                        raise StreamFailure("invalid_stream")
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if not isinstance(reason, str) or finished:
                            raise StreamFailure("invalid_stream")
                        finished, finish_reason = True, reason
                if has_text:
                    first = arrived if first is None else first
                    last = arrived
                    deliveries += 1
            if not done:
                raise StreamFailure("truncated_stream")
            if not finished:
                raise StreamFailure("unexpected_finish")
            if finish_reason not in {"stop", "length"}:
                raise StreamFailure("unexpected_finish")
            if usage is None:
                raise StreamFailure("missing_usage")
            prompt_tokens = _count(usage.get("prompt_tokens"))
            completion_tokens = _count(usage.get("completion_tokens"))
            if prompt_tokens is None or completion_tokens is None:
                raise StreamFailure("invalid_usage")
            sample.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            details = usage.get("prompt_tokens_details")
            cached = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
            if cached is None:
                cached = _count(runtime.get("cached_tokens", runtime.get("cached")))
            if cached is not None and cached > prompt_tokens:
                raise StreamFailure("invalid_usage")
            sample["cached_tokens"] = cached
            if first is None or completion_tokens == 0:
                raise StreamFailure("empty_output")
            if completion_tokens > tokens:
                raise StreamFailure("token_count_mismatch")
            if completion_tokens < tokens:
                sample["status"] = "early_eos"
            elif deliveries < 2 or last is None or last <= first:
                sample["status"] = "unmeasured"
                sample["error_code"] = "unmeasured_delivery"
            else:
                sample["status"] = "ok"
    except StreamFailure as exc:
        error = exc.code
    except KeyboardInterrupt:
        error = "cancelled"
    except urllib.error.HTTPError as exc:
        exc.close()
        error = "http_error"
    except (TimeoutError, socket.timeout):
        error = "timeout"
    except urllib.error.URLError as exc:
        error = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "connection_error"
    except (OSError, EOFError):
        error = "connection_error"
    except http.client.HTTPException:
        error = "truncated_stream"
    finally:
        sample["end_to_end_seconds"] = max(0.0, time.perf_counter() - start)
        sample.update(_runtime_fields(runtime))
        if first is not None:
            sample["ttft_seconds"] = max(0.0, first - start)
        if deliveries >= 2 and last is not None and first is not None and last > first:
            sample["delivery_seconds"] = last - first
            if sample["completion_tokens"] is not None and sample["completion_tokens"] > 1:
                sample["delivery_tps"] = (sample["completion_tokens"] - 1) / (last - first)
        if error is not None:
            sample.update(status="error", error_code=error)
    return sample
