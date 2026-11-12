"""Bench stream."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field

import httpx

from . import bench, httpread


@dataclass
class Reply:
    text: str = ""
    reasoning: str = ""
    finish: str | None = None
    usage: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    sent_at: float = 0.0
    first_at: float | None = None                         # first token of any kind, reasoning included
    answer_at: float | None = None                        # first token of the answer itself
    end_at: float = 0.0
    gaps: list[float] = field(default_factory=list)       # seconds between content events after the first
    events: int = 0
    error: str | None = None


def text_digest(text: str) -> str:
    """The whole reply's digest: the stored text stops at 4,000 characters, the comparison must not."""

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stream_once(client: httpx.Client, url: str, headers: dict, body: dict, cancel: threading.Event,
                on_text=None) -> Reply:

    rep = Reply(sent_at=time.perf_counter())
    opened: list[httpx.Response] = []
    try:
        httpread.until_cancelled(lambda: _stream(client, url, headers, body, cancel, rep, opened, on_text), cancel,
                                 close=lambda: opened and opened[0].close(), name="bench-reply")
    except InterruptedError:
        rep.error = "cancelled"
        rep.end_at = time.perf_counter()
    return rep


def _stream(client: httpx.Client, url: str, headers: dict, body: dict, cancel: threading.Event, rep: Reply,
            opened: list, on_text=None) -> None:
    last = None
    done = False
    try:
        with client.stream("POST", url, headers=headers, json=body) as r:
            opened.append(r)
            if r.status_code != 200:
                # Error bodies are limited to 4 KB and 10 seconds; cancellation must also release the run's gate.
                text = httpread.error_text(r, cancel=cancel)
                try:
                    msg = json.loads(text).get("error", {}).get("message") or text
                except (ValueError, AttributeError):
                    msg = text
                rep.error = f"{r.status_code}: {str(msg)[:300]}"
                rep.end_at = time.perf_counter()
                return
            buf = ""
            for chunk in r.iter_text():
                if cancel.is_set():
                    rep.error = "cancelled"
                    break
                # normalised on what has gathered, not chunk by chunk: a CRLF can arrive split between two chunks
                buf = (buf + chunk).replace("\r\n", "\n")
                while "\n\n" in buf:
                    event, buf = buf.split("\n\n", 1)
                    data = "\n".join(line[5:].strip() for line in event.split("\n") if line.startswith("data:"))
                    if data == "[DONE]":
                        done = True
                        # the reply is whole: a server keeping the line open is not waited for
                        break
                    if not data:
                        continue
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        continue
                    # an engine failing mid-reply says so in the stream
                    if isinstance(msg.get("error"), (dict, str)):
                        err = msg["error"]
                        rep.error = "the engine stopped: " + str(err.get("message") if isinstance(err, dict) else err)[:280]
                        break
                    if msg.get("usage"):
                        rep.usage = msg["usage"]
                    if isinstance(msg.get("tensorfold"), dict):
                        rep.runtime = msg["tensorfold"]
                    if isinstance(msg.get("speculative"), dict):  # the native engine's draft counters sit beside it
                        rep.runtime = {**rep.runtime, **msg["speculative"]}
                    for choice in msg.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if piece or think:
                            now = time.perf_counter()
                            rep.first_at = rep.first_at or now
                            if last is not None:
                                rep.gaps.append(now - last)
                            last = now
                            rep.events += 1
                            if piece and rep.answer_at is None:
                                rep.answer_at = now
                            rep.text += piece
                            rep.reasoning += think
                            if on_text is not None:          # a page watching the reply being written
                                on_text(rep)
                        if choice.get("finish_reason"):
                            rep.finish = choice["finish_reason"]
                if rep.error or done:                       # [DONE] ends the reading: the response is closed here
                    break
    except httpx.HTTPError as exc:
        rep.error = f"{type(exc).__name__}: {exc}"[:300]
    if rep.error is None and rep.finish is None and not done:
        # a stream cut before its end: the reply is not whole, so it counts neither for speed nor for exactness
        rep.error = "the reply ended before the engine finished it (no finish_reason, no [DONE])"
    rep.end_at = time.perf_counter()


def drafts_off_confirmed(runtime: dict) -> bool:

    drafts, drafted = runtime.get("drafts"), runtime.get("drafted")
    if drafts is True or (isinstance(drafted, (int, float)) and drafted > 0):
        return False
    return drafts is False or drafted == 0


def drafts_on_confirmed(runtime: dict) -> bool:

    drafts, drafted = runtime.get("drafts"), runtime.get("drafted")
    if drafts is False:
        return False
    if isinstance(drafted, (int, float)):
        return drafted > 0
    return drafts is True


def thought(rep: Reply) -> bool:
    """Whether a reply carried reasoning, as its own field or inline in a think tag."""

    return bool(rep.reasoning.strip()) or "<think>" in rep.text or "</think>" in rep.text


def measure(rep: Reply) -> dict:
    """The figures of one reply: the engine's own when its block has them, this TensorFold Web's clock otherwise."""

    rt = {k: rep.runtime[k] for k in bench.RUNTIME_KEYS if k in rep.runtime}
    tokens = rep.usage.get("completion_tokens")
    ttft_client = (rep.first_at - rep.sent_at) if rep.first_at else None
    decode_client = (rep.end_at - rep.first_at) if rep.first_at else None
    # Without engine token counts, stream events remain events and cannot establish a token rate.
    tps_client = tokens / decode_client if tokens and decode_client and decode_client > 0 else None
    from_engine = rt.get("tokens_per_second") is not None
    return {
        "tokens": tokens,
        "tokens_counted_by": "engine" if tokens is not None else None,
        "prompt_tokens": rep.usage.get("prompt_tokens"),
        "cached_tokens": (rep.usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (rep.usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "tps": rt["tokens_per_second"] if from_engine else tps_client,
        "ttft": rt.get("time_to_first_token", ttft_client),
        # The page times the first answer token separately from reasoning because the engine does not report it.
        "ttft_answer": (rep.answer_at - rep.sent_at) if rep.answer_at and rep.reasoning else None,
        "prefill_s": rt.get("prefill_seconds"),
        "timing": "engine" if from_engine else "studio",
        "tps_client": tps_client, "ttft_client": ttft_client,
        "gap_p50": bench.percentile(rep.gaps, 0.5), "gap_p95": bench.percentile(rep.gaps, 0.95), "events": rep.events,
        "tokens_per_event": (tokens / rep.events) if tokens and rep.events else None,
        "wall_s": rep.end_at - rep.sent_at, "finish": rep.finish, "sha": rt.get("token_sha"), "runtime": rt,
    }
