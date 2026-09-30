"""OpenAI-compatible model, health, chat and completion endpoints with streaming, tool calls and reasoning text."""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tensorfold.engine import grammar
from tensorfold.server import responses
from tensorfold.server.tools import (active_tool_specs, parse_tool_calls_from_content, stream_tool_call_deltas,
                                     tool_choice_requires_call)
from tensorfold.server.errors import CapacityError, RequestError, error_body
from tensorfold.server.request_options import parse_numbers, thinking_fields
from tensorfold.server.probabilities import probability_options
from tensorfold.server.messages import normalize_messages, validate_modalities
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server import metrics
from tensorfold.server.stacks import Rearming

# TENSORFOLD_REQUEST_LOG=path appends every request body (one JSON a line), for exact replays of real traffic
_REQUEST_LOG = os.environ.get("TENSORFOLD_REQUEST_LOG", "")


def _memory(reset_peak: bool, *, admission: Any = None) -> dict[str, int]:
    """MLX's memory in bytes: live buffers, its cache of freed ones, and the peak (since the last reset)."""

    if admission is not None:
        return admission.memory_snapshot(reset_peak)
    try:
        import mlx.core as mx
    except ImportError:          # the CUDA server
        return {}
    memory = {"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory()),
              "peak": int(mx.get_peak_memory())}
    if reset_peak:
        mx.reset_peak_memory()
    return memory


class Server(ThreadingHTTPServer):
    """One thread a connection; the listen backlog takes a burst of clients connecting at once."""

    request_queue_size = 128


def redact_images(value: Any) -> Any:
    """A request body for the request log: every image part's URL or data replaced, the rest kept."""

    if isinstance(value, list):
        return [redact_images(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "image_url":
        return {**value, "image_url": {"url": "<redacted>"}}
    return {key: redact_images(item) for key, item in value.items()}


def reply_model(app: Any, body: Any) -> str:
    """The id a reply names: the one the request asked for when this endpoint answers to it, else the served name."""

    asked = body.get("model") if isinstance(body, dict) else None
    name = str(getattr(app, "served_name", "") or getattr(app, "served", "") or "")
    return asked if isinstance(asked, str) and asked in (getattr(app, "model_ids", None) or [name]) else name


def served_model_ids(served_name: str, aliases: list[str] | None = None) -> list[str]:
    """Return the OpenAI model ids this endpoint advertises."""

    ids: list[str] = []
    for value in [served_name, *(aliases or [])]:
        model_id = str(value or "").strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids


def make_handler(app: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(Rearming):              # USR1's stack dump armed again after each request
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
            # Tolerate query strings, trailing slashes and client URLs with or without the /v1 prefix.
            return self.path.split("?", 1)[0].rstrip("/")

        def do_GET(self) -> None:
            route = self._route()
            if responses.route(route):
                return responses.get(self, app, responses.route(route))
            if route in {"/metrics", "/v1/metrics"}:
                return metrics.send(self, app)
            if route in {"", "/health"}:
                self._send_json(
                    {
                        "status": "ok",
                        "model": app.served_name,
                        "model_ids": app.model_ids,
                        "max_batch_size": app.max_batch_size,
                        "warming": bool(getattr(app, "warming", False)),
                        "memory": _memory("reset_peak=1" in self.path, admission=getattr(app, "prompt_memory", None)),
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

        def _legacy_prompt(self, prompt: Any) -> str | list[int]:
            """A completion's prompt as the model reads it: token ids as given, anything else as text."""

            if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                with app.tokenizer_lock:
                    tokenizer = app.tokenizer
                    if not hasattr(type(tokenizer), "__len__"):
                        tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
                    try:
                        vocab = len(tokenizer)
                    except TypeError:
                        vocab = int(tokenizer.vocab_size)
                if any(type(t) is not int or not 0 <= t < vocab for t in prompt):
                    raise RequestError(f"prompt token ids must be integers in the valid range 0 to {vocab - 1}")
                return list(prompt)
            return self._legacy_prompt_to_text(prompt)

        def do_DELETE(self) -> None:
            responses.delete(self, app, responses.route(self._route()))

        def do_POST(self) -> None:
            route = self._route()
            if route.endswith("/decisions"):
                return self._post_decisions(app)
            if responses.route(route) == "":         # a Response: this handler's chat completion, translated
                return responses.post(self, app)

            is_chat_completion = route.endswith("/chat/completions")
            is_text_completion = route.endswith("/completions") and not is_chat_completion
            if not is_chat_completion and not is_text_completion:
                self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 32 * 1024**2:
                    raise RequestError("request body exceeds the 32 MiB limit")
                body = parse_numbers(json.loads(self.rfile.read(length) or b"{}"))
                validate_modalities(body)
                probability_options(body)
                named = reply_model(app, body)          # the id the request asked for, as vLLM names it
                if _REQUEST_LOG and body.get("priority") != "background":   # batch jobs are not client traffic
                    with open(_REQUEST_LOG, "a") as handle:
                        handle.write(json.dumps(redact_images(body)) + "\n")
                raw_kw: dict[str, Any] = {}
                if is_chat_completion:
                    messages = normalize_messages(body.get("messages"), allow_images=getattr(app, "vision", None) is not None)
                    tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
                elif isinstance(body.get("messages"), list) and body["messages"]:
                    messages, tools = normalize_messages(body["messages"], allow_images=getattr(app, "vision", None) is not None), []
                elif getattr(app, "accepts_raw_prompt", False):
                    # a text completion reads its prompt raw, as vLLM and mlx_lm do: no chat template, no think block
                    messages, tools = [], []
                    raw_kw["prompt"] = self._legacy_prompt(body.get("prompt", ""))
                else:
                    messages = [{"role": "user", "content": self._legacy_prompt_to_text(body.get("prompt", ""))}]
                    tools = []
                max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
                temperature = float(body.get("temperature") or 0.0)
                # Preserve raw sampling and scheduling options; an absent temperature differs from temperature zero.
                sampling_fields = {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "priority",
                                                        "draft", "thinking_budget", "ignore_eos", "stop",
                                                        *grammar.FIELDS)
                                   if k in body}
                problem = grammar.refusal(body, app)        # compiled before a stream's headers: a bad grammar is a 400
                if problem:
                    raise RequestError(problem)
                if tools and tool_choice_requires_call(body.get("tool_choice")):
                    sampling_fields["tool_call_required"] = True     # the engine opens the answer with a call
                sampling_fields.update(thinking_fields(body, getattr(app, "effort_levels", frozenset())))
                sampling_kw = ({"sampling": sampling_fields}
                               if getattr(app, "accepts_sampling", False) else {})
                if getattr(app, "accepts_cancellation", False):
                    sampling_kw["cancellation"] = socket_cancellation(self.connection)
                stream = bool(body.get("stream", False))
                tool_policy = ToolCallPolicy(body)
            except RequestError as exc:
                self._send_json({"error": error_body(exc)},
                                status=503 if isinstance(exc, CapacityError) else 400)
                return
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
                    "completion_tokens_details": {"reasoning_tokens": reply.get("reasoning_tokens", 0)},
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
                return tool_policy.finish(reply, tools, parse_tool_calls_from_content)

            def stream_chunk(
                delta: str | dict[str, Any] = "",
                finish_reason: str | None = None,
            ) -> dict[str, Any]:
                if is_text_completion:
                    return {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": named,
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
                    "model": named,
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
                        if error is not None:
                            payload = {"error": {"message": str(error), "type": "server_error"}}
                        else:
                            payload = stream_chunk("", finish_reason or "length")
                            if extras:
                                payload.update(extras)
                        emit(payload)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()

                    def on_delta(delta: str | dict[str, Any]) -> None:
                        # Text completions carry content strings only, excluding reasoning deltas as non-streamed replies do.
                        if is_text_completion and not isinstance(delta, str):
                            return
                        emit(stream_chunk(delta))

                    try:
                        if tools:
                            streamed = [False]

                            def on_prose(delta: str | dict[str, Any]) -> None:
                                delta = tool_policy.delta(delta)
                                if not delta:
                                    return
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
                                    **raw_kw,
                                )
                            )
                            tail = tool_policy.flush()
                            if tail:
                                if not streamed[0]:
                                    streamed[0] = True
                                    emit(stream_chunk({"role": "assistant"}))
                                emit(stream_chunk(tail))
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
                                **raw_kw,
                            )
                    except (BrokenPipeError, ConnectionResetError):
                        raise
                    except RequestCancelled:
                        return
                    except RequestError as exc:
                        emit({"error": error_body(exc)})
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
                            finish_stream(None, error=exc)
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
                        **raw_kw,
                    )
                )
                if is_text_completion:
                    self._send_json(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": named,
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
                        "model": named,
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
            except (BrokenPipeError, ConnectionResetError, RequestCancelled):
                pass
            except RequestError as exc:
                self._send_json({"error": error_body(exc)}, status=400)
            except Exception as exc:  # surface runner errors to the client
                print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    self._send_json({"error": {"message": str(exc)}}, status=500)
                except Exception:
                    pass


        def _post_decisions(self, app: Any) -> None:
            decide = getattr(app, "decisions", None)
            if decide is None:
                self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 32 * 1024**2:
                    raise RequestError("request body exceeds the 32 MiB limit")
                body = parse_numbers(json.loads(self.rfile.read(length) or b"{}"))
                if not isinstance(body, dict):
                    raise RequestError("request body must be an object")
                payload = decide(body)
            except RequestError as exc:
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
                return
            except Exception as exc:  # noqa: BLE001 - a bad body is a client error
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
                return
            self._send_json(payload)

    return Handler
