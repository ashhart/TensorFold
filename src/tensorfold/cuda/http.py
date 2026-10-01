"""The CUDA server's HTTP side: OpenAI routes over ``App`` (tensorfold.cuda.server), streamed or not."""
from __future__ import annotations

import json
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any

from tensorfold.cuda import health
from tensorfold.server import metrics, responses
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server.decisions import DecisionError
from tensorfold.server.errors import CapacityError, RequestError, error_body
from tensorfold.server.http import Server
from tensorfold.server.stacks import Rearming

if TYPE_CHECKING:
    from tensorfold.cuda.server import App


def usage_of(result: dict[str, Any]) -> dict[str, Any]:
    """A reply's usage as the Mac server reports it, the prompt tokens found cached included."""

    return {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": result.get("cached_tokens", 0)},
            "completion_tokens_details": {"reasoning_tokens": result.get("reasoning_tokens", 0)}}


def _error_message(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _log_error(exc: BaseException) -> None:
    print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exception(exc)


POLLED = ("/metrics", "/v1/metrics", "/health", "/v1/health")


def make_handler(app: App):
    class Handler(Rearming):              # USR1's stack dump armed again after each request
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):    # one line a request, as the Mac server prints
            print(f"[tensorfold] {self.address_string()} {fmt % args}", flush=True)

        def log_request(self, code="-", size="-"):
            # a scraper polls these every few seconds and would bury the requests; a failed poll still prints
            if not (self.command == "GET" and self.path.split("?", 1)[0].rstrip("/") in POLLED and code == 200):
                super().log_request(code, size)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if self.close_connection:                # so a pooling client does not reuse the socket
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):          # the client has gone
                self.close_connection = True

        def _discard_body(self) -> None:
            """Read a refused request's body, so it cannot reach the next request on this connection."""

            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = -1
            if 0 <= length <= 32 * 1024**2:
                self.rfile.read(length)
            else:
                self.close_connection = True

        def _stream_error(self, error: dict[str, Any]) -> None:
            """End an open stream with an error event and ``[DONE]``, as the MLX server does."""

            try:
                self.wfile.write(f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

        def do_GET(self):
            route = self.path.split("?", 1)[0].rstrip("/")
            if route in ("/metrics", "/v1/metrics"):
                return metrics.send(self, app)
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "tensorfold"}
                                                            for model_id in app.model_ids]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._json(200, health.of(app).snapshot(app))
            elif responses.route(self.path):
                responses.get(self, app, responses.route(self.path))
            else:
                self._json(404, {"error": "not found"})

        def do_DELETE(self):
            responses.delete(self, app, responses.route(self.path))

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path.endswith("/decisions"):
                return self._post_decisions()
            if responses.route(self.path) == "":         # a Response: this handler's chat completion, translated
                return responses.post(self, app)
            chat = self.path.rstrip("/").endswith("/chat/completions")
            if not chat and not self.path.rstrip("/").endswith("/completions"):
                self._discard_body()
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= 32 * 1024**2:
                    self.close_connection = True             # the unread body must not reach the next request
                    return self._json(400, {"error": {"message": "request body exceeds the 32 MiB limit",
                                                      "type": "invalid_request_error"}})
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": {"message": "the request body is not JSON", "type": "invalid_request_error"}})
            try:
                prepared = app.prepare(body, chat)
            except RequestError as exc:
                return self._json(503 if isinstance(exc, CapacityError) else 400,
                                  {"error": error_body(exc)})
            except Exception as exc:        # any other failure to read the request is refused too, as on MLX
                _log_error(exc)
                return self._json(400, {"error": {"message": _error_message(exc)}})
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            model = app.reply_model(body)
            stream = bool(body.get("stream"))
            kind = "chat.completion.chunk" if chat else "text_completion"
            gone = socket_cancellation(self.connection)          # the Mac server's check: the client has closed
            cancelled = lambda: gone.cancelled                  # noqa: E731

            def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                if chat:
                    return {"id": rid, "object": kind, "created": created, "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                return {"id": rid, "object": kind, "created": created, "model": model,
                        "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def emit(delta: dict[str, Any]) -> bool:
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk(delta))}\n\n".encode())
                        self.wfile.flush()
                        return True
                    except OSError:             # reset, broken pipe, timed out, host unreachable: the client has gone
                        return False

                if chat:
                    emit({"role": "assistant"})
                try:
                    result = app.run(body, chat, emit, prepared=prepared, cancelled=cancelled)
                except RequestCancelled:
                    self.close_connection = True
                    return
                except RequestError as exc:
                    return self._stream_error(error_body(exc))
                except Exception as exc:
                    _log_error(exc)
                    return self._stream_error({"message": _error_message(exc), "type": "server_error"})
                if result["final"]:
                    emit(result["final"])
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        if i < result.get("calls_streamed", 0):      # sent as deltas already
                            continue
                        emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"],
                                                           "arguments": call["function"]["arguments"]}}]})
                end = chunk({}, result["finish"])
                end["tensorfold"] = result["stats"]
                end["usage"] = usage_of(result)          # every stream, as the Mac server's: clients count from it
                try:
                    self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            try:
                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)
            except RequestCancelled:
                self.close_connection = True
                return
            except RequestError as exc:
                return self._json(503 if isinstance(exc, CapacityError) else 400,
                                  {"error": error_body(exc)})
            except Exception as exc:
                _log_error(exc)
                try:
                    self._json(500, {"error": {"message": _error_message(exc)}})
                except OSError:
                    pass
                return
            usage = usage_of(result)
            if chat:
                message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
                if result["reasoning"]:
                    message["reasoning_content"] = result["reasoning"]
                if result["calls"]:
                    message["tool_calls"] = result["calls"]
                payload = {"id": rid, "object": "chat.completion", "created": created, "model": model,
                           "choices": [{"index": 0, "message": message, "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
                if result.get("logprobs") is not None:
                    payload["choices"][0]["logprobs"] = result["logprobs"]
            else:
                payload = {"id": rid, "object": "text_completion", "created": created, "model": model,
                           "choices": [{"index": 0, "text": result["content"], "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
            self._json(200, payload)

        def _post_decisions(self) -> None:
            decide = getattr(app, "decisions", None)
            if decide is None:
                return self._json(404, {"error": {"message": f"unknown path {self.path}",
                                                  "type": "invalid_request_error"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 32 * 1024**2:
                    return self._json(400, {"error": {"message": "request body exceeds the 32 MiB limit",
                                                      "type": "invalid_request_error"}})
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise RequestError("request body must be an object")
            except RequestError as exc:
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:        # noqa: BLE001 - a bad body is a client error
                return self._json(400, {"error": {"message": _error_message(exc), "type": "invalid_request_error"}})
            try:
                payload = decide(body)
            except (RequestError, DecisionError) as exc:
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:        # a scoring failure is the server's, not a bad body
                _log_error(exc)
                try:
                    return self._json(500, {"error": {"message": _error_message(exc)}})
                except OSError:
                    return
            self._json(200, payload)

    return Handler


def serve(app: App, host: str, port: int) -> None:
    """Serve until interrupted (SIGTERM included)."""

    import signal

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    server = Server((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
