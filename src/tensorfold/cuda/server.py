"""OpenAI server for the CUDA engines: a family's ``cuda_engine`` gives ``eos``, ``generate`` and ``follow``."""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
import traceback
import uuid
from datetime import datetime
from contextlib import nullcontext
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server.errors import RequestError
from tensorfold.server.http import Server
from tensorfold.server.messages import (_normalize_tool_call_arguments, late_system_role, normalize_messages,
                                        validate_modalities)
from tensorfold.server.request_options import parse_numbers
from tensorfold.server.stopping import stop_options
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.engine.call_gate import CallGate, call_format, generate_gated
from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

from tensorfold.cuda.reply_text import StopStrings, StreamDecoder, hide_tool_calls, parse_tool_calls
from tensorfold.server.text import split_thinking


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg["chat_template"]

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}
        self.late_system = late_system_role(
            lambda messages: self.template.render(**self.specials, messages=messages, add_generation_prompt=False))

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None) -> str:
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=self.late_system))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)


# -- HTTP ------------------------------------------------------------------------------------

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "seed")


@dataclass(slots=True)
class PreparedRequest:
    prompt: list[int]
    max_tokens: int
    tools: list[dict[str, Any]]
    thinking: bool
    sampling: Any          # the engine's ``Sampling``, or None for greedy decoding
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()


def _native_context(model_dir: Path) -> int:
    path = model_dir / "config.json"
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


class App:
    """Serve one engine with sampling and reply-length defaults for requests that omit them."""

    reads_ignore_eos = False            # True where the engine reads ``ignore_eos`` itself; a ``stop_eos`` engine is given it

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096,
                 context_window: int | None = None):
        from tokenizers import Tokenizer

        self.engine = engine
        self.served = served
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.native_context_window = _native_context(model_dir)
        self.context_window = self.native_context_window if context_window is None else int(context_window)
        if self.context_window < 0:
            raise ValueError("context_window must be 0 or a positive token count")
        self.lock = threading.Lock()

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        import inspect

        if not isinstance(body, dict):
            return "the request body must be a JSON object"
        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        return None

    def _engine_capacity(self) -> int | None:
        capacities = []
        for name in ("context_window", "limit"):
            limit = getattr(self.engine, name, None)
            if isinstance(limit, int):
                capacities.append(max(0, limit))
        return min(capacities) if capacities else None

    def _restart(self, need: int, ranks: str = "") -> str:
        """A larger ``--context`` to restart with, only where the startup admission would accept it."""

        largest = (getattr(self.engine, "capacity_plan", None) or {}).get("largest_window")
        if largest is None or need > largest:
            return ""
        return f", or restart{ranks} with --context {need} or more (this memory admits up to {largest})"

    def _context_limit(self) -> int | None:
        limits = [self.context_window] if self.context_window > 0 else []
        capacity = self._engine_capacity()
        if capacity is not None:
            limits.append(capacity)
        return min(limits) if limits else None

    @property
    def effective_context_window(self) -> int | None:
        """Safe prompt-plus-reply capacity; None is unlimited, while zero refuses every prompt."""

        return self._context_limit()

    def _requested_tokens(self, body: dict[str, Any]) -> int:
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is not None:
                try:
                    int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RequestError(f"{name} must be an integer token count") from exc
        return max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens))

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        from jinja2.exceptions import TemplateError

        validate_modalities(body)
        ToolCallPolicy(body)
        ignore_eos, stop = stop_options(body)
        max_tokens = self._requested_tokens(body)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:                   # absent or null: the server's defaults
            kwargs = {}
        elif not isinstance(kwargs, dict):   # [], "", false and 0 included
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = dict(kwargs)
        thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
        if chat:
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            try:
                text = self.template.render(body["messages"], tools=tools, enable_thinking=thinking, extra=kwargs)
            except TemplateError as exc:     # the checkpoint's template refuses the request (``raise_exception``)
                raise RequestError(f"the chat template rejected the request: {exc}") from exc
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string")
        prompt = self.tok.encode(text, add_special_tokens=False).ids
        if not prompt:
            raise RequestError("rendered prompt is empty")
        # sampling is resolved here, so a malformed control is refused before a stream opens
        return PreparedRequest(prompt, max_tokens, tools, thinking, self.sampling_for(body, prompt), ignore_eos, stop)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Why the request cannot run, or None; rendered before a stream's headers are sent."""

        problem = self._check_fields(body)
        if problem:
            return problem
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        limit = self._context_limit()
        if limit is not None and len(prepared.prompt) >= limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            native = f" (model window: {self.native_context_window} tokens)" if self.native_context_window else ""
            return (f"the rendered prompt has {len(prepared.prompt)} tokens and leaves no room for a reply in "
                    f"the server's {limit}-token {kind}{native}; shorten the prompt"
                    f"{self._restart(len(prepared.prompt) + 1)}")
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"the rendered prompt has {len(prepared.prompt)} tokens and requests {prepared.max_tokens} "
                    f"reply tokens, exceeding the server's {limit}-token {kind}; reduce the prompt or reply "
                    f"length{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
        return None

    def prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        problem = self._check_fields(body)
        if problem:
            raise RequestError(problem)
        prepared = self._prepare(body, chat)
        problem = self.check(body, prepared=prepared)
        if problem:
            raise RequestError(problem)
        limit = self._context_limit()
        if limit is not None:
            prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))
        return prepared

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        temp = float(fields.get("temperature", self.sampling["temperature"]))
        if temp <= 0:
            return None
        seed = fields.get("seed")
        top_k = fields.get("top_k", self.sampling["top_k"])
        top_p = fields.get("top_p", self.sampling["top_p"])
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``."""

        prepared = prepared if prepared is not None else self.prepare(body, chat)
        prompt, max_tokens = prepared.prompt, prepared.max_tokens
        tools, thinking = prepared.tools, prepared.thinking
        policy = ToolCallPolicy(body)
        sampling = prepared.sampling
        # end tokens end the reply and stay out of its text, unless it asks ignore_eos of an engine that reads it
        takes_stop_eos = "stop_eos" in inspect.signature(self.engine.generate).parameters
        ends = () if prepared.ignore_eos and (self.reads_ignore_eos or takes_stop_eos) else tuple(self.engine.eos)
        stops = StopStrings(prepared.stop, self.tok, ends)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False, "stop": False}
        failed: list[Exception] = []
        stream = StreamDecoder(self.tok, ends)

        def visible(finished: bool) -> tuple[str, str]:
            raw = stream.final() if finished else stream.text
            # stop strings match the generated text, reasoning included, before it is split (as on the Mac)
            raw = stops.visible(raw, partial=not finished) if stops.strings else raw
            if chat and thinking:
                reasoning, answer = split_thinking(raw, finished=finished)
            else:
                reasoning, answer = "", raw
            if tools:
                answer = (policy.content(answer, finished=finished) if policy.single
                          else hide_tool_calls(answer, finished=finished))
            return reasoning, answer

        def on_tokens(new: list[int]) -> bool:
            # True stops the engine after this round; engines that finish on both ranks keep calling and get True
            if stopped["client"] or stopped["stop"] or failed:
                return True
            try:
                if stops.strings:
                    kept = []
                    for token in new:             # token by token: the round's width cannot move the cut
                        kept.append(token)
                        out.append(token)
                        if stops.hit(out):
                            stopped["stop"] = True
                            break
                    new = kept
                else:
                    out.extend(new)
                stream.add(new)
                reasoning, answer = visible(False)
                delta: dict[str, Any] = {}
                if len(reasoning) > sent["reasoning"]:
                    delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                    sent["reasoning"] = len(reasoning)
                if len(answer) > sent["content"]:
                    delta["content"] = answer[sent["content"]:]
                    sent["content"] = len(answer)
                if delta and not emit(delta):
                    stopped["client"] = True
                elif cancelled is not None and cancelled():     # every round, with or without new text
                    stopped["client"] = True
            except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
                failed.append(exc)
                return True
            return stopped["client"] or stopped["stop"]

        draft = body.get("draft", True) is not False
        gate = self._call_gate(prompt, tools) if tools and tool_choice_requires_call(body.get("tool_choice")) else None

        options: dict[str, Any] = {} if draft else {"draft": False}
        if takes_stop_eos:
            options["stop_eos"] = not prepared.ignore_eos

        def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
            return self.engine.generate(ids, count, sampling, feed, **options)

        # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
        with (nullcontext() if getattr(self.engine, "concurrent", False) else self.lock):
            if cancelled is not None and cancelled():                # the client left while this request waited
                raise RequestCancelled("the client left before the request started")
            stats = generate_gated(generate, prompt, max_tokens, gate, on_tokens)
        if failed:
            raise failed[0]
        if stopped["client"]:                                        # as the Mac server: nothing more is written
            raise RequestCancelled("the client left during the reply")
        stats = {**(stats or {}), "token_sha": token_sha(out)}
        reasoning, answer = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final["reasoning_content"] = reasoning[sent["reasoning"]:]
        text = stops.visible(self.tok.decode([t for t in out if t not in ends], skip_special_tokens=False))
        raw_answer = split_thinking(text, finished=True)[1] if chat and thinking else text
        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
        content = policy.content(content) if tools else content
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        finish = "tool_calls" if calls else ("stop" if stopped["stop"] or (out and out[-1] in ends) else "length")
        if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
            stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
        return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "stats": stats}

    def _call_gate(self, prompt: list[int], tools: list[dict[str, Any]]) -> CallGate:
        """The gate a required tool call needs, from this template's call markup and the rendered prompt."""

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            try:
                text = self.template.render(probe, tools=None, enable_thinking=False)
            except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                text = ""
            openers = [o for o in ("<tool_call>", "<|tool_call>") if self.tok.token_to_id(o) is not None]
            self._form = call_format(text, "tfprobe_fn", openers) or ((openers[0], None, None) if openers else None)
        if self._form is None:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opener, lead, tail = self._form
        eos = set(self.engine.eos)

        def text(token: int) -> str:
            return self.tok.decode([token], skip_special_tokens=False)

        def blank(token: int) -> bool:
            return token not in eos and not text(token).strip()

        think = [-1 if self.tok.token_to_id(t) is None else self.tok.token_to_id(t) for t in ("<think>", "</think>")]
        names = [str((t.get("function") or t).get("name") or "") for t in tools] if lead is not None else []
        return CallGate.after_prompt(prompt, self.tok.token_to_id(opener), blank, think_open=think[0],
                                     think_end=think[1], text=text, lead=lead or "", names=names, tail=tail or "",
                                     encode=lambda t: list(self.tok.encode(t, add_special_tokens=False).ids))


def token_sha(tokens: list[int]) -> str:
    """A reply's token ids, hashed as the Mac server does: drafted and ``"draft": false`` replies must match."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


def _error_message(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _log_error(exc: BaseException) -> None:
    print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exception(exc)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):          # the client has gone
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
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": app.served, "object": "model", "owned_by": "tensorfold"}]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            chat = self.path.rstrip("/").endswith("/chat/completions")
            if not chat and not self.path.rstrip("/").endswith("/completions"):
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": {"message": "the request body is not JSON", "type": "invalid_request_error"}})
            try:
                prepared = app.prepare(body, chat)
            except RequestError as exc:
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:        # any other failure to read the request is refused too, as on MLX
                _log_error(exc)
                return self._json(400, {"error": {"message": _error_message(exc)}})
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            stream = bool(body.get("stream"))
            kind = "chat.completion.chunk" if chat else "text_completion"
            gone = socket_cancellation(self.connection)          # the Mac server's check: the client has closed
            cancelled = lambda: gone.cancelled                  # noqa: E731

            def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                if chat:
                    return {"id": rid, "object": kind, "created": created, "model": app.served,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                return {"id": rid, "object": kind, "created": created, "model": app.served,
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
                    return self._stream_error({"message": str(exc), "type": "invalid_request_error"})
                except Exception as exc:
                    _log_error(exc)
                    return self._stream_error({"message": _error_message(exc), "type": "server_error"})
                if result["final"]:
                    emit(result["final"])
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"],
                                                           "arguments": call["function"]["arguments"]}}]})
                end = chunk({}, result["finish"])
                end["tensorfold"] = result["stats"]
                usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                         "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}
                if (body.get("stream_options") or {}).get("include_usage"):
                    end["usage"] = usage
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
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:
                _log_error(exc)
                try:
                    self._json(500, {"error": {"message": _error_message(exc)}})
                except OSError:
                    pass
                return
            usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                     "total_tokens": result["prompt_tokens"] + result["completion_tokens"]}
            if chat:
                message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
                if result["reasoning"]:
                    message["reasoning_content"] = result["reasoning"]
                if result["calls"]:
                    message["tool_calls"] = result["calls"]
                payload = {"id": rid, "object": "chat.completion", "created": created, "model": app.served,
                           "choices": [{"index": 0, "message": message, "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
            else:
                payload = {"id": rid, "object": "text_completion", "created": created, "model": app.served,
                           "choices": [{"index": 0, "text": result["content"], "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
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
