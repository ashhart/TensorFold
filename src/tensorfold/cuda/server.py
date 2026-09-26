"""OpenAI-compatible server for TensorFold's CUDA engines (DGX Spark and other NVIDIA GPUs).

One request decodes at a time, as one exact stream: drafted output is byte-identical to serial decoding on the
same engine. The chat template is the model's own ``chat_template.jinja``; Qwen tool calls are parsed from the
reply; with thinking on, text before ``</think>`` streams as ``reasoning_content``.

A family's CUDA engine (``cuda_engine`` in its package) provides:

    eos                                              # token ids that end a reply
    generate(prompt, max_tokens, sampling, on_tokens) -> dict
                                                     # decode; on_tokens(new_ids) returns True to stop early
    follow()                                         # two GPUs: rank 1 mirrors every request rank 0 serves

``tensorfold serve MODEL`` builds the engine and serves it here (see ``tensorfold.cli``).
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

_THINK_END = "</think>"
_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)


# -- text helpers (same rules as the Mac lane server) ---------------------------------------

def _partial_tag(text: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class StreamDecoder:
    """The text of a growing token list, decoding only a short window each time tokens arrive.

    Decoding the whole reply every round grows with its length. Each step decodes the tokens from
    ``prefix`` on twice, with and without the newest ones, and appends the difference, the way vLLM
    detokenizes: the shared window keeps a decoder's leading-space and byte-level rules the same on
    both sides. A trailing partial multi-byte character waits for the next tokens.
    """

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok, self.skip = tok, frozenset(skip)
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]


def split_thinking(text: str, *, finished: bool) -> tuple[str, str]:
    end = text.find(_THINK_END)
    if end < 0:
        return text[: len(text) - (0 if finished else _partial_tag(text, _THINK_END))], ""
    return text[:end], text[end + len(_THINK_END):].lstrip("\n")


def hide_tool_calls(text: str, *, finished: bool) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            out.append(tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out)
        pos = end + len(_CALL_CLOSE)


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def parse_tool_calls(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`` or JSON bodies."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        residue.append(text[cursor:match.start()])
        cursor = match.end()
        block = match.group(1).strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                name = m.group(1).strip()
                args = {p.group(1).strip(): p.group(2) for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2))}
        if not name or str(name).lower() not in known:
            residue.append(match.group(0))
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


# -- chat template -------------------------------------------------------------------------

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

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None) -> str:
        for m in messages:                       # tool_call arguments arrive as JSON strings
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)


# -- HTTP ------------------------------------------------------------------------------------

class App:
    """One engine behind the OpenAI routes. ``sampling``: temperature, top_k and top_p for requests that do not
    set them (the model's generation config and the CLI's flags); ``max_tokens``: the reply length likewise."""

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096):
        from tokenizers import Tokenizer

        self.engine = engine
        self.served = served
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.lock = threading.Lock()

    def check(self, body: dict[str, Any]) -> str | None:
        """Why the request cannot run, or None. Checked before a stream's headers are sent."""

        import inspect

        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        return None

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy decoding."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        temp = float(body["temperature"] if body.get("temperature") is not None else self.sampling["temperature"])
        if temp <= 0:
            return None
        seed = body.get("seed")
        top_k = body["top_k"] if body.get("top_k") is not None else self.sampling["top_k"]
        top_p = body["top_p"] if body.get("top_p") is not None else self.sampling["top_p"]
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        tools = body.get("tools") or []
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
        if chat:
            text = self.template.render(body["messages"], tools=tools, enable_thinking=thinking, extra=kwargs)
        else:
            text = body["prompt"]
        prompt = self.tok.encode(text, add_special_tokens=False).ids
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens)
        sampling = self.sampling_for(body, prompt)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False}
        stream = StreamDecoder(self.tok, tuple(self.engine.eos))

        def visible(finished: bool) -> tuple[str, str]:
            raw = stream.final() if finished else stream.text
            if chat and thinking:
                reasoning, answer = split_thinking(raw, finished=finished)
            else:
                reasoning, answer = "", raw
            if tools:
                answer = hide_tool_calls(answer, finished=finished)
            return reasoning, answer

        def on_tokens(new: list[int]) -> bool:
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
            return stopped["client"]

        draft = body.get("draft", True) is not False
        with self.lock:
            stats = self.engine.generate(prompt, max_tokens, sampling, on_tokens,
                                         **({} if draft else {"draft": False}))
        reasoning, answer = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final["reasoning_content"] = reasoning[sent["reasoning"]:]
        raw_answer = split_thinking(self.tok.decode([t for t in out if t not in self.engine.eos],
                                                    skip_special_tokens=False), finished=True)[1] \
            if chat and thinking else self.tok.decode([t for t in out if t not in self.engine.eos],
                                                      skip_special_tokens=False)
        content, calls = parse_tool_calls(raw_answer, tools) if tools else (answer, None)
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        finish = "tool_calls" if calls else ("stop" if out and out[-1] in self.engine.eos else "length")
        return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "stats": stats}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
            except json.JSONDecodeError:
                return self._json(400, {"error": {"message": "the request body is not JSON", "type": "invalid_request_error"}})
            problem = app.check(body)
            if problem:
                return self._json(400, {"error": {"message": problem, "type": "invalid_request_error"}})
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            stream = bool(body.get("stream"))
            kind = "chat.completion.chunk" if chat else "text_completion"

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
                    except (BrokenPipeError, ConnectionResetError):
                        return False

                if chat:
                    emit({"role": "assistant"})
                result = app.run(body, chat, emit)
                if result["final"]:
                    emit(result["final"])
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"],
                                                           "arguments": call["function"]["arguments"]}}]})
                end = chunk({}, result["finish"])
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
            result = app.run(body, chat, lambda delta: True)
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
    from http.server import ThreadingHTTPServer

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
