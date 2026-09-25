import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from typing import Any

from tensorfold.server.http import make_handler


class FakeTokenizer:
    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class FakeApp:
    served_name = "fake-model"
    model_ids = ["fake-model"]
    max_batch_size = 1
    exact_mode = {"mode": "target-verified"}

    def __init__(
        self,
        *,
        fail_stream: bool = False,
        content: str = "Hello",
    ) -> None:
        self.fail_stream = fail_stream
        self.content = content
        self.tokenizer = FakeTokenizer()
        self.tokenizer_lock = threading.Lock()
        self.messages: list[dict[str, Any]] | None = None
        self.tools: list[dict[str, Any]] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        on_delta: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.messages = messages
        self.tools = tools
        if on_delta is not None:
            if self.fail_stream:
                raise RuntimeError("boom")
            on_delta("Hel")
            on_delta("lo")
        return {
            "content": self.content,
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "cached_tokens": 0,
            "completion_tokens": 2,
            "runtime": {"tokens_per_second": 42.0},
        }


def serve_fake(app: FakeApp):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def post_json(server: ThreadingHTTPServer, path: str, payload: dict[str, Any]):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    return response.status, body


def test_legacy_completions_accepts_openai_completions_prompt() -> None:
    app = FakeApp()
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/completions",
            {"model": "fake-model", "prompt": "Hi", "max_tokens": 8},
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = json.loads(body)
    assert status == 200
    assert app.messages == [{"role": "user", "content": "Hi"}]
    assert payload["object"] == "text_completion"
    assert payload["id"].startswith("cmpl-")
    assert payload["choices"][0]["text"] == "Hello"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_legacy_completions_stream_finishes_with_finish_reason() -> None:
    app = FakeApp()
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/completions",
            {"model": "fake-model", "prompt": "Hi", "stream": True, "max_tokens": 8},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert '"object": "text_completion"' in body
    assert '"text": "Hel"' in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_chat_completions_stream_starts_with_assistant_role() -> None:
    app = FakeApp()
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "max_tokens": 8,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert '"delta": {"role": "assistant"}' in body
    assert body.index('"delta": {"role": "assistant"}') < body.index(
        '"delta": {"content": "Hel"}'
    )
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body


def test_stream_error_after_headers_still_sends_finish_reason() -> None:
    app = FakeApp(fail_stream=True)
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert '"object": "chat.completion.chunk"' in body
    assert '"finish_reason": "stop"' in body
    assert '"tensorfold_error"' in body
    assert "data: [DONE]" in body


def test_chat_completions_parses_tool_call_response() -> None:
    tool_call_text = '<tool_call>{"name":"lookup","arguments":{"query":"speed"}}</tool_call>'
    app = FakeApp(content=tool_call_text)
    server = serve_fake(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Search"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = json.loads(body)
    message = payload["choices"][0]["message"]
    call = message["tool_calls"][0]
    assert status == 200
    assert app.tools == tools
    assert message["content"] is None
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"query": "speed"}


def test_chat_completions_parses_bare_json_tool_call_response() -> None:
    app = FakeApp(content='{"tool":"lookup","query":"ping"}')
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Search"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = json.loads(body)
    message = payload["choices"][0]["message"]
    call = message["tool_calls"][0]
    assert status == 200
    assert message["content"] is None
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"query": "ping"}


def test_chat_completions_streams_tool_call_deltas() -> None:
    tool_call_text = '<tool_call>{"name":"lookup","arguments":{"query":"speed"}}</tool_call>'
    app = FakeApp(content=tool_call_text)
    server = serve_fake(app)
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Search"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "stream": True,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert '"tool_calls"' in body
    assert '"name": "lookup"' in body
    assert '"arguments": "{\\"query\\":\\"speed\\"}"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert "<tool_call>" not in body
    assert "data: [DONE]" in body
