import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from tensorfold.server.http import make_handler


class FakeTokenizer:
    vocab_size = 256

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
        reasoning: str = "",
        cached: int = 0,
    ) -> None:
        self.fail_stream = fail_stream
        self.content = content
        self.reasoning = reasoning
        self.cached = cached
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
            if self.reasoning:
                on_delta({"reasoning_content": self.reasoning})
            on_delta("Hel")
            on_delta("lo")
        return {
            "content": self.content,
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "cached_tokens": self.cached,
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


def test_legacy_completions_stream_sends_text_as_strings_without_reasoning() -> None:
    app = FakeApp(reasoning="thinking it over")
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
    chunks = [json.loads(line[5:]) for line in body.splitlines()
              if line.startswith("data:") and line != "data: [DONE]"]
    texts = [choice["text"] for chunk in chunks for choice in chunk.get("choices", [])]
    assert texts and all(isinstance(text, str) for text in texts)
    assert "".join(texts) == "Hello" and "thinking it over" not in body


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


def test_stream_error_after_headers_sends_an_error_event() -> None:
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
    assert '"finish_reason": "stop"' not in body
    assert '"error": {"message": "boom", "type": "server_error"}' in body
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


@pytest.mark.parametrize("chat", [False, True])
def test_include_usage_moves_usage_into_its_own_chunk_before_done(chat: bool) -> None:
    """litellm reads usage only from the spec's chunk: no choices, the reply's id and model, after the finish chunk."""

    server = serve_fake(FakeApp(cached=7))
    prompt = {"messages": [{"role": "user", "content": "Hi"}]} if chat else {"prompt": "Hi"}
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions" if chat else "/v1/completions",
            {**prompt, "model": "fake-model", "stream": True, "max_tokens": 8,
             "stream_options": {"include_usage": True}},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    lines = [line for line in body.splitlines() if line.startswith("data:")]
    chunks = [json.loads(line[5:]) for line in lines if line != "data: [DONE]"]
    assert lines[-1] == "data: [DONE]" and lines.count("data: [DONE]") == 1
    usage_chunk, end = chunks[-1], [c for c in chunks if c["choices"]][-1]
    assert chunks.index(end) == len(chunks) - 2 and "usage" not in end
    assert end["choices"][0]["finish_reason"] == "stop" and "exact_mode" in end
    assert usage_chunk["choices"] == [] and usage_chunk["object"] == end["object"]
    assert all(usage_chunk[key] == end[key] for key in ("id", "created", "model"))
    assert usage_chunk["usage"] == {
        "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 7},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


@pytest.mark.parametrize("chat", [False, True])
def test_stream_without_include_usage_keeps_usage_on_the_finish_chunk(chat: bool) -> None:
    """The clients that never send stream_options keep counting tokens from the finish chunk, as they always did."""

    server = serve_fake(FakeApp(cached=7))
    prompt = {"messages": [{"role": "user", "content": "Hi"}]} if chat else {"prompt": "Hi"}
    try:
        status, body = post_json(
            server,
            "/v1/chat/completions" if chat else "/v1/completions",
            {**prompt, "model": "fake-model", "stream": True, "max_tokens": 8},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    chunks = [json.loads(line[5:]) for line in body.splitlines()
              if line.startswith("data:") and line != "data: [DONE]"]
    assert all(c["choices"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 7
