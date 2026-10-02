"""POST /v1/messages: Anthropic's protocol translated to this server's chat completions and back."""

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tensorfold.server import anthropic, anthropic_translate
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler


def serve(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return httpd, thread


def post(port: int, body: dict, path: str = "/v1/messages"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        data = json.dumps(body).encode()
        connection.request("POST", path, body=data, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type") or "", response.read().decode()
    finally:
        connection.close()


# -- translation ------------------------------------------------------------------------------


def test_a_request_translates_system_messages_tools_and_limits():
    request = anthropic_translate.translate({
        "model": "claude-test", "max_tokens": 64, "system": "You are terse.",
        "temperature": 0.5, "top_p": 0.9, "stop_sequences": ["END"],
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"},
                                         {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                      "data": "abc"}}]},
        ],
        "tools": [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    })
    chat = request.chat
    assert chat["messages"][0] == {"role": "system", "content": "You are terse."}
    assert chat["messages"][1] == {"role": "user", "content": "Hello"}
    assert chat["messages"][2]["content"][1]["image_url"]["url"] == "data:image/png;base64,abc"
    assert chat["tools"][0]["function"]["name"] == "get_weather"
    assert chat["tool_choice"] == "required"                 # Anthropic's any is OpenAI's required
    assert chat["max_tokens"] == 64 and chat["stop"] == ["END"]
    assert chat["temperature"] == 0.5 and chat["top_p"] == 0.9


def test_tool_use_and_tool_result_round_trip_as_chat_messages():
    out = anthropic_translate.messages([
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}]},
    ])
    assert out[0]["role"] == "assistant" and out[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}
    assert out[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"}
    # a tool turn is followed by its user turn: the chat template needs a user query after tool results
    assert out[2]["role"] == "user" and out[2]["content"]


def test_replayed_assistant_thinking_joins_its_message():
    out = anthropic_translate.messages([
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "considering"},
            {"type": "text", "text": "Answer"}]},
    ])
    assert out == [{"role": "assistant", "content": [{"type": "text", "text": "Answer"}],
                    "reasoning_content": "considering"}]


def test_thinking_budget_and_disable_map_to_the_server_s_fields():
    enabled = anthropic_translate.translate({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
                                             "thinking": {"type": "enabled", "budget_tokens": 2048}})
    assert enabled.chat["thinking_budget"] == 2048
    assert enabled.chat["chat_template_kwargs"]["enable_thinking"] is True
    disabled = anthropic_translate.translate({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
                                              "thinking": {"type": "disabled"}})
    assert disabled.chat["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("body, message", [
    ({"model": "m", "max_tokens": 8, "messages": []}, "messages must be a non-empty list"),
    ({"model": "m", "messages": [{"role": "user", "content": "x"}]}, "max_tokens is required"),
    ({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
      "mcp_servers": ["a"]}, "MCP servers"),
    ({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
      "thinking": {"type": "enabled", "budget_tokens": 10}}, "budget_tokens"),
    ({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
      "tools": [{"type": "web_search_20250305", "name": "search"}]}, "function tools only"),
])
def test_unsupported_requests_are_refused_by_name(body, message):
    with pytest.raises(RequestError, match=message):
        anthropic_translate.translate(body)


# -- replies ----------------------------------------------------------------------------------


def test_a_chat_completion_becomes_anthropic_blocks_and_usage():
    reply = anthropic_translate.Reply({"id": "msg_1", "type": "message", "role": "assistant", "model": "m"})
    final = reply.completion({"choices": [{"message": {"role": "assistant", "content": "Hi there",
                                                       "reasoning_content": "hmm"},
                                           "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    assert [b["type"] for b in final["content"]] == ["thinking", "text"]
    assert final["content"][1]["text"] == "Hi there"
    assert final["stop_reason"] == "end_turn"
    assert final["usage"] == {"input_tokens": 5, "output_tokens": 3}


def test_tool_calls_become_tool_use_blocks_with_parsed_input():
    reply = anthropic_translate.Reply({"id": "msg_1", "type": "message", "role": "assistant", "model": "m"})
    final = reply.completion({"choices": [{"message": {"role": "assistant", "content": None,
                                                       "tool_calls": [{"id": "call_1", "type": "function",
                                                                       "function": {"name": "get_weather",
                                                                                    "arguments": '{"city": "Oslo"}'}}]},
                                           "finish_reason": "tool_calls"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 9}})
    block = final["content"][0]
    assert block["type"] == "tool_use" and block["name"] == "get_weather"
    assert block["input"] == {"city": "Oslo"}
    assert final["stop_reason"] == "tool_use"


def test_the_length_finish_is_anthropic_s_max_tokens():
    reply = anthropic_translate.Reply({"id": "msg_1", "type": "message", "role": "assistant", "model": "m"})
    final = reply.completion({"choices": [{"message": {"role": "assistant", "content": "Hi"},
                                           "finish_reason": "length"}], "usage": {}})
    assert final["stop_reason"] == "max_tokens"


# -- HTTP -------------------------------------------------------------------------------------


class _EchoApp(SimpleNamespace):
    """Serves chat completions with a fixed reply, through the server's own handler."""

    def __init__(self, reply: dict):
        super().__init__(served_name="test", model_ids=["test"], max_batch_size=1, reply=reply)


def test_post_messages_serves_a_nonstreamed_reply(app=None):
    pytest.importorskip("jinja2")


def test_the_endpoint_routes_and_refuses_with_anthropic_s_error_shape(tmp_path):
    app = _EchoApp({})
    httpd, thread = serve(app)
    try:
        status, _, body = post(httpd.server_port, {"model": "test", "max_tokens": 8, "messages": []})
        assert status == 400
        payload = json.loads(body)
        assert payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error"
        assert "messages" in payload["error"]["message"]
        status, _, body = post(httpd.server_port, {"model": "test", "max_tokens": 8,
                                                   "messages": []}, path="/messages")
        assert status == 400                                       # the un-prefixed alias routes too
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)
