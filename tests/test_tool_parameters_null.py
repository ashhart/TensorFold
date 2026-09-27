"""A function tool whose "parameters" is null (a zero-argument tool) must not fail the request."""

import http.client
import json

import pytest

from tensorfold.engine.tool_draft import ToolCallStreamer

from tensorfold.server.tools import parse_tool_calls_from_content
from tests.test_lane_server import make_app
from tests.test_server_openai_compat import post_json, serve_fake

TOOLS = [{"type": "function", "function": {"name": "get_time", "description": "now", "parameters": None}}]


def test_parse_accepts_null_parameters():
    content, calls = parse_tool_calls_from_content("no call here", TOOLS)
    assert content == "no call here" and calls is None


def test_http_chat_with_null_parameters_tool():
    app = make_app()
    server = serve_fake(app)
    try:
        status, body = post_json(server, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
                                  "max_tokens": 4})
        assert status == 200, body
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_http_stream_with_null_parameters_tool():
    app = make_app()
    server = serve_fake(app)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps({"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
                                      "max_tokens": 4, "stream": True}).encode(),
                     headers={"Content-Type": "application/json"})
        text = conn.getresponse().read().decode()
        conn.close()
        assert "tensorfold_error" not in text and '"error"' not in text, text
        assert "data: [DONE]" in text
    finally:
        server.shutdown()
        server.server_close()
        app.close()


@pytest.mark.parametrize("schema", [None, {}, {"properties": None}, {"properties": {"value": True}}])
def test_optional_schemas_match_streamed_and_complete_calls(schema):
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": schema}}]
    text = '<tool_call><function=get_time><parameter=value>123</parameter></function></tool_call>'
    _, calls = parse_tool_calls_from_content(text, tools)
    streamer = ToolCallStreamer(tools)
    deltas = streamer.feed(text)
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == json.loads(calls[0]["function"]["arguments"]) == {"value": "123"}


def test_input_schema_retains_typed_parameters():
    tools = [{"name": "get_time", "input_schema": {"properties": {"value": {"type": "integer"}}}}]
    text = '<tool_call><function=get_time><parameter=value>123</parameter></function></tool_call>'
    _, calls = parse_tool_calls_from_content(text, tools)
    deltas = ToolCallStreamer(tools).feed(text)
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == json.loads(calls[0]["function"]["arguments"]) == {"value": 123}
