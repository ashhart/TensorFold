import copy
import json

import pytest

from tensorfold.engine.tool_draft import ToolCallStreamer
from tests.test_server_openai_compat import FakeApp, post_json, serve_fake


TOOLS = [{"type": "function", "function": {"name": "measure", "parameters": {
    "type": "object", "properties": {"value": {"type": "integer"}}}}}]
CALL = '<tool_call><function=measure><parameter=value>{}</parameter></function></tool_call>'


class PolicyApp(FakeApp):
    streams_prose_with_tools = True

    def __init__(self, *, content="Hello", incremental=False):
        super().__init__(content=content)
        self.incremental = incremental
        self.chats = 0

    def chat(self, messages, *, on_delta=None, tools=None, **kwargs):
        self.chats += 1
        self.messages = messages
        reply = super().chat(messages, tools=tools, **kwargs)
        if on_delta is not None and self.incremental:
            streamer = ToolCallStreamer(tools)
            for stop in range(1, len(self.content) + 1):
                for delta in streamer.feed(self.content[:stop]):
                    on_delta(delta)
            reply["tool_calls_streamed"] = streamer.streamed
        return reply


def request(app, body):
    server = serve_fake(app)
    try:
        return post_json(server, "/v1/chat/completions", body)
    finally:
        server.shutdown()
        server.server_close()


def stream_calls(body):
    calls = {}
    for line in body.splitlines():
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        assert "tensorfold_error" not in chunk and "error" not in chunk
        for choice in chunk.get("choices", []):
            for call in choice["delta"].get("tool_calls", []):
                target = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                target["name"] += call.get("function", {}).get("name", "")
                target["arguments"] += call.get("function", {}).get("arguments", "")
    return calls


@pytest.mark.parametrize("stream", [False, True])
def test_developer_and_late_system_messages_are_preserved(stream):
    messages = [{"role": "developer", "content": "First."},
                {"role": "system", "content": "Second."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "developer", "content": "Third."},
                {"role": "user", "content": "Continue"}]
    original = copy.deepcopy(messages)
    app = PolicyApp()
    status, body = request(app, {"messages": messages, "stream": stream})
    assert status == 200
    if stream:
        stream_calls(body)
    # a later instruction keeps its place; the app's template decides its role (tests/test_message_roles.py)
    assert app.messages == [{"role": "system", "content": "First.\n\nSecond."},
                            messages[2], messages[3], {"role": "system", "content": "Third."}, messages[5]]
    assert messages == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "https://example.com/picture.png"}},
    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/movie.mp4"}},
])
def test_unsupported_modalities_refuse_before_chat_or_stream(part, stream):
    app = PolicyApp()
    status, body = request(app, {"stream": stream, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe it."}, part]}]})
    assert status == 400
    assert app.chats == 0
    assert "text" in json.loads(body)["error"]["message"].lower()
    assert not body.startswith("data:")


def test_text_parts_join_without_losing_text_or_mutating_input():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "first "}, {"type": "text", "text": "second"}]}]
    original = copy.deepcopy(messages)
    app = PolicyApp()
    assert request(app, {"messages": messages})[0] == 200
    assert app.messages == [{"role": "user", "content": "first second"}]
    assert messages == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_parallel_false_returns_only_first_complete_call(stream, incremental):
    app = PolicyApp(content=CALL.format(1) + CALL.format(2), incremental=incremental)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure twice"}],
                               "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    if stream:
        calls = stream_calls(body)
        assert set(calls) == {0}
        assert calls[0]["name"] == "measure"
        assert json.loads(calls[0]["arguments"]) == {"value": 1}
        assert '"finish_reason": "tool_calls"' in body and "data: [DONE]" in body
    else:
        calls = json.loads(body)["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"value": 1}


def test_parallel_false_does_not_emit_an_incomplete_call():
    app = PolicyApp(content='<tool_call><function=measure><parameter=value>1', incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
                               "tools": TOOLS, "stream": True, "parallel_tool_calls": False})
    assert status == 200
    assert stream_calls(body) == {}


@pytest.mark.parametrize("parallel", [None, True])
def test_default_parallel_calls_remain_available(parallel):
    body = {"messages": [{"role": "user", "content": "Measure"}], "tools": TOOLS, "stream": True}
    if parallel is not None:
        body["parallel_tool_calls"] = parallel
    app = PolicyApp(content=CALL.format(1) + CALL.format(2), incremental=True)
    status, raw = request(app, body)
    assert status == 200 and set(stream_calls(raw)) == {0, 1}


def test_parallel_flag_requires_a_boolean_before_streaming():
    app = PolicyApp()
    status, body = request(app, {"messages": [{"role": "user", "content": "Hi"}],
                               "parallel_tool_calls": "false", "stream": True})
    assert status == 400 and app.chats == 0
    assert "parallel_tool_calls" in json.loads(body)["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_single_call_skips_malformed_parameters_instead_of_inventing_empty_args(stream):
    malformed = '<tool_call><function=measure><parameter=value>broken</function></tool_call>'
    app = PolicyApp(content=malformed + CALL.format(2), incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    arguments = calls[0]["arguments"] if stream else calls[0]["function"]["arguments"]
    assert json.loads(arguments) == {"value": 2}


@pytest.mark.parametrize("stream", [False, True])
def test_single_call_ignores_a_malformed_extra_call(stream):
    app = PolicyApp(content=CALL.format(1) + '<tool_call>broken</tool_call>', incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1


@pytest.mark.parametrize("options", [
    {"images": ["https://example.com/picture.png"]}, {"modalities": ["audio"]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi",
        "image_url": {"url": "https://example.com/picture.png"}}]}]},
])
def test_hidden_or_requested_media_is_not_silently_ignored(options):
    app = PolicyApp()
    status, body = request(app, {"messages": [{"role": "user", "content": "Hi"}],
                               "stream": True, **options})
    assert status == 400 and app.chats == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":NaN}', '"{\\"value\\":1e999}"'])
def test_single_call_skips_nonfinite_json_arguments(stream, arguments):
    bad = '<tool_call>{"name":"measure","arguments":' + arguments + '}</tool_call>'
    app = PolicyApp(content=bad + CALL.format(2), incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    arguments = calls[0]["arguments"] if stream else calls[0]["function"]["arguments"]
    assert len(calls) == 1 and json.loads(arguments) == {"value": 2}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content, visible", [
    ('Before<m:tool_call><function=measure><parameter=value>1', 'Before'),
    ('Before<m', 'Before<m'), ('Before<math>x</math>', 'Before<math>x</math>'),
])
def test_single_call_hides_incomplete_namespaced_envelopes(stream, content, visible):
    class ProseApp(PolicyApp):
        def chat(self, messages, *, on_delta=None, **kwargs):
            reply = super().chat(messages, **kwargs)
            if on_delta is not None:
                for character in self.content:
                    on_delta(character)
            return reply

    app = ProseApp(content=content)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    if stream:
        assert stream_calls(body) == {}
        content = "".join(json.loads(line[5:])["choices"][0]["delta"].get("content", "")
                          for line in body.splitlines() if line.startswith("data:") and line != "data: [DONE]")
    else:
        message = json.loads(body)["choices"][0]["message"]
        assert not message.get("tool_calls")
        content = message["content"]
    assert content == visible
