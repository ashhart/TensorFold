"""Messages protocol integration tests: real HTTP handlers with deterministic test engines, without model inference."""
import json
from contextlib import contextmanager

import pytest

from tensorfold.server.anthropic_translate import Reply, translate, usage
from tensorfold.server.errors import RequestError
from tests.test_cuda_admission import http_server
from tests.test_cuda_tool_choice import Engine, app_for
from tests.test_lane_server import make_app
from tests.test_responses_api import call, stream_events
from tests.test_server_openai_compat import FakeApp, serve_fake

BASE = {"model": "local-model", "max_tokens": 128, "messages": [{"role": "user", "content": "Hi"}]}


@contextmanager
def serving(backend, tmp_path, *, fail=False):
    if backend == "cuda":
        app = app_for(tmp_path, Engine())
        with http_server(app) as port:
            yield app, port
    else:
        app = FakeApp(reasoning="hmm", fail_stream=fail)
        server = serve_fake(app)
        try:
            yield app, server.server_port
        finally:
            server.shutdown()
            server.server_close()


def reconstruct(events):
    assert events[0]["type"] == "message_start"
    assert events[-1]["type"] == "message_stop"
    blocks, open_blocks = [], set()
    for event in events[1:-2]:
        kind = event["type"]
        at = event["index"]
        if kind == "content_block_start":
            assert at == len(blocks) and not open_blocks
            blocks.append(dict(event["content_block"]))
            open_blocks.add(at)
            if blocks[at]["type"] == "tool_use":
                blocks[at]["input"] = ""
        elif kind == "content_block_delta":
            assert at in open_blocks
            delta, block = event["delta"], blocks[at]
            mapping = {"text_delta": ("text", "text"), "thinking_delta": ("thinking", "thinking"),
                       "signature_delta": ("thinking", "signature"), "input_json_delta": ("tool_use", "partial_json")}
            expected, key = mapping[delta["type"]]
            assert block["type"] == expected
            target = "input" if key == "partial_json" else key
            block[target] = block.get(target, "") + delta[key]
        elif kind == "content_block_stop":
            open_blocks.remove(at)
            if blocks[at]["type"] == "tool_use":
                blocks[at]["input"] = json.loads(blocks[at]["input"])
        else:
            raise AssertionError(kind)
    assert not open_blocks
    assert events[-2]["type"] == "message_delta"
    return blocks


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("stream", [False, True])
def test_text_and_thinking_over_http(backend, stream, tmp_path):
    with serving(backend, tmp_path) as (_, port):
        status, body = call(port, "POST", "/v1/messages?beta=true", {**BASE, "stream": stream,
                            "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
    assert status == 200
    if stream:
        events = stream_events(body)
        blocks = reconstruct(events)
        assert any(b["type"] == "thinking" and b["thinking"] for b in blocks)
        assert any(b["type"] == "text" and b["text"] for b in blocks)
        assert events[-2]["delta"]["stop_reason"] == "end_turn"
        assert events[-2]["usage"]["output_tokens"] > 0
    else:
        payload = json.loads(body)
        assert payload["type"] == "message" and payload["role"] == "assistant"
        assert payload["content"] and payload["usage"]["output_tokens"] > 0
        assert payload["stop_reason"] == "end_turn"


def test_parallel_tool_history_and_error_result_are_preserved():
    body = {**BASE, "system": [{"type": "text", "text": "Rules", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": ""},
                {"type": "tool_use", "id": "a", "name": "weather", "input": {"city": "Taipei"}},
                {"type": "tool_use", "id": "b", "name": "weather", "input": {"city": "Oslo"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "sun"}]},
                    {"type": "tool_result", "tool_use_id": "b", "content": "offline", "is_error": True},
                    {"type": "text", "text": "Summarize"}]}]}
    chat = translate(body)
    assert [m["role"] for m in chat["messages"]] == ["system", "assistant", "tool", "tool", "user"]
    assistant = chat["messages"][1]
    assert assistant["reasoning_content"] == "plan"
    assert [json.loads(c["function"]["arguments"])["city"] for c in assistant["tool_calls"]] == ["Taipei", "Oslo"]
    assert chat["messages"][3]["content"] == "Tool error: offline"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_required_tool_and_result_round_trip(stream, tmp_path):
    tools = [{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    with serving("cuda", tmp_path) as (_, port):
        status, raw = call(port, "POST", "/messages/", {**BASE, "tools": tools, "stream": stream,
                                                       "tool_choice": {"type": "tool", "name": "get_weather"}})
        assert status == 200
        if stream:
            events = stream_events(raw)
            blocks = reconstruct(events)
            assert events[-2]["delta"]["stop_reason"] == "tool_use"
        else:
            response = json.loads(raw)
            blocks = response["content"]
            assert response["stop_reason"] == "tool_use"
        block = next(b for b in blocks if b["type"] == "tool_use")
        assert block["name"] == "get_weather" and block["input"] == {"city": "Oslo"}
        history = [*BASE["messages"], {"role": "assistant", "content": blocks}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block["id"], "content": "sunny"}]}]
        status, raw = call(port, "POST", "/v1/messages", {**BASE, "messages": history, "tools": tools,
                                                        "tool_choice": {"type": "none"}})
        assert status == 200 and json.loads(raw)["stop_reason"] == "end_turn"


def test_interleaved_tool_arguments_do_not_receive_text_deltas():
    events = []
    reply = Reply("model", events.append)
    reply.start()
    def chunk(delta):
        reply.chunk({"choices": [{"delta": delta}]})
    chunk({"reasoning_content": "plan"})
    chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "first", "arguments": '{"x":'}}]})
    chunk({"content": "\n"})
    chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "second", "arguments": "{}"}}]})
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]})
    reply.chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    reply.chunk(None)
    blocks = reconstruct(events)
    assert [b["input"] for b in blocks if b["type"] == "tool_use"] == [{"x": 1}, {}]


def test_cache_usage_is_not_double_counted():
    result = usage({"prompt_tokens": 90, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 64}})
    assert result == {"input_tokens": 26, "output_tokens": 12, "cache_read_input_tokens": 64,
                      "cache_creation_input_tokens": 0}


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("patch", [
    {"max_tokens": 0}, {"max_tokens": True}, {"stream": "yes"}, {"messages": []},
    {"messages": [{"role": "user", "content": [{"type": "document"}]}]},
    {"thinking": {"type": "enabled", "budget_tokens": 150}},
    {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    {"tool_choice": {"type": "bogus"}}, {"output_config": {"effort": "bogus"}},
    {"stop_sequences": "END"},
])
def test_request_errors_have_anthropic_shape_before_streaming(backend, patch, tmp_path):
    with serving(backend, tmp_path) as (_, port):
        status, raw = call(port, "POST", "/v1/messages", {**BASE, "stream": True, **patch})
    problem = json.loads(raw)
    assert status == 400 and problem["type"] == "error" and problem["error"]["type"] == "invalid_request_error"


def test_stream_failure_is_an_error_not_success(tmp_path):
    with serving("mlx", tmp_path, fail=True) as (_, port):
        status, raw = call(port, "POST", "/v1/messages", {**BASE, "stream": True})
    events = stream_events(raw)
    assert status == 200 and events[-1]["type"] == "error"
    assert events[-1]["error"]["type"] == "api_error"
    assert all(e["type"] != "message_stop" for e in events)


def test_token_count_uses_cuda_prompt_without_generation(tmp_path):
    with serving("cuda", tmp_path) as (app, port):
        body = {k: v for k, v in BASE.items() if k != "max_tokens"}
        status, raw = call(port, "POST", "/v1/messages/count_tokens", body)
        assert status == 200
        assert json.loads(raw)["input_tokens"] == len(app.prepare(translate(body, count=True), True).prompt)
        assert not app.engine.calls


def test_image_sources_and_tool_result_images():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
    result = translate({**BASE, "messages": [{"role": "user", "content": [image]}]})
    assert result["messages"][0]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    result = translate({**BASE, "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "screenshot"}, image]}]}]})
    assert [m["role"] for m in result["messages"]] == ["tool"]
    assert result["messages"][0]["tool_call_id"] == "a"
    assert result["messages"][0]["content"] == [
        {"type": "text", "text": "screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]


def test_parallel_tool_images_keep_their_tool_id_and_content_order():
    image = lambda data: {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": data}}
    body = {**BASE, "messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": [
            image("AA"), {"type": "text", "text": "first"}]},
        {"type": "tool_result", "tool_use_id": "b", "content": [
            {"type": "text", "text": "second"}, image("BB")]},
        {"type": "text", "text": "Compare them"},
    ]}]}
    original = json.loads(json.dumps(body))
    messages = translate(body)["messages"]
    assert [m["role"] for m in messages] == ["tool", "tool", "user"]
    assert [m["tool_call_id"] for m in messages[:2]] == ["a", "b"]
    assert messages[0]["content"] == [
        {"type": "text", "text": "Tool error: "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        {"type": "text", "text": "first"},
    ]
    assert messages[1]["content"] == [
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}},
    ]
    assert messages[2]["content"] == [{"type": "text", "text": "Compare them"}]
    assert body == original


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_tool_image_reaches_the_vision_renderer_as_its_tool_result(backend, tmp_path):
    from types import SimpleNamespace

    from tensorfold.server.messages import normalize_messages
    from tests.test_vision_server import Frontend, Tokenizer, image_messages

    url = image_messages()[0]["content"][1]["image_url"]["url"]
    body = {**BASE, "messages": [{"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "shot", "content": [
            {"type": "text", "text": "PAGE"},
            {"type": "image", "source": {"type": "url", "url": url}},
        ],
    }]}]}
    rendered = []
    with serving(backend, tmp_path) as (app, port):
        app.vision = Frontend()
        if backend == "cuda":
            def render(messages, **kwargs):
                rendered.append(normalize_messages(messages, allow_images=kwargs.get("allow_images", False)))
                return "rendered image prompt"
            app.template = SimpleNamespace(render=render)
        else:
            app.tokenizer = Tokenizer()
            app.effort_levels, app.enable_thinking, app.reasoning_effort = frozenset(), False, None
            app.late_system, app.context_window = "system", 32
        status, raw = call(port, "POST", "/v1/messages/count_tokens", body)
        assert status == 200 and json.loads(raw)["input_tokens"] == 4
        template = rendered[0] if backend == "cuda" else app.tokenizer.calls[0][0]
        assert template == [{"role": "tool", "tool_call_id": "shot", "content": [
            {"type": "text", "text": "PAGE"}, {"type": "image", "detail": "auto"},
        ]}]
        assert len(app.vision.calls[0][1]) == 1


def test_thinking_controls_and_output_schema():
    request = translate({**BASE, "thinking": {"type": "enabled", "budget_tokens": 20},
                         "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}})
    assert request["thinking_budget"] == 20 and request["chat_template_kwargs"]["enable_thinking"]
    assert request["response_format"]["json_schema"]["schema"] == {"type": "object"}
    request = translate({**BASE, "thinking": {"type": "disabled"}, "output_config": {"effort": "high"}})
    assert request["reasoning_effort"] == "none" and not request["chat_template_kwargs"]["enable_thinking"]


def test_non_object_request_is_refused():
    with pytest.raises(RequestError):
        translate([])


@pytest.mark.parametrize("stream", [False, True])
def test_stop_sequence_returns_the_match_and_strips_it(stream, tmp_path):
    with serving("cuda", tmp_path) as (_, port):
        status, raw = call(port, "POST", "/v1/messages", {**BASE, "stream": stream, "stop_sequences": [" How"]})
    assert status == 200
    if stream:
        events = stream_events(raw)
        content = reconstruct(events)
        delta = events[-2]["delta"]
    else:
        reply = json.loads(raw)
        content, delta = reply["content"], reply
    assert delta["stop_reason"] == "stop_sequence" and delta["stop_sequence"] == " How"
    assert content == [{"type": "text", "text": "Hello!"}]


def test_claude_keep_all_thinking_is_accepted_and_other_edits_are_refused():
    body = {**BASE, "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}}
    assert translate(body)["messages"] == translate(BASE)["messages"]
    with pytest.raises(RequestError):
        translate({**BASE, "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}})


def test_tool_arguments_stream_before_completion():
    events = []
    reply = Reply("model", events.append)
    reply.start()
    reply.chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {
        "name": "write", "arguments": '{"content":"hello'}}]}}]})
    assert events[-1]["delta"] == {"type": "input_json_delta", "partial_json": '{"content":"hello'}
    reply.chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]},
                             "finish_reason": "tool_calls"}]})
    reply.chunk(None)
    assert reconstruct(events) == [{"type": "tool_use", "id": "a", "name": "write", "input": {"content": "hello"}}]


@pytest.mark.parametrize("thinking", [{"type": "disabled"}, {"type": "adaptive"}])
def test_mlx_token_count_renders_without_generation(thinking):
    app = make_app(enable_thinking=False, reasoning_effort="medium")
    def refuse_generation(*args, **kwargs):
        raise AssertionError("token counting must not generate")
    app.chat = refuse_generation
    server = serve_fake(app)
    try:
        body = {**BASE, "max_tokens": 2, "thinking": thinking, "output_config": {"effort": "low"}}
        status, raw = call(server.server_port, "POST", "/v1/messages/count_tokens", body)
        assert status == 200
        count = json.loads(raw)["input_tokens"]
        # The model's char tokenizer renders Hi, one message delimiter and the two-token generation marker.
        assert count == 5
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_thinking_token_usage_is_preserved():
    result = usage({"prompt_tokens": 90, "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 12}})
    assert result["output_tokens"] == 20 and result["output_tokens_details"] == {"thinking_tokens": 12}


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_thinking_is_opt_in(thinking):
    chat = translate({**BASE, "thinking": thinking, "output_config": {"effort": "high"}})
    assert chat["chat_template_kwargs"]["enable_thinking"] is False
    assert chat["reasoning_effort"] == "none"


@pytest.mark.parametrize("tool_result", [False, True])
@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_mid_conversation_system_after_user_or_tool_result(backend, tool_result, tmp_path):
    history = [{"role": "user", "content": "Read the file"}]
    if tool_result:
        history.extend([{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "read", "input": {}}]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "data"}]}])
    before = translate({**BASE, "system": "Top-level rules", "messages": history})["messages"]
    history.extend([{"role": "system", "content": [{"type": "text", "text": "New instructions"}]},
                    {"role": "system", "content": "More instructions"}])
    translated = translate({**BASE, "system": "Top-level rules", "messages": history})["messages"]
    assert translated[:len(before)] == before
    assert [m["role"] for m in translated[-2:]] == ["system", "system"]
    with serving(backend, tmp_path) as (_, port):
        status, raw = call(port, "POST", "/v1/messages", {**BASE, "messages": history})
        assert status == 200 and json.loads(raw)["stop_reason"] == "end_turn"


@pytest.mark.parametrize("history", [
    [{"role": "system", "content": "rules"}, {"role": "user", "content": "hello"}],
    [{"role": "assistant", "content": "hello"}, {"role": "system", "content": "rules"}],
    [{"role": "user", "content": "hello"}, {"role": "system", "content": "rules"}, {"role": "user", "content": "again"}],
    [{"role": "user", "content": "hello"}, {"role": "system", "content": [{"type": "image", "source": {}}]}],
    [{"role": "user", "content": "hello"}, {"role": "system", "content": "rules", "clear_at": "next_user_message"}],
])
def test_invalid_mid_conversation_system_is_rejected(history):
    with pytest.raises(RequestError):
        translate({**BASE, "messages": history})


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_mid_conversation_system_does_not_fall_back_to_user(path, tmp_path):
    with serving("mlx", tmp_path) as (app, port):
        app.late_system = "user"
        status, raw = call(port, "POST", path, {**BASE, "messages": [*BASE["messages"],
                            {"role": "system", "content": "rules"}]})
        assert status == 400 and "chat template" in json.loads(raw)["error"]["message"]
