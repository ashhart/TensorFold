import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda import server
from tests.test_cuda_admission import http_server, post
from tests.test_request_policy import CALL, TOOLS, stream_calls


class TextTokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(c) for c in ids)


class Engine:
    eos = (0,)

    def __init__(self, content):
        self.content = content
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append((list(prompt), max_tokens, draft))
        on_tokens([ord(c) for c in self.content])
        return {"generated": len(self.content)}


def app_for(tmp_path, content="Hello"):
    template = ("{% for m in messages %}{% if m.role == 'developer' or "
                "(m.role == 'system' and not loop.first) %}{{ raise_exception('unsupported role') }}"
                "{% endif %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine = Engine(content)
    app.served = "fake-cuda"
    app.tok = TextTokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.vision = None                      # served without --vision
    return app


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_developer_and_multiple_system_messages(tmp_path, stream):
    app = app_for(tmp_path)
    messages = [{"role": "developer", "content": "First"},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Second"}]
    with http_server(app) as port:
        status, body = post(port, {"messages": messages, "stream": stream}, True)
    assert status == 200
    if stream:
        stream_calls(body)
    # this template rejects a later system message, so it keeps its place as a user turn
    assert app.tok.decode(app.engine.calls[0][0]) == "system:First;user:Hi;user:Second;assistant:"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_image_refuses_before_headers_and_generate(tmp_path, stream):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/picture.png"}}]}],
            "stream": stream}, True)
    assert status == 400 and app.engine.calls == []
    assert "--vision" in json.loads(body)["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_cuda_tool_limit_keeps_generation_and_usage_unchanged(tmp_path, stream, parallel):
    content = CALL.format(1) + CALL.format(2)
    app = app_for(tmp_path, content)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Measure"}],
            "tools": TOOLS, "stream": stream, "parallel_tool_calls": parallel,
            "stream_options": {"include_usage": True}}, True)
    assert status == 200
    if stream:
        calls = stream_calls(body)
        usage = [json.loads(line[5:])["usage"] for line in body.splitlines()
                 if line.startswith("data:") and line != "data: [DONE]" and '"usage"' in line][0]
    else:
        payload = json.loads(body)
        calls = payload["choices"][0]["message"]["tool_calls"]
        usage = payload["usage"]
    assert len(calls) == (2 if parallel else 1)
    assert usage["completion_tokens"] == len(content)
    assert app.engine.calls[0][2] is True


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_bad_parallel_flag_refuses_before_generate(tmp_path, stream):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Hi"}],
                                 "parallel_tool_calls": "false", "stream": stream}, True)
    assert status == 400 and app.engine.calls == []


def test_cuda_template_does_not_mutate_tool_call_history(tmp_path):
    app = app_for(tmp_path)
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "measure", "arguments": '{"value":1}'}}]}, {"role": "user", "content": "Continue"}]
    app.template.render(messages, tools=TOOLS, enable_thinking=False)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"value":1}'


def test_cuda_single_call_skips_broken_parameters(tmp_path):
    broken = '<tool_call><function=measure><parameter=value>broken</function></tool_call>'
    app = app_for(tmp_path, broken + CALL.format(2))
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Measure"}],
                                 "tools": TOOLS, "parallel_tool_calls": False}, True)
    assert status == 200
    arguments = json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"value": "2"}


@pytest.mark.parametrize("options", [{"images": ["https://example.com/picture.png"]}, {"modalities": ["audio"]}])
def test_cuda_requested_media_refuses_before_generate(tmp_path, options):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Hi"}],
                                 "stream": True, **options}, True)
    assert status == 400 and app.engine.calls == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":NaN}', '"{\\"value\\":1e999}"'])
def test_cuda_single_call_skips_nonfinite_json_arguments(tmp_path, stream, arguments):
    bad = '<tool_call>{"name":"measure","arguments":' + arguments + '}</tool_call>'
    app = app_for(tmp_path, bad + CALL.format(2))
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Measure"}],
            "tools": TOOLS, "stream": stream, "parallel_tool_calls": False}, True)
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    arguments = calls[0]["arguments"] if stream else calls[0]["function"]["arguments"]
    assert len(calls) == 1 and json.loads(arguments) == {"value": "2"}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content, visible", [
    ('Before<m:tool_call><function=measure><parameter=value>1', 'Before'),
    ('Before<m', 'Before<m'), ('Before<math>x</math>', 'Before<math>x</math>'),
])
def test_cuda_single_call_hides_incomplete_namespaced_envelopes(tmp_path, stream, content, visible):
    app = app_for(tmp_path, content)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Measure"}],
            "tools": TOOLS, "stream": stream, "parallel_tool_calls": False}, True)
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
