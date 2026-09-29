"""CUDA server errors: malformed requests get 400 before any stream opens; failed ones get 500 or an error event."""

import http.client
import json
import math
import re
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tensorfold.engine.exact_sampling import Sampling, seed_for
from tests.test_cuda_admission import http_server

# a Qwen checkpoint template's own refusals: an unknown reasoning effort or mode, a history without a user message
TEMPLATE = (
    "{%- set effort = reasoning_effort|default('xhigh') %}"
    "{%- if effort not in ('xhigh', 'medium', 'low') %}"
    "{{ raise_exception('Unexpected reasoning effort ' ~ effort ~ '.') }}{% endif %}"
    "{%- if mode is defined and mode != 'plain' %}{{ raise_exception('Unexpected mode ' ~ mode ~ '.') }}{% endif %}"
    "{%- set ns = namespace(user=false) %}"
    "{%- for m in messages %}{% if m.role == 'user' %}{% set ns.user = true %}{% endif %}{% endfor %}"
    "{%- if not ns.user %}{{ raise_exception('No user query found in messages.') }}{% endif %}"
    "{%- for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
)
DEFAULTS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
HI = [{"role": "user", "content": "Hi"}]
EOS = 0


class TextTokenizer:
    def encode(self, text, **kwargs):
        if "BOOM" in text:
            raise RuntimeError("tokenizer stand-in failed")
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(c) for c in ids)


class Engine:
    eos = (EOS,)

    def __init__(self):
        self.calls = []
        self.fail = None          # an exception to raise from generate
        self.before_fail = ""     # text emitted before it is raised

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append({"prompt": list(prompt), "max_tokens": max_tokens, "sampling": sampling, "draft": draft})
        if self.before_fail:
            on_tokens([ord(c) for c in self.before_fail])
        if self.fail is not None:
            raise self.fail
        on_tokens([ord(c) for c in "Hello"] + [EOS])
        return {"generated": 6}


def app_for(tmp_path, cls=server.App):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    app = cls.__new__(cls)
    app.engine = Engine()
    app.served = "fake-cuda"
    app.tok = TextTokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = dict(DEFAULTS)
    app.max_tokens = 64
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def request(port, body=None, *, chat=True, raw=None):
    """(status, content type, body text); raises if the server closes the connection without a response."""

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        route = "/v1/chat/completions" if chat else "/v1/completions"
        data = raw if raw is not None else json.dumps(body).encode()
        connection.request("POST", route, data, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read().decode()
    finally:
        connection.close()


def events(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(line[6:] if line == "data: [DONE]" else json.loads(line[6:]))
    return out


def sampling_before(defaults, body, prompt):
    """``App.sampling_for`` as it was before requests' sampling fields were checked: the reference for valid ones."""

    temp = float(body["temperature"] if body.get("temperature") is not None else defaults["temperature"])
    if temp <= 0:
        return None
    seed = body.get("seed")
    top_k = body["top_k"] if body.get("top_k") is not None else defaults["top_k"]
    top_p = body["top_p"] if body.get("top_p") is not None else defaults["top_p"]
    return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))


MALFORMED = {
    "template raises": ({"messages": [{"role": "tool", "content": "x"}]},
                        "the chat template rejected the request: No user query found in messages."),
    "template raises on a kwarg": ({"messages": HI, "chat_template_kwargs": {"mode": "fancy"}},
                                   "the chat template rejected the request: Unexpected mode fancy."),
    "an unknown effort": ({"messages": HI, "chat_template_kwargs": {"reasoning_effort": "extreme"}},
                          "reasoning_effort must be none, minimal, low, medium, high or xhigh"),
    **{f"chat_template_kwargs {name}": ({"messages": HI, "chat_template_kwargs": value},
                                        "chat_template_kwargs must be a JSON object or null")
       for name, value in [("[]", []), ('""', ""), ("false", False), ("0", 0),              # falsy: not read as absent
                           ("true", True), ("a number", 1.5), ("a string", "x"),
                           ("a JSON string", '{"enable_thinking": true}'),
                           ("a list", [{"enable_thinking": True}]),
                           ("a list of pairs", [["enable_thinking", True]])]},
    "temperature 'hot'": ({"messages": HI, "temperature": "hot"}, "temperature must be a finite number or null"),
    "temperature true": ({"messages": HI, "temperature": True}, "temperature must be a finite number or null"),
    "temperature NaN": ({"messages": HI, "temperature": math.nan}, "temperature must be a finite number or null"),
    "top_p 'inf'": ({"messages": HI, "top_p": "inf"}, "top_p must be a finite number or null"),
    "top_k 2.5": ({"messages": HI, "top_k": 2.5}, "top_k must be an integer or null"),
    "top_k a list": ({"messages": HI, "top_k": [20]}, "top_k must be an integer or null"),
    "seed 'abc'": ({"messages": HI, "temperature": 0.7, "seed": "abc"}, "seed must be an integer or null"),
    "seed 'abc', greedy": ({"messages": HI, "temperature": 0, "seed": "abc"}, "seed must be an integer or null"),
    "seed 1.5": ({"messages": HI, "seed": 1.5}, "seed must be an integer or null"),
    "completion, temperature 'hot'": ({"prompt": "Hi", "temperature": "hot"},
                                      "temperature must be a finite number or null"),
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_requests_get_400_before_any_stream_or_generate(tmp_path, case, stream):
    body, message = MALFORMED[case]
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, kind, text = request(port, {**body, "stream": stream}, chat="messages" in body)
        after = request(port, {"messages": HI, "max_tokens": 8, "stream": stream})
    assert (status, kind) == (400, "application/json")
    assert json.loads(text) == {"error": {"message": message, "type": "invalid_request_error"}}
    assert after[0] == 200 and len(app.engine.calls) == 1


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_check_reports_what_prepare_refuses(tmp_path, case):
    body, message = MALFORMED[case]
    app = app_for(tmp_path)
    assert app.check(body) == message
    with pytest.raises(server.RequestError, match="^" + re.escape(message) + "$"):
        app.prepare(body, "messages" in body)


@pytest.mark.parametrize("value", [None, {}], ids=["null", "{}"])
def test_chat_template_kwargs_null_or_empty_object_is_read_as_absent(tmp_path, value):
    app = app_for(tmp_path)
    body = {"messages": HI, "chat_template_kwargs": value}
    assert app.check(body) is None
    prepared, absent = app.prepare(body, True), app.prepare({"messages": HI}, True)
    assert (prepared.prompt, prepared.thinking) == (absent.prompt, absent.thinking)
    with http_server(app) as port:
        assert [request(port, {**body, "stream": stream})[0] for stream in (False, True)] == [200, 200]
    assert [call["prompt"] for call in app.engine.calls] == [absent.prompt, absent.prompt]


def test_glm_check_reports_what_prepare_refuses(tmp_path):
    from tensorfold.families.glm5_next.cuda.app import GlmApp, ThinkingOffTemplate

    app = app_for(tmp_path, GlmApp)
    app.template = ThinkingOffTemplate(app.template)
    app.engine.limit = 4096
    app.engine.request = SimpleNamespace(policy=None, stop_eos=True)
    for case in ("template raises", "chat_template_kwargs a string", "chat_template_kwargs []",
                 "chat_template_kwargs 0", "temperature 'hot'", "seed 'abc'"):
        body, message = MALFORMED[case]
        assert app.check(body) == message
    assert app.check({"messages": HI, "temperature": 0.5}) is None


@pytest.mark.parametrize("stream", [False, True])
def test_unexpected_error_while_preparing_gets_400_and_is_logged(tmp_path, stream, capsys):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, kind, text = request(port, {"messages": [{"role": "user", "content": "BOOM"}], "stream": stream})
        after = request(port, {"messages": HI, "max_tokens": 8})
    assert (status, kind) == (400, "application/json")
    assert json.loads(text) == {"error": {"message": "tokenizer stand-in failed"}}
    assert after[0] == 200 and len(app.engine.calls) == 1
    err = capsys.readouterr()
    assert "[tensorfold] request error: RuntimeError: tokenizer stand-in failed" in err.out
    assert "Traceback" in err.err


def test_a_body_that_is_not_utf8_gets_400(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, kind, text = request(port, raw=b'{"messages": [{"role": "user", "content": "\xc3\x28"}]}')
    assert (status, kind) == (400, "application/json")
    assert json.loads(text)["error"]["message"] == "the request body is not JSON"


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("emitted", ["", "Par"])
def test_engine_failure_non_streamed_gets_500(tmp_path, chat, emitted, capsys):
    app = app_for(tmp_path)
    app.engine.fail, app.engine.before_fail = RuntimeError("CUDA out of memory (stand-in)"), emitted
    body = {"messages": HI} if chat else {"prompt": "Hi"}
    with http_server(app) as port:
        status, kind, text = request(port, body, chat=chat)
        app.engine.fail, app.engine.before_fail = None, ""
        after = request(port, {**body, "max_tokens": 8}, chat=chat)
    assert (status, kind) == (500, "application/json")
    assert json.loads(text) == {"error": {"message": "CUDA out of memory (stand-in)"}}
    assert after[0] == 200 and len(app.engine.calls) == 2      # the lock was released
    err = capsys.readouterr()
    assert "[tensorfold] request error: RuntimeError: CUDA out of memory (stand-in)" in err.out
    assert "Traceback" in err.err


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("emitted", ["", "Par"])
def test_engine_failure_streamed_gets_an_error_event_then_done(tmp_path, chat, emitted, capsys):
    app = app_for(tmp_path)
    app.engine.fail, app.engine.before_fail = RuntimeError("CUDA out of memory (stand-in)"), emitted
    body = {"messages": HI, "stream": True} if chat else {"prompt": "Hi", "stream": True}
    with http_server(app) as port:
        status, kind, text = request(port, body, chat=chat)
        app.engine.fail, app.engine.before_fail = None, ""
        after = request(port, {**body, "max_tokens": 8}, chat=chat)
    assert (status, kind) == (200, "text/event-stream")
    got = events(text)
    assert got[-2:] == [{"error": {"message": "CUDA out of memory (stand-in)", "type": "server_error"}}, "[DONE]"]
    chunks = got[:-2]
    if chat:
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:])
    else:
        streamed = "".join(c["choices"][0]["text"] for c in chunks)
    assert streamed == emitted
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks)
    assert after[0] == 200 and events(after[2])[-1] == "[DONE]" and len(app.engine.calls) == 2
    err = capsys.readouterr()
    assert "[tensorfold] request error: RuntimeError: CUDA out of memory (stand-in)" in err.out
    assert "Traceback" in err.err


def test_an_error_without_a_message_names_its_type(tmp_path, capsys):
    app = app_for(tmp_path)
    app.engine.fail = AssertionError()
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI})
    assert status == 500 and json.loads(text) == {"error": {"message": "AssertionError"}}
    capsys.readouterr()


VALID = [
    {},
    {"temperature": 0},
    {"temperature": 0.0, "seed": 5},
    {"temperature": -1},
    {"temperature": None, "top_k": None, "top_p": None, "seed": None},
    {"temperature": 0.7},
    {"temperature": 1},
    {"temperature": "0.7"},
    {"temperature": "0"},
    {"temperature": 0.6, "top_p": 0.8, "top_k": 40, "seed": 1234},
    {"temperature": 0.6, "top_p": "0.8", "top_k": "40", "seed": "1234"},
    {"top_k": 0},
    {"top_k": -3},
    {"top_k": 20.0},
    {"top_p": 1},
    {"seed": 0},
    {"seed": -7},
    {"seed": 42.0},
    {"seed": 2 ** 70},
    {"seed": "-12"},
]


SERVER_DEFAULTS = [DEFAULTS, {"temperature": 0.0, "top_k": 50, "top_p": 1.0},
                   {"temperature": "0.8", "top_k": "5", "top_p": "0.5"}]


@pytest.mark.parametrize("defaults", SERVER_DEFAULTS)
@pytest.mark.parametrize("fields", VALID, ids=[json.dumps(v) for v in VALID])
def test_valid_requests_reach_the_engine_with_the_same_sampling(tmp_path, fields, defaults):
    """Passes before and after the change: every request valid before generates with the Sampling it had."""

    app = app_for(tmp_path)
    app.sampling = dict(defaults)
    body = {"messages": HI, **fields}
    prompt = app.prepare(body, True).prompt
    expected = sampling_before(defaults, body, prompt)
    with http_server(app) as port:
        for stream in (False, True):
            status, _, _ = request(port, {**body, "stream": stream})
            assert status == 200
    assert [call["sampling"] for call in app.engine.calls] == [expected, expected]
    assert all(call["prompt"] == prompt for call in app.engine.calls)


@pytest.mark.parametrize("defaults", SERVER_DEFAULTS)
@pytest.mark.parametrize("fields", VALID, ids=[json.dumps(v) for v in VALID])
def test_prepare_resolves_the_same_sampling(tmp_path, fields, defaults):
    app = app_for(tmp_path)
    app.sampling = dict(defaults)
    body = {"messages": HI, **fields}
    prepared = app.prepare(body, True)
    expected = sampling_before(defaults, body, prepared.prompt)
    assert prepared.sampling == expected
    assert app.sampling_for(body, prepared.prompt) == expected


def test_run_uses_the_prepared_sampling(tmp_path):
    app = app_for(tmp_path)
    body = {"messages": HI, "temperature": 0.5, "seed": 9}
    prepared = app.prepare(body, True)
    app.run(body, True, lambda delta: True, prepared=prepared)
    assert app.engine.calls[0]["sampling"] is prepared.sampling
    assert prepared.sampling == Sampling(9, 0.5, 20, 0.95)
