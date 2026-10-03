"""vLLM's /tokenize and /detokenize on both servers, a completion prompt given as token ids on CUDA, and OpenAI's
``param`` on context_length_exceeded. The fakes run everywhere; the checkpoint cases need the tokenizer and chat
template in the Hugging Face cache (as ``test_prompt_parity``) and check that /tokenize gives the chat route's own
prompt ids on both servers."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tensorfold.server.errors import CONTEXT_LIMIT, ContextLengthError, RequestError
from tests.test_cuda_admission import http_server
from tests.test_cuda_tool_choice import EOS, THINK, Tokens
from tests.test_server_openai_compat import FakeApp, serve_fake

TEMPLATE = ("{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% if enable_thinking %}<think>{% endif %}{% endif %}")
HI = [{"role": "user", "content": "Hi"}]


class Engine:
    eos = (EOS,)

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        on_tokens([ord("o"), ord("k")])
        return {"rounds": 1}


class VocabTokens(Tokens):
    """The fake tokenizer with a vocabulary size and token strings, as ``tokenizers.Tokenizer`` has them."""

    def get_vocab_size(self, with_added_tokens=True):
        return 1100

    def id_to_token(self, i):
        return "<think>" if i == THINK else chr(i)


def cuda_app(tmp_path, window=0):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(), "fake-cuda", VocabTokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 8
    app.native_context_window, app.context_window = 64, window
    app.lock = threading.Lock()
    return app


def call(port, route, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode() or "null")
    finally:
        connection.close()


def ids(text):
    return Tokens().encode(text).ids


# -- CUDA -----------------------------------------------------------------------------------------


def test_cuda_tokenize_gives_the_chat_route_s_prompt(tmp_path):
    app = cuda_app(tmp_path, window=32)
    with http_server(app) as port:
        status, reply = call(port, "/tokenize", {"messages": HI})
        assert status == 200
        assert reply == {"count": 19, "max_model_len": 32, "tokens": ids("user:Hi;assistant:") + [THINK]}
        assert reply["tokens"] == app.prepare({"messages": HI}, True).prompt
        status, off = call(port, "/v1/tokenize", {"messages": HI, "chat_template_kwargs": {"enable_thinking": False},
                                                   "add_generation_prompt": False, "return_token_strs": True})
        assert status == 200 and off["tokens"] == ids("user:Hi;") and off["token_strs"] == list("user:Hi;")
        status, text = call(port, "/v1/tokenize/", {"prompt": "a<think>b"})
        assert status == 200 and text["tokens"] == [ord("a"), THINK, ord("b")] and text["count"] == 3
        status, back = call(port, "/detokenize", {"tokens": text["tokens"]})
        assert status == 200 and back == {"prompt": "a<think>b"}           # special tokens included
        assert call(port, "/v1/detokenize", {"tokens": [[104, 105]]}) == (200, {"prompt": "hi"})
    assert not app.engine.prompts                                           # nothing ran


@pytest.mark.parametrize("route, body, words", [
    ("/tokenize", {"prompt": 3}, "prompt must be a string"),
    ("/tokenize", {"prompt": "x", "add_special_tokens": "yes"}, "add_special_tokens must be a boolean"),
    ("/tokenize", {"messages": HI, "add_generation_prompt": 1}, "add_generation_prompt must be a boolean"),
    ("/tokenize", {"messages": "Hi"}, "messages must be a list"),
    ("/tokenize", [1, 2], "must be a JSON object"),
    ("/detokenize", {"tokens": "abc"}, "tokens must be a list of integer token ids"),
    ("/detokenize", {"tokens": [1, True]}, "tokens must be a list of integer token ids"),
    ("/detokenize", {"tokens": [5, 1100]}, "range 0 to 1099"),
    ("/detokenize", {"tokens": [-1]}, "range 0 to 1099"),
])
def test_cuda_tokenizer_routes_refuse_malformed_bodies(tmp_path, route, body, words):
    with http_server(cuda_app(tmp_path)) as port:
        status, reply = call(port, route, body)
    assert status == 400 and words in reply["error"]["message"]
    assert reply["error"]["type"] == "invalid_request_error"


def test_cuda_completion_prompt_as_token_ids_is_served_as_given(tmp_path):
    app = cuda_app(tmp_path)
    prompt = [ord("a"), THINK, ord("b")]
    with http_server(app) as port:
        for given in (prompt, [prompt]):
            status, reply = call(port, "/v1/completions", {"prompt": given, "max_tokens": 2})
            assert status == 200 and reply["usage"]["prompt_tokens"] == 3
        status, reply = call(port, "/v1/completions", {"prompt": [ord("a"), 5000], "max_tokens": 2})
        assert status == 400 and "range 0 to 1099" in reply["error"]["message"]
    assert app.engine.prompts == [prompt, prompt]
    text = app.prepare({"prompt": "ab"}, False).prompt                      # text prompts as before: no specials
    assert text == [ord("a"), ord("b")]


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_cuda_context_errors_name_the_field(tmp_path, chat, stream):
    app = cuda_app(tmp_path, window=16)
    body = ({"messages": [{"role": "user", "content": "x" * 30}]} if chat else {"prompt": "x" * 30})
    with http_server(app) as port:
        status, reply = call(port, "/v1/chat/completions" if chat else "/v1/completions", {**body, "stream": stream})
    error = reply["error"]
    assert status == 400 and error["code"] == "context_length_exceeded" and error["message"].startswith(CONTEXT_LIMIT)
    assert error["param"] == ("messages" if chat else "prompt")


def test_an_error_without_a_code_has_no_param():
    from tensorfold.server.errors import error_body

    assert error_body(RequestError("bad"), "messages") == {"message": "bad", "type": "invalid_request_error"}
    assert error_body(ContextLengthError("long"), "prompt") == {
        "message": "long", "type": "invalid_request_error", "param": "prompt", "code": "context_length_exceeded"}


def test_glm_s_own_context_refusal_is_context_length_exceeded(tmp_path):
    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda.app import GlmApp

    app = cuda_app(tmp_path)
    app.__class__ = GlmApp
    app.engine.limit = 12
    app.engine.request = SimpleNamespace(policy=None, stop_eos=True)
    with http_server(app) as port:
        status, reply = call(port, "/v1/chat/completions", {"messages": HI, "max_tokens": 4})
    error = reply["error"]
    assert status == 400 and error["code"] == "context_length_exceeded" and error["param"] == "messages"
    assert error["message"].startswith(f"{CONTEXT_LIMIT} 12 tokens: this request needs a 23-token context")


# -- Mac ------------------------------------------------------------------------------------------


class MacTokenizer:
    """Characters as ids, rendered by a fixed chat template, as a Hugging Face tokenizer answers."""

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 300

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        text = "".join(f"{m['role']}:{m['content']};" for m in messages)
        if kwargs.get("add_generation_prompt"):
            text += "assistant:" + ("<" if kwargs.get("enable_thinking") else "")
        return [ord(c) for c in text]

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join("<s>" if t == 1 else chr(t) for t in tokens if not (skip_special_tokens and t == 1))

    def convert_ids_to_tokens(self, tokens):
        return ["<s>" if t == 1 else chr(t) for t in tokens]


class MacApp(FakeApp):
    late_system = "system"
    enable_thinking = False
    effort_levels = frozenset()
    context_window = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = MacTokenizer()

    def effort_for(self, explicit):
        return explicit

    def chat(self, messages, **kwargs):
        if messages and "x" * 30 in str(messages[-1].get("content")):
            raise ContextLengthError(f"{CONTEXT_LIMIT} 16 tokens: too long")
        return super().chat(messages, **{k: v for k, v in kwargs.items() if k in ("max_tokens", "on_delta")})


def test_mac_tokenizer_routes():
    app = MacApp()
    httpd = serve_fake(app)
    try:
        status, body = _post(httpd, "/v1/tokenize", {"messages": HI})
        assert status == 200
        assert body == {"count": 18, "max_model_len": None, "tokens": [ord(c) for c in "user:Hi;assistant:"]}
        status, body = _post(httpd, "/tokenize", {"messages": HI, "add_generation_prompt": False,
                                                  "chat_template_kwargs": {"enable_thinking": True}})
        assert body["tokens"] == [ord(c) for c in "user:Hi;"] and app.tokenizer.calls[-1]["enable_thinking"] is True
        status, body = _post(httpd, "/tokenize", {"prompt": "ab", "return_token_strs": True})
        assert body["tokens"] == [1, 97, 98] and body["token_strs"] == ["<s>", "a", "b"]
        status, body = _post(httpd, "/tokenize", {"prompt": "ab", "add_special_tokens": False})
        assert body["tokens"] == [97, 98]
        assert _post(httpd, "/detokenize", {"tokens": [1, 104, 105]}) == (200, {"prompt": "<s>hi"})
        status, body = _post(httpd, "/detokenize", {"tokens": [300]})
        assert status == 400 and "range 0 to 299" in body["error"]["message"]
        status, body = _post(httpd, "/tokenize", {"prompt": "x", "add_special_tokens": None})
        assert status == 400 and "add_special_tokens must be a boolean" in body["error"]["message"]
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("chat", [False, True])
def test_mac_context_errors_name_the_field(chat):
    httpd = serve_fake(MacApp())
    try:
        long = "x" * 30
        body = {"messages": [{"role": "user", "content": long}]} if chat else {"prompt": long}
        status, reply = _post(httpd, "/v1/chat/completions" if chat else "/v1/completions", body)
    finally:
        httpd.shutdown()
    error = reply["error"]
    assert status == 400 and error["code"] == "context_length_exceeded"
    assert error["param"] == ("messages" if chat else "prompt")


def _post(httpd, route, body):
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    try:
        connection.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode())
    finally:
        connection.close()
