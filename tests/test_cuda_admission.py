"""CUDA request admission and HTTP refusal receipts without model or device loads."""

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers

from tensorfold.cuda import server
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine


class Engine:
    eos = (0,)

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append((list(prompt), max_tokens))
        on_tokens([1])
        return {}


class FlashEngine(Engine):
    max_len = 20
    depth = 6
    context_window = FlashNextEngine.context_window

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        max_tokens = FlashNextEngine._limit(self, prompt, max_tokens)
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


@pytest.fixture
def model_dir(tmp_path):
    words = ["[UNK]", "answer", "header", "thinking", "tools", "extra"] + [f"w{i}" for i in range(40)]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))
    template = ("{% for m in messages %}{{ m.content }} {% endfor %}header"
                "{% if enable_thinking %} thinking{% endif %}{% if tools %} tools{% endif %}"
                "{% if suffix %} {{ suffix }}{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    return tmp_path


def words(count):
    return " ".join(f"w{i}" for i in range(count))


@contextmanager
def http_server(app):
    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass
    httpd = QuietServer(("127.0.0.1", 0), server.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def post(port, body, chat):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        route = "/v1/chat/completions" if chat else "/v1/completions"
        connection.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def request(count, chat, **options):
    prompt = {"messages": [{"role": "user", "content": words(count - 1)}]} if chat else {"prompt": words(count)}
    return {**prompt, "temperature": 0, **options}


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_flash_capacity_refusal_precedes_headers(model_dir, chat, stream):
    # The real Flash limiter refuses this boundary; the HTTP layer must refuse before starting a stream.
    engine = FlashEngine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(13, chat, max_tokens=1, stream=stream), chat)
    assert status == 400
    error = json.loads(payload)["error"]
    assert error["type"] == "invalid_request_error"
    assert "13-token safe cache capacity" in error["message"]
    assert "shorten the prompt" in error["message"]
    assert not engine.calls


@pytest.mark.parametrize("nested", [False, True])
def test_native_metadata_guards_engine_without_capacity(model_dir, nested):
    metadata = {"max_position_embeddings": 12}
    (model_dir / "config.json").write_text(json.dumps({"text_config": metadata} if nested else metadata))
    engine = Engine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(12, True, max_tokens=1, stream=True), True)
    assert status == 400 and "12-token context window" in json.loads(payload)["error"]["message"]
    assert not engine.calls


def test_explicit_zero_disables_native_guard_but_keeps_capacity(model_dir):
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 12}))
    app = server.App(Engine(), model_dir, "test", context_window=0)
    assert app.check(request(13, False, max_tokens=4)) is None
    bounded = server.App(FlashEngine(), model_dir, "test", context_window=0)
    assert "safe cache capacity" in bounded.check(request(13, False, max_tokens=1))


def test_explicit_native_window_and_truthful_safe_capacity(model_dir):
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
    app = server.App(Engine(), model_dir, "test", context_window=12)
    assert "12-token context window" in app.check(request(12, False))
    bounded = server.App(FlashEngine(), model_dir, "test", context_window=262144)
    problem = bounded.check(request(13, False))
    assert bounded.effective_context_window == 13
    assert "13-token safe cache capacity" in problem and "model window: 262144 tokens" in problem


def test_zero_engine_capacity_is_finite_and_refused(model_dir):
    engine = FlashEngine()
    engine.max_len = engine.depth + 1
    app = server.App(engine, model_dir, "test", context_window=0)
    assert app.effective_context_window == 0
    with http_server(app) as port:
        status, payload = post(port, request(1, False, stream=True), False)
    assert status == 400 and "0-token safe cache capacity" in json.loads(payload)["error"]["message"]
    assert not engine.calls
    assert server.App(Engine(), model_dir, "test", context_window=0).effective_context_window is None


@pytest.mark.parametrize("chat", [False, True])
def test_native_boundary_refuses_explicit_completion_alias(model_dir, chat):
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 12}))
    engine = Engine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(11, chat, max_completion_tokens=9), chat)
    assert status == 400
    error = json.loads(payload)["error"]
    assert error["type"] == "invalid_request_error"
    assert "9 reply tokens" in error["message"] and "12-token context window" in error["message"]
    assert not engine.calls


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_last_safe_prompt_reply_is_served(model_dir, chat, stream):
    engine = FlashEngine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(12, chat, max_tokens=1, stream=stream), chat)
    assert status == 200
    assert len(engine.calls[0][0]) == 12 and engine.calls[0][1] == 1
    assert "answer" in payload
    if stream:
        assert "data: [DONE]" in payload


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_reply_overflow_refuses_before_generation_and_recovers(model_dir, chat, stream):
    engine = FlashEngine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(12, chat, max_tokens=9, stream=stream), chat)
        assert status == 400
        error = json.loads(payload)["error"]
        assert error["type"] == "invalid_request_error"
        assert "9 reply tokens" in error["message"] and "13-token safe cache capacity" in error["message"]
        assert "reduce" in error["message"]
        assert not engine.calls
        status, _ = post(port, request(12, chat, max_tokens=1, stream=stream), chat)
    assert status == 200 and engine.calls[0][1] == 1


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("options", [{}, {"max_tokens": 0}, {"max_completion_tokens": None}])
def test_omitted_reply_reservation_keeps_default_capping(model_dir, chat, options):
    engine = FlashEngine()
    app = server.App(engine, model_dir, "test", max_tokens=9)
    app.run(request(12, chat, **options), chat, lambda delta: True)
    assert engine.calls[0][1] == 1


def test_template_tools_and_thinking_are_prepared_once(model_dir):
    engine = Engine()
    app = server.App(engine, model_dir, "test", context_window=12)
    counts = {"render": 0, "encode": 0}
    original_template, original_tok = app.template, app.tok

    def render(*args, **kwargs):
        counts["render"] += 1
        return original_template.render(*args, **kwargs)

    def encode(*args, **kwargs):
        counts["encode"] += 1
        return original_tok.encode(*args, **kwargs)

    app.template = SimpleNamespace(render=render)
    app.tok = SimpleNamespace(encode=encode, decode=original_tok.decode)
    body = request(8, True, max_tokens=1, stream=True,
                   tools=[{"type": "function", "function": {"name": "test"}}],
                   chat_template_kwargs={"enable_thinking": True, "suffix": "extra"})
    with http_server(app) as port:
        status, payload = post(port, body, True)
    assert status == 200 and "data: [DONE]" in payload
    assert counts == {"render": 1, "encode": 1}
    assert len(engine.calls[0][0]) == 11 and engine.calls[0][1] == 1


def test_glm_explicit_reply_refusal_and_request_policy(model_dir):
    class GlmEngine(Engine):
        limit = 12
        request = threading.local()
        capacity_plan = {"largest_window": 64}

        def generate(self, *args, **kwargs):
            self.policy = self.request.policy
            self.stop_eos = self.request.stop_eos
            return super().generate(*args, **kwargs)

    engine = GlmEngine()
    app = GlmApp(engine, model_dir, "test")
    assert app.check(request(8, False, max_tokens=4)) is None
    assert "--context 13" in app.check(request(8, False, max_tokens=5))
    with http_server(app) as port:
        status, payload = post(port, request(8, False, max_tokens=5, stream=True), False)
        assert status == 400 and "--context 13" in json.loads(payload)["error"]["message"]
        status, _ = post(port, request(8, False, model="test@model-policy", tf_policy="request-policy",
                                      ignore_eos=True), False)
    assert status == 200 and engine.calls[0][1] == 4
    assert engine.policy == "request-policy" and engine.stop_eos is False


@pytest.mark.parametrize("largest", [None, 12, 64])
def test_refusals_suggest_only_a_context_the_admission_accepts(model_dir, largest):
    class Windowed(Engine):
        context_window = limit = 12
        request = threading.local()
        capacity_plan = {} if largest is None else {"largest_window": largest}

    for app in (server.App(Windowed(), model_dir, "test"), GlmApp(Windowed(), model_dir, "test")):
        problem = app.check(request(8, False, max_tokens=5))
        assert ("--context 13 or more" in problem) == (largest == 64)
        assert "shorten the prompt" in problem or "reduce the prompt" in problem


@pytest.mark.parametrize("stream", [False, True])
def test_late_named_request_error_is_serialized(model_dir, stream):
    class Refusing(Engine):
        def generate(self, *args, **kwargs):
            raise server.RequestError("request cannot fit")

    app = server.App(Refusing(), model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(2, True, stream=stream), True)
    assert status == (200 if stream else 400)
    assert "invalid_request_error" in payload and "request cannot fit" in payload
    if stream:
        assert "data: [DONE]" in payload


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", ["invalid", {}, float("inf")])
def test_malformed_reply_limit_is_named_http_refusal(model_dir, field, value):
    engine = Engine()
    app = server.App(engine, model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, request(2, True, stream=True, **{field: value}), True)
    assert status == 400
    error = json.loads(payload)["error"]
    assert error["type"] == "invalid_request_error" and field in error["message"]
    assert not engine.calls


@pytest.mark.parametrize("body", [None, []])
def test_non_object_json_is_named_http_refusal(model_dir, body):
    app = server.App(Engine(), model_dir, "test")
    with http_server(app) as port:
        status, payload = post(port, body, False)
    assert status == 400
    error = json.loads(payload)["error"]
    assert error["type"] == "invalid_request_error" and "JSON object" in error["message"]


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [None, 0])
def test_zero_or_null_reply_limit_keeps_default(model_dir, field, value):
    engine = Engine()
    app = server.App(engine, model_dir, "test", context_window=12, max_tokens=3)
    app.run(request(8, False, **{field: value}), False, lambda delta: True)
    assert engine.calls[0][1] == 3


def test_template_value_error_is_not_request_error(model_dir):
    app = server.App(Engine(), model_dir, "test")

    def broken_template(*args, **kwargs):
        raise ValueError("template failed")

    app.template = SimpleNamespace(render=broken_template)
    with pytest.raises(ValueError, match="template failed") as error:
        app.prepare(request(2, True), True)
    assert not isinstance(error.value, server.RequestError)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_unrelated_engine_error_is_not_request_error(model_dir, error_type):
    class Broken(Engine):
        def generate(self, *args, **kwargs):
            raise error_type("engine math failed")

    app = server.App(Broken(), model_dir, "test")
    with pytest.raises(error_type, match="engine math failed") as error:
        app.run(request(2, False), False, lambda delta: True)
    assert not isinstance(error.value, server.RequestError)
