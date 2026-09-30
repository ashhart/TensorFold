import copy
import json

import pytest

import tensorfold.server.http as http

from tests.test_lane_server import make_app
from tests.test_server_openai_compat import FakeApp, post_json, serve_fake


class SamplingApp(FakeApp):
    accepts_sampling = True

    def chat(self, *args, sampling=None, **kwargs):
        self.sampling = sampling
        return super().chat(*args, **kwargs)


@pytest.mark.parametrize(
    "effort,thinking,normalized",
    [
        ("none", False, "none"),
        ("low", True, "low"),
        ("medium", True, "medium"),
        ("high", True, "xhigh"),
        ("xhigh", True, "xhigh"),
    ],
)
def test_http_reasoning_controls(effort, thinking, normalized):
    app = SamplingApp()
    server = serve_fake(app)
    try:
        status, _ = post_json(
            server,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort},
        )
        assert status == 200
        assert app.sampling == {"enable_thinking": thinking, "reasoning_effort": normalized}
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_effort_and_explicit_toggle_precedence():
    app = SamplingApp()
    server = serve_fake(app)
    try:
        payload = {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "bogus"}
        assert post_json(server, "/v1/chat/completions", payload)[0] == 400
        payload.update(reasoning_effort="high", chat_template_kwargs={"enable_thinking": False})
        assert post_json(server, "/v1/chat/completions", payload)[0] == 200
        assert app.sampling["enable_thinking"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_request_effort_reaches_template_without_changing_server_default():
    app = make_app(enable_thinking=False, reasoning_effort="medium")
    try:
        for effort in ["low", "xhigh"]:
            app.tokenizer.template_calls.clear()
            reply = app.chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=2,
                sampling={"enable_thinking": True, "reasoning_effort": effort},
            )
            assert all(c["enable_thinking"] and c["reasoning_effort"] == effort for c in app.tokenizer.template_calls)
            assert reply["runtime"]["reasoning_effort"] == effort
        app.tokenizer.template_calls.clear()
        reply = app.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert all(not c["enable_thinking"] for c in app.tokenizer.template_calls)
        assert reply["runtime"]["enable_thinking"] is False
        assert app.reasoning_effort == "medium"
    finally:
        app.close()


@pytest.mark.parametrize("default_effort", ["low", "medium", "xhigh"])
def test_explicit_thinking_overrides_none_and_preserves_default(default_effort, monkeypatch):
    app = make_app(enable_thinking=True, reasoning_effort=default_effort)
    app.tokenizer.template_calls.clear()       # the startup probe of the template's roles
    template = app.tokenizer.apply_chat_template
    parsed_requests = []
    loads = json.loads

    def record_body(value, *args, **kwargs):
        parsed = loads(value, *args, **kwargs)
        if isinstance(value, bytes) and isinstance(parsed, dict):
            parsed_requests.append((parsed, copy.deepcopy(parsed)))
        return parsed

    def checked_template(messages, **kwargs):
        if kwargs.get("enable_thinking") and kwargs.get("reasoning_effort") == "none":
            raise ValueError("enabled thinking requires low, medium or xhigh effort")
        return template(messages, **kwargs)

    monkeypatch.setattr(http.json, "loads", record_body)
    monkeypatch.setattr(app.tokenizer, "apply_chat_template", checked_template)
    server = serve_fake(app)
    try:
        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                   "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}}
        status, body = post_json(server, "/v1/chat/completions", payload)
        assert status == 200
        assert loads(body)["tensorfold"]["reasoning_effort"] == default_effort
        assert all(c["enable_thinking"] and c["reasoning_effort"] == default_effort
                   for c in app.tokenizer.template_calls)
        for fields in [{"reasoning_effort": "none"},
                       {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}]:
            app.tokenizer.template_calls.clear()
            status, body = post_json(server, "/v1/chat/completions", {
                "messages": payload["messages"], "max_tokens": 2, **fields})
            assert status == 200
            assert loads(body)["tensorfold"]["enable_thinking"] is False
            assert all(not c["enable_thinking"] for c in app.tokenizer.template_calls)
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions", {
            "messages": payload["messages"], "max_tokens": 2})
        assert status == 200
        assert loads(body)["tensorfold"]["reasoning_effort"] == default_effort
        assert all(c["enable_thinking"] and c["reasoning_effort"] == default_effort
                   for c in app.tokenizer.template_calls)
        assert app.enable_thinking is True and app.reasoning_effort == default_effort
        assert len(parsed_requests) == 4
        assert all(body == original for body, original in parsed_requests)
    finally:
        server.shutdown()
        server.server_close()
        app.close()


@pytest.mark.parametrize("names, effort, want", [
    ("{# 'xhigh' 'medium' 'low' #}", "high", "xhigh"),       # Qwen3.8 names no high: OpenAI's high is its xhigh
    ("{# 'xhigh' 'medium' 'low' #}", "minimal", "low"),
    ("{# 'low' 'high' #}", "high", "high"),                  # GLM-5.3 names high: it renders High, not Max
    ("{# 'low' 'high' #}", "minimal", "low"),
    ("", "high", "xhigh"),
])
def test_a_template_that_names_an_effort_is_given_that_effort(names, effort, want):
    app = make_app(enable_thinking=True)
    app.tokenizer.chat_template = names
    server = serve_fake(app)
    try:
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "reasoning_effort": effort})
        assert status == 200 and json.loads(body)["tensorfold"]["reasoning_effort"] == want
        assert all(c["reasoning_effort"] == want for c in app.tokenizer.template_calls if c.get("enable_thinking"))
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_no_effort_leaves_the_template_its_own_default():
    app = make_app(enable_thinking=True)                  # no --reasoning-effort: as vLLM and mlx-lm render it
    try:
        app.tokenizer.template_calls.clear()
        app.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert app.tokenizer.template_calls and all(
            c.get("enable_thinking") and "reasoning_effort" not in c for c in app.tokenizer.template_calls)
    finally:
        app.close()
