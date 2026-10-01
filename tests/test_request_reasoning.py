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
        ("max", True, "xhigh"),
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
    ("{# 'xhigh' 'medium' 'low' #}", "low", "low"),
    ("{# 'xhigh' 'medium' 'low' #}", "medium", "medium"),
    ("{# 'xhigh' 'medium' 'low' #}", "xhigh", "xhigh"),
    ("{# 'xhigh' 'medium' 'low' #}", "max", "xhigh"),        # no max: the nearest named level is xhigh
    ("{# 'low' 'high' #}", "high", "high"),                  # GLM-5.3 names high: it renders High, not Max
    ("{# 'low' 'high' #}", "minimal", "low"),
    ("{# 'low' 'high' #}", "low", "low"),
    ("{# 'low' 'high' #}", "medium", "high"),                # medium is not a GLM name; Max was the silent result
    ("{# 'low' 'high' #}", "xhigh", "xhigh"),               # GLM renders xhigh as its own Max; leave the name
    ("{# 'low' 'high' #}", "none", "none"),
    ("{# 'low' 'high' 'max' #}", "max", "max"),              # the template names max, so the name stays
    ("{# 'low' 'high' 'max' #}", "medium", "high"),
    ("", "high", "xhigh"),
    ("", "max", "xhigh"),
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


def test_glm_omitted_effort_stays_the_template_default_and_a_medium_default_is_high():
    jinja2 = pytest.importorskip("jinja2")
    from tensorfold.server.request_options import coerce_effort, effort_levels

    source = ("{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined "
              "and reasoning_effort in ['low', 'high'] else 'max' -%}{{ effective_reasoning_effort }}")
    levels = effort_levels(source)
    assert levels == frozenset({"low", "high", "max"})
    template = jinja2.Environment().from_string(source)
    assert template.render() == "max"
    assert template.render(reasoning_effort="medium") == "max"
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        heard = coerce_effort(effort, levels)
        assert template.render(reasoning_effort=heard) == {"minimal": "low", "low": "low", "medium": "high",
                                                           "high": "high", "xhigh": "max", "max": "max"}[effort]

    app = make_app(enable_thinking=True, reasoning_effort="medium")
    app.tokenizer.chat_template = "{# 'low' 'high' #}"
    server = serve_fake(app)
    try:
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
        assert status == 200 and json.loads(body)["tensorfold"]["reasoning_effort"] == "high"
        assert all(c.get("reasoning_effort") == "high" for c in app.tokenizer.template_calls if c.get("enable_thinking"))
    finally:
        server.shutdown()
        server.server_close()
        app.close()

    plain = make_app(enable_thinking=True)
    plain.tokenizer.chat_template = "{# 'low' 'high' #}"
    try:
        plain.tokenizer.template_calls.clear()
        plain.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert plain.tokenizer.template_calls and all(
            c.get("enable_thinking") and "reasoning_effort" not in c for c in plain.tokenizer.template_calls)
    finally:
        plain.close()


def test_no_effort_leaves_the_template_its_own_default():
    app = make_app(enable_thinking=True)                  # no --reasoning-effort: as vLLM and mlx-lm render it
    try:
        app.tokenizer.template_calls.clear()
        app.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert app.tokenizer.template_calls and all(
            c.get("enable_thinking") and "reasoning_effort" not in c for c in app.tokenizer.template_calls)
    finally:
        app.close()


@pytest.mark.parametrize("names, effort, want", [
    ("{# 'low' 'high' #}", "medium", "high"),      # GLM-5.3 names no medium: the higher of the two levels as near
    ("{# 'low' 'high' #}", "xhigh", "xhigh"),      # the else-branch is the template's own ceiling: as sent
    ("{# 'low' 'high' #}", "low", "low"),
    ("{# 'low' 'high' #}", "high", "high"),
    ("{# 'high' #}", "medium", "high"),                 # a single named level: straight to it
    ("{# 'low' #}", "medium", "low"),
    ("{# 'medium' #}", "medium", "medium"),             # a named medium is honoured
    ("{# 'xhigh' 'medium' 'low' #}", "medium", "medium"),   # Qwen3.8 names medium: untouched
    ("", "medium", "medium"),                     # a template naming no level keeps the effort as sent
])
def test_a_medium_the_template_does_not_name_lands_on_the_nearest_named_level(names, effort, want):
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


def test_medium_reaches_the_template_through_chat_template_kwargs_too():
    app = make_app(enable_thinking=True)
    app.tokenizer.chat_template = "{# 'low' 'high' #}"
    server = serve_fake(app)
    try:
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
            "chat_template_kwargs": {"reasoning_effort": "medium"}})
        assert status == 200 and json.loads(body)["tensorfold"]["reasoning_effort"] == "high"
        assert all(c["reasoning_effort"] == "high" for c in app.tokenizer.template_calls if c.get("enable_thinking"))
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_the_server_default_effort_uses_the_nearest_named_level():
    """``--reasoning-effort`` goes through the same nearest-level rule as a request."""

    glm = make_app(enable_thinking=True, reasoning_effort="medium")
    glm.tokenizer.chat_template = "{# 'low' 'high' #}"
    qwen = make_app(enable_thinking=True, reasoning_effort="high")
    qwen.tokenizer.chat_template = "{# 'xhigh' 'medium' 'low' #}"
    try:
        glm.tokenizer.template_calls.clear()
        glm.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert all(c["reasoning_effort"] == "high" for c in glm.tokenizer.template_calls)
        qwen.tokenizer.template_calls.clear()
        qwen.chat([{"role": "user", "content": "hi"}], max_tokens=2)
        assert all(c["reasoning_effort"] == "xhigh" for c in qwen.tokenizer.template_calls)
    finally:
        glm.close()
        qwen.close()


def test_glm_and_qwen_templates_hear_the_nearest_named_level():
    """GLM-5.3's effort line and Qwen3.8's effort line, the text in those checkpoints."""

    jinja2 = pytest.importorskip("jinja2")
    from tensorfold.server.request_options import coerce_effort, effort_levels, heard_effort

    glm = ("{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined "
           "and reasoning_effort in ['low', 'high'] else 'max' -%}{{ effective_reasoning_effort }}")
    qwen = ("{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
            "{{- raise_exception('Unexpected reasoning effort') }}{%- endif -%}{{ resolved_reasoning_effort }}")
    glm_levels, qwen_levels = effort_levels(glm), effort_levels(qwen)
    assert glm_levels == frozenset({"low", "high", "max"})
    assert qwen_levels == frozenset({"low", "medium", "xhigh"})
    glm_template = jinja2.Environment().from_string(glm)
    qwen_template = jinja2.Environment().from_string(qwen)
    assert glm_template.render(reasoning_effort=coerce_effort("medium", glm_levels)) == "high"
    assert glm_template.render(reasoning_effort=coerce_effort("minimal", glm_levels)) == "low"
    assert glm_template.render(reasoning_effort=coerce_effort("xhigh", glm_levels)) == "max"
    assert glm_template.render(reasoning_effort=coerce_effort("max", glm_levels)) == "max"
    assert qwen_template.render(reasoning_effort=coerce_effort("high", qwen_levels)) == "xhigh"
    assert qwen_template.render(reasoning_effort=coerce_effort("max", qwen_levels)) == "xhigh"
    assert qwen_template.render(reasoning_effort=coerce_effort("minimal", qwen_levels)) == "low"
    assert qwen_template.render(reasoning_effort=coerce_effort("medium", qwen_levels)) == "medium"
    assert heard_effort(None, "medium", glm_levels) == "high"
    assert heard_effort(None, "high", qwen_levels) == "xhigh"
    assert heard_effort(None, "medium", qwen_levels) == "medium"


def test_nearest_named_effort_direct():
    from tensorfold.server.request_options import nearest_named_effort

    nearest = nearest_named_effort
    assert nearest("medium", frozenset({"xhigh", "low"})) == "low"      # non-tie, downward
    assert nearest("medium", frozenset({"minimal"})) == "minimal"       # the only named level
    assert nearest("medium", frozenset({"xhigh", "high"})) == "high"    # distance 2 vs 1
    assert nearest("medium", frozenset()) == "medium"                   # no named levels: as sent
    assert nearest("none", frozenset({"low", "high"})) == "none"        # none is not a level: as sent
    assert nearest("medium", frozenset({"low", "high"})) == "high"      # the GLM-5.3 tie: upward


def test_glm_medium_with_thinking_forced_off_reaches_no_effort():
    app = make_app(enable_thinking=True)
    app.tokenizer.chat_template = "{# \x27low\x27 \x27high\x27 #}"
    server = serve_fake(app)
    try:
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
            "reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": False}})
        assert status == 200 and json.loads(body)["tensorfold"]["enable_thinking"] is False
        assert all(not c["enable_thinking"] and "reasoning_effort" not in c
                   for c in app.tokenizer.template_calls)
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_a_top_level_effort_wins_over_chat_template_kwargs():
    app = make_app(enable_thinking=True)
    app.tokenizer.chat_template = ""                    # names no level: nothing maps
    server = serve_fake(app)
    try:
        app.tokenizer.template_calls.clear()
        status, body = post_json(server, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
            "reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "medium"}})
        assert status == 200 and json.loads(body)["tensorfold"]["reasoning_effort"] == "low"
        assert all(c["reasoning_effort"] == "low" for c in app.tokenizer.template_calls
                   if c.get("enable_thinking"))
    finally:
        server.shutdown()
        server.server_close()
        app.close()
