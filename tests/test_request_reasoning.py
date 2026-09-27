import pytest

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
