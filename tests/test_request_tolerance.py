"""Fields that OpenAI clients send as defaults are accepted."""

from tensorfold.server.tool_policy import ToolCallPolicy
from tests.test_request_reasoning import SamplingApp
from tests.test_server_openai_compat import post_json, serve_fake


def ask(app, **fields):
    server = serve_fake(app)
    try:
        return post_json(server, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], **fields})
    finally:
        server.shutdown()
        server.server_close()


def test_minimal_effort_is_the_templates_lowest():
    app = SamplingApp()
    assert ask(app, reasoning_effort="minimal")[0] == 200
    assert app.sampling == {"enable_thinking": True, "reasoning_effort": "low"}


def test_null_effort_and_null_parallel_tool_calls_mean_the_defaults():
    app = SamplingApp()
    assert ask(app, reasoning_effort=None, parallel_tool_calls=None)[0] == 200
    assert "reasoning_effort" not in app.sampling
    assert not ToolCallPolicy({"parallel_tool_calls": None}).single
