"""A text completion reads its prompt raw, as vLLM and mlx_lm do: no chat template and no think block."""

from tests.test_lane_server import make_app
from tests.test_server_openai_compat import FakeApp, post_json, serve_fake


class RawApp(FakeApp):
    accepts_raw_prompt = True

    def chat(self, messages, *, prompt=None, **kwargs):
        self.prompt = prompt
        return super().chat(messages, **kwargs)


def complete(app, body):
    server = serve_fake(app)
    try:
        return post_json(server, "/v1/completions", {"model": "fake-model", "max_tokens": 8, **body})
    finally:
        server.shutdown()
        server.server_close()


def test_text_and_token_prompts_reach_the_app_raw():
    app = RawApp()
    assert complete(app, {"prompt": "def add(a, b):"})[0] == 200
    assert app.prompt == "def add(a, b):" and app.messages == []
    assert complete(app, {"prompt": [11, 12, 13]})[0] == 200
    assert app.prompt == [11, 12, 13]


def test_a_completion_that_sends_messages_stays_a_chat():
    app = RawApp()
    assert complete(app, {"messages": [{"role": "user", "content": "Hi"}]})[0] == 200
    assert app.prompt is None and app.messages == [{"role": "user", "content": "Hi"}]


def test_the_engine_gets_exactly_the_encoded_prompt():
    app = make_app()
    try:
        calls = len(app.tokenizer.template_calls)
        reply = app.chat([], prompt="abc", max_tokens=2)
        assert reply["prompt_tokens"] == len(app.tokenizer.encode("abc"))
        assert len(app.tokenizer.template_calls) == calls
        assert reply["runtime"]["enable_thinking"] is False
    finally:
        app.close()
