import pytest

from tensorfold.server.http import RequestError
from tests.test_lane_server import FakeTokenizer, make_app
from tests.test_server_openai_compat import post_json, serve_fake


class ThinkingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        tokens = super().apply_chat_template(messages, **kwargs)
        return tokens + ([5, 6] if kwargs.get("enable_thinking") else [])


def assert_unsubmitted(app, monkeypatch):
    monkeypatch.setattr(app, "system_prefix_len", lambda *a, **k: pytest.fail("prefix inspected"))
    monkeypatch.setattr(app.scheduler, "submit", lambda job: pytest.fail("job submitted"))


def test_requested_reply_reserve_one_over_refuses_before_prefix_or_submission(monkeypatch):
    app = make_app(context_window=10)
    try:
        messages = [{"role": "user", "content": "abcdef"}]
        assert len(app.render(messages)[0]) == 9
        assert_unsubmitted(app, monkeypatch)
        with pytest.raises(RequestError, match=r"9.*2.*10") as error:
            app.chat(messages, max_tokens=2)
        assert "8 prompt tokens" in str(error.value)
        assert "1 reply tokens" in str(error.value)
        assert not app.engine.prefill_calls and app._preparing == 0
    finally:
        app.close()


def test_exact_prompt_and_reply_reservation_keeps_requested_limit():
    app = make_app(context_window=10)
    try:
        result = app.chat([{"role": "user", "content": "abcde"}], max_tokens=2)
        assert result["prompt_tokens"] == 8
        assert result["completion_tokens"] == 2
    finally:
        app.close()


@pytest.mark.parametrize("thinking", [False, True])
def test_template_and_thinking_tokens_count_toward_reservation(monkeypatch, thinking):
    app = make_app(context_window=10)
    app.tokenizer = ThinkingTokenizer()
    try:
        messages = [{"role": "user", "content": "abc"}]
        prompt_tokens = len(app.render(messages, thinking=thinking)[0])
        assert prompt_tokens == (8 if thinking else 6)
        assert_unsubmitted(app, monkeypatch)
        with pytest.raises(RequestError, match=rf"{prompt_tokens}.*{11 - prompt_tokens}.*10"):
            app.chat(messages, max_tokens=11 - prompt_tokens,
                     sampling={"enable_thinking": thinking})
        assert not app.engine.prefill_calls and app._preparing == 0
    finally:
        app.close()


def test_omitted_max_tokens_uses_default_capped_by_remaining_context():
    app = make_app(context_window=10, default_max_tokens=12)
    try:
        result = app.chat([{"role": "user", "content": "abcdef"}])
        assert result["prompt_tokens"] == 9
        assert result["completion_tokens"] == 1
    finally:
        app.close()


def test_requested_default_is_explicit_and_refusal_allows_next_request():
    app = make_app(context_window=10, default_max_tokens=12)
    try:
        with pytest.raises(RequestError, match="12 reply tokens"):
            app.chat([{"role": "user", "content": "abcdef"}], max_tokens=12)
        assert not app.engine.prefill_calls and app._preparing == 0
        result = app.chat([{"role": "user", "content": "abcde"}], max_tokens=2)
        assert result["prompt_tokens"] == 8 and result["completion_tokens"] == 2
        assert len(app.engine.prefill_calls) == 1
    finally:
        app.close()


def test_unlimited_context_preserves_explicit_reply_limit():
    app = make_app(context_window=0)
    try:
        result = app.chat([{"role": "user", "content": "abcdef"}], max_tokens=2)
        assert result["prompt_tokens"] == 9 and result["completion_tokens"] == 2
    finally:
        app.close()


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_http_requested_reserve_refuses_with_counts_and_next_request_recovers(field):
    app = make_app(context_window=10)
    server = serve_fake(app)
    try:
        payload = {"messages": [{"role": "user", "content": "abcdef"}], field: 2}
        status, body = post_json(server, "/v1/chat/completions", payload)
        assert status == 400 and "9 tokens" in body and "2 reply tokens" in body
        assert not app.engine.prefill_calls
        payload["messages"][0]["content"] = "abcde"
        status, body = post_json(server, "/v1/chat/completions", payload)
        assert status == 200 and '"completion_tokens": 2' in body
        assert len(app.engine.prefill_calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        app.close()
