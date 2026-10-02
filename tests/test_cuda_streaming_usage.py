"""A streamed chat reply ends with its finish chunk, then OpenAI's usage-only ``choices: []`` event (#216).

Spec-following gateways (litellm and the like) count a stream's usage from the trailing empty-choices
event; usage riding the finish chunk is dropped there, losing ``cached_tokens``.
"""

import json

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for, serving
from tensorfold.cuda import server

WAIT = 10


def post(port, body, path="/v1/chat/completions"):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def events(body: str) -> list[dict]:
    """The reply's SSE payloads, in order, with the ``[DONE]`` sentinel last."""
    chunks = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]", "the stream does not end with [DONE]"
    return [json.loads(chunk) for chunk in chunks[:-1]]


def test_streamed_chat_usage_trails_in_its_own_choices_empty_event(tmp_path):
    engine = PacedEngine("Hi there", finish_all=True)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        status, body = post(port, {"messages": MESSAGES, "max_tokens": 4, "stream": True})
    assert status == 200
    reply = events(body)

    finish = reply[-2]
    assert finish["choices"][0]["finish_reason"] in ("stop", "length")
    assert finish["choices"][0]["delta"] == {}
    assert "usage" not in finish, "the finish chunk rides the usage spec-following clients drop"
    assert finish["tensorfold"]                                  # the engine's stats stay on the finish chunk

    usage = reply[-1]
    assert usage["choices"] == []
    assert usage["object"] == "chat.completion.chunk" and usage["id"] == finish["id"] == reply[0]["id"]
    assert usage["usage"]["prompt_tokens"] > 0 and usage["usage"]["completion_tokens"] > 0
    assert usage["usage"]["total_tokens"] == (usage["usage"]["prompt_tokens"]
                                              + usage["usage"]["completion_tokens"])
    assert "cached_tokens" in usage["usage"]["prompt_tokens_details"]


def test_streamed_completion_keeps_its_usage_on_the_end_chunk(tmp_path):
    engine = PacedEngine("Hi there", finish_all=True)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        status, body = post(port, {"prompt": "Hi", "max_tokens": 4, "stream": True},
                            path="/v1/completions")
    assert status == 200
    reply = events(body)
    end = reply[-1]
    assert end["choices"][0]["finish_reason"] == "length"
    assert end["usage"]["prompt_tokens"] > 0                     # unchanged by #216's chat-side fix
    assert all("usage" not in chunk for chunk in reply[:-1])


def test_unstreamed_chat_keeps_its_usage_in_the_reply_body(tmp_path):
    engine = PacedEngine("Hi there", finish_all=True)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        status, body = post(port, {"messages": MESSAGES, "max_tokens": 4})
    assert status == 200
    reply = json.loads(body)
    assert reply["usage"]["prompt_tokens"] > 0
    assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
