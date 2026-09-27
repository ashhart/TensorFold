"""Request EOS policy and user stops are identical for serial and drafted commits."""

import json
from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_engine import LaneEngine
from tests.http_fakes import post
from tests.lane_fakes import FakeEngine, PatternProposer, fake_serial
from tests.test_lane_server import make_app
from tests.test_server_openai_compat import FakeApp


REPLY = [11, 49, 12, 50, 13, 14, 15, 49]
ROUTES = ["/v1/chat/completions", "/v1/completions"]


class CommitEngine(LaneEngine):
    def __init__(self, model=None, **kwargs):
        super().__init__(SimpleNamespace(lane_family=True, exact_width=4), **kwargs)

    def _family_prefill(self, stream, **kwargs):
        stream.cache_len = len(stream.prompt_ids)
        stream.commit(REPLY[:1])
        return []

    def _family_round(self, stream, cache):
        start = len(stream.emitted)
        width = 3 if stream.drafts else 1
        landed = stream.commit(REPLY[start:start + width])
        return landed, width, len(landed)


def response(app, route, stream, **fields):
    app.accepts_cancellation = False
    prompt = {"prompt": "x"} if route == "/v1/completions" else {"messages": [{"role": "user", "content": "x"}]}
    status, body = post(app, {**prompt, "stream": stream, "max_tokens": 8, **fields}, route)
    assert status == 200, body
    if stream:
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        assert all("error" not in event for event in events), body
        content = "".join(event["choices"][0].get("text", event["choices"][0].get("delta", {}).get("content", ""))
                          for event in events)
        assert body.count("data: [DONE]") == 1
        return content, events[-1]["choices"][0]["finish_reason"], events[-1]["usage"]["completion_tokens"]
    payload = json.loads(body)
    choice = payload["choices"][0]
    return choice.get("text", choice.get("message", {}).get("content")), choice["finish_reason"], payload["usage"]["completion_tokens"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ignore", [None, False, True])
@pytest.mark.parametrize("first_is_eos", [False, True])
def test_ignore_eos_matches_drafted_and_serial_requests(route, stream, ignore, first_is_eos):
    app = make_app(engine_factory=CommitEngine, use_proposer=True)
    eos_ids = frozenset({11, 49, 50} if first_is_eos else {49, 50})
    app.stop_ids = app.scheduler.eos_ids = eos_ids
    try:
        fields = {} if ignore is None else {"ignore_eos": ignore}
        replies = [response(app, route, stream, draft=draft, **fields) for draft in (False, True)]
        expected = (app.tokenizer.decode(REPLY), "length", 8) if ignore else (app.tokenizer.decode([11]), "stop", 2)
        if not ignore and first_is_eos:
            expected = ("", "stop", 1)
        assert replies == [expected, expected]
        assert app.stop_ids == app.scheduler.eos_ids == eos_ids
    finally:
        app.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_unmatched_stop_prefix_is_flushed_at_the_length_limit(route, stream):
    app = make_app(engine_factory=CommitEngine, use_proposer=True)
    try:
        stop = app.tokenizer.decode(REPLY[-2:]) + "!"
        replies = [response(app, route, stream, draft=draft, ignore_eos=True, stop=stop) for draft in (False, True)]
        assert replies == [(app.tokenizer.decode(REPLY), "length", 8)] * 2
    finally:
        app.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
def test_ignore_eos_preserves_user_stop_strings_across_rounds(route, stream, as_list):
    app = make_app(engine_factory=CommitEngine, use_proposer=True)
    try:
        stop = app.tokenizer.decode(REPLY[2:5])
        fields = {"ignore_eos": True, "stop": ["absent", stop] if as_list else stop}
        replies = [response(app, route, stream, draft=draft, **fields) for draft in (False, True)]
        expected = (app.tokenizer.decode(REPLY[:2]), "stop", 5)
        assert replies == [expected, expected]
    finally:
        app.close()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_ignore_eos_rejects_non_booleans_before_stream_headers(route, stream, value):
    status, body = post(FakeApp(), {"messages": [{"role": "user", "content": "x"}], "prompt": "x",
                                  "stream": stream, "ignore_eos": value}, route)
    assert status == 400 and "ignore_eos" in json.loads(body)["error"]["message"]
    assert not body.startswith("data:")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("user_stop", [False, True])
def test_native_fake_engine_ignore_eos_matches_serial(monkeypatch, route, stream, user_stop):
    from tensorfold.server import app as app_module

    monkeypatch.setattr(app_module, "SuffixLookupProposer", lambda **kwargs: PatternProposer([4]))
    app = make_app(engine_factory=FakeEngine, use_proposer=True)
    try:
        prompt = app.render([{"role": "user", "content": "x"}])[0]
        if route == "/v1/completions":
            prompt = app.tokenizer.encode("x")
        expected = fake_serial(prompt, 8, set())
        app.stop_ids = app.scheduler.eos_ids = frozenset({expected[1], expected[3], expected[-1]})
        fields = {"stop": app.tokenizer.decode(expected[3:6])} if user_stop else {}
        replies = [response(app, route, stream, draft=draft, ignore_eos=True, **fields) for draft in (False, True)]
        target = (app.tokenizer.decode(expected[:3]), "stop", 6) if user_stop else (app.tokenizer.decode(expected), "length", 8)
        assert replies == [target] * 2
        assert any(stat.width > 1 for stat in app.engine.round_stats)
    finally:
        app.close()
