"""GET /health publishes the lane engine's own totals and the live replies' tokens, so a poller can read tok/s."""

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import health
from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for, post, serving

WAIT = 10


class StatsEngine(PacedEngine):
    """The paced engine, returning a lane engine's stats for its request."""

    def generate(self, *args, **kwargs):
        rounds = super().generate(*args, **kwargs)["rounds"]
        return {"prefill_s": 0.25, "decode_s": 0.5, "rounds": rounds, "drafted": 3 * rounds, "accepted": rounds,
                "cached": 2}


def health_of(port) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


def test_health_counts_live_tokens_and_folds_the_engine_s_stats_when_a_request_ends(tmp_path):
    engine = StatsEngine(hold_at=2)                  # two tokens out, then held
    app = app_for(tmp_path, engine)
    app.context_window = 262144
    with serving(app) as port:
        before = health_of(port)
        assert before["ok"] is True and before["backend"] == "tensorfold" and "streams" not in before
        assert before["busy"] is False and before["requests_running"] == 0 and before["requests_total"] == 0
        assert before["context_length"] == 262144
        reply = {}

        def run():
            reply["status"], reply["body"] = post(port, {"messages": MESSAGES, "max_tokens": 4})

        worker = threading.Thread(target=run)
        worker.start()
        assert engine.held.wait(WAIT)
        during = health_of(port)
        assert during["busy"] is True and during["requests_running"] == 1
        assert during["completion_tokens_total"] == 2                     # the live reply's tokens so far
        assert during["prompt_tokens_total"] == 0 and during["rounds_total"] == 0     # engine stats come at the end
        engine.release.set()
        worker.join(WAIT)
        assert reply["status"] == 200, reply.get("body", "")[:300]
        after = health_of(port)
    assert after["busy"] is False and after["requests_running"] == 0 and after["requests_total"] == 1
    assert after["completion_tokens_total"] == 4 and after["prompt_tokens_total"] > 0
    assert after["prefill_seconds_total"] == 0.25 and after["decode_seconds_total"] == 0.5
    assert (after["rounds_total"], after["drafted_total"], after["accepted_total"]) == (4, 12, 4)
    assert after["cached_tokens_total"] == 2


def test_a_failed_request_still_counts_what_it_emitted(tmp_path):
    class Failing(PacedEngine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            on_tokens([ord("a"), ord("b")])
            raise RuntimeError("the engine failed")

    app = app_for(tmp_path, Failing())
    with serving(app) as port:
        status, _ = post(port, {"messages": MESSAGES, "max_tokens": 4})
        after = health_of(port)
    assert status == 500 and after["requests_running"] == 0 and after["requests_total"] == 1
    assert after["completion_tokens_total"] == 2 and after["rounds_total"] == 0


def test_a_concurrent_engine_reports_its_streams(tmp_path):
    engine = PacedEngine()
    decoder = SimpleNamespace(streams={1: None, 2: None}, filling=[None])
    engine.scheduler = SimpleNamespace(decoder=decoder, max_streams=4)
    app = app_for(tmp_path, engine)
    assert health.of(app).snapshot(app)["streams"] == {"decoding": 2, "prefilling": 1, "max": 4}
    assert health.of(app) is health.of(app)


def test_a_bare_app_still_answers(tmp_path):
    app = SimpleNamespace(served="fake")
    body = health.of(app).snapshot(app)
    assert body["ok"] is True and body["busy"] is False and "streams" not in body and "context_length" not in body


def test_a_concurrent_stream_reports_its_drafted_rows_and_kept_drafts():
    from tensorfold.cuda.streams import Stream

    s = Stream([1, 2], 10, None)
    s.take([5])                               # the prefill's first token
    s.counted(4)
    s.take([6, 7])                            # a 4-row window: 3 drafted rows, 1 kept, then the round's own token
    s.counted(1)
    s.take([8])                               # a one-row round
    stats = s.stats()
    assert (stats["rounds"], stats["drafted"], stats["accepted"]) == (2, 3, 1)
