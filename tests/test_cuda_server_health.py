"""GET /health publishes cumulative token totals while a request runs, so a poller can read."""

import http.client
import json
import threading

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for, post, serving

WAIT = 10


def health(port) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


def test_health_counts_tokens_as_they_are_emitted_and_idles_after(tmp_path):
    engine = PacedEngine(hold_at=0)  # pauses before the first token
    app = app_for(tmp_path, engine)
    app.context_window = 262144
    with serving(app) as port:
        before = health(port)
        assert before["ok"] is True and before["backend"] == "tensorfold"
        assert before["busy"] is False and before["requests_running"] == 0
        assert before["context_length"] == 262144
        reply = {}

        def run():
            reply["status"], reply["body"] = post(port, {"messages": MESSAGES, "max_tokens": 4})

        worker = threading.Thread(target=run)
        worker.start()
        assert engine.held.wait(WAIT)
        during = health(port)
        assert during["busy"] is True and during["requests_running"] == before["requests_running"] + 1
        # Prefill has not finished, so the counters have not moved yet.
        assert during["completion_tokens_total"] == before["completion_tokens_total"]
        assert during["prompt_tokens_total"] == before["prompt_tokens_total"]
        engine.release.set()
        worker.join(WAIT)
        assert reply["status"] == 200, reply.get("body", "")[:300]
        after = health(port)
    assert after["busy"] is False and after["requests_running"] == before["requests_running"]
    assert after["completion_tokens_total"] == before["completion_tokens_total"] + 4
    assert after["prompt_tokens_total"] > before["prompt_tokens_total"]
    assert after["prefill_seconds_total"] > before["prefill_seconds_total"]
