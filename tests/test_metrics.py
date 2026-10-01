"""GET /metrics is Prometheus text: running against waiting, tokens, KV, drafts, latency."""

import http.client
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tensorfold.cuda import health
from tensorfold.server import metrics
from tensorfold.server.http import make_handler

NAMES = ("requests_running", "requests_waiting", "prompt_tokens_total", "generation_tokens_total",
         "kv_cache_usage_ratio", "mtp_drafted_total", "mtp_accepted_total",
         "request_latency_seconds", "time_to_first_token_seconds")


def sample(body: str, name: str) -> str:
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return line.split()[-1]
    raise AssertionError(f"{name} missing")


def bucket(body: str, name: str, le: str) -> str:
    return sample(body, f"{metrics.PREFIX}{name}_bucket{{le=\"{le}\"}}")


def serve(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return httpd, thread


def get(port: int, path: str) -> tuple[int, str, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type") or "", response.read().decode()
    finally:
        connection.close()


def test_mac_metrics_is_prometheus_and_health_stays_json():
    app = SimpleNamespace(served_name="test", model_ids=["test"], max_batch_size=1)
    httpd, thread = serve(app)
    try:
        status, content_type, body = get(httpd.server_port, "/metrics")
        assert status == 200 and content_type.startswith("text/plain")
        for name in NAMES:
            assert f"# TYPE {metrics.PREFIX}{name} " in body
        assert sample(body, f"{metrics.PREFIX}requests_running") == "0"
        assert sample(body, f"{metrics.PREFIX}requests_waiting") == "0"
        assert sample(body, f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="0"}}') == "0"
        assert bucket(body, "request_latency_seconds", "+Inf") == "0"
        other, _, again = get(httpd.server_port, "/v1/metrics")
        assert other == 200 and again == body
        health_status, health_type, health_body = get(httpd.server_port, "/health")
        assert health_status == 200 and "application/json" in health_type and '"status": "ok"' in health_body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def test_histogram_buckets_are_cumulative_and_a_missing_first_token_is_not_counted():
    app = SimpleNamespace()
    metrics.note(app, prompt=4, generation=1, drafted=3, accepted=1, latency=0.2, ttft=0.02)
    metrics.note(app, prompt=5, generation=2, drafted=0, accepted=0, latency=10.0, ttft=None)
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}prompt_tokens_total") == "9"
    assert sample(body, f"{metrics.PREFIX}generation_tokens_total") == "3"
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == "3"
    assert sample(body, f"{metrics.PREFIX}mtp_accepted_total") == "1"
    assert bucket(body, "request_latency_seconds", "0.1") == "0"
    assert bucket(body, "request_latency_seconds", "0.25") == "1"
    assert bucket(body, "request_latency_seconds", "5") == "1"
    assert bucket(body, "request_latency_seconds", "10") == "2"
    assert bucket(body, "request_latency_seconds", "+Inf") == "2"
    assert sample(body, f"{metrics.PREFIX}request_latency_seconds_count") == "2"
    assert bucket(body, "time_to_first_token_seconds", "0.01") == "0"
    assert bucket(body, "time_to_first_token_seconds", "0.05") == "1"
    assert sample(body, f"{metrics.PREFIX}time_to_first_token_seconds_count") == "1"


def test_mac_running_includes_the_prefill_and_kv_uses_the_window():
    app = SimpleNamespace(
        scheduler=SimpleNamespace(active=1, waiting=2, filling=[object()]),
        context_window=80,
        engine=SimpleNamespace(_live=[(SimpleNamespace(cache_len=40, finished=False), None),
                                      (SimpleNamespace(cache_len=8, finished=True), None)],
                               context_window=0),
    )
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}requests_running") == "2"
    assert sample(body, f"{metrics.PREFIX}requests_waiting") == "2"
    assert sample(body, f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="0"}}') == "0.5"
    assert f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="1"}}' not in body


def test_a_concurrent_queue_is_waiting_and_its_cache_is_its_own_pool():
    stream = SimpleNamespace(context=list(range(25)), prompt=list(range(20)))
    filling = SimpleNamespace(context=[], prompt=list(range(10)))
    decoder = SimpleNamespace(streams={1: stream}, filling=[filling], live=lambda: 2, context=100)

    class Queue:
        def qsize(self) -> int:
            return 2

    app = SimpleNamespace(engine=SimpleNamespace(
        scheduler=SimpleNamespace(decoder=decoder, waiting=Queue(), held=object(), max_streams=4),
        context_window=100), health=SimpleNamespace(live=[1, 2, 3, 4]))
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}requests_running") == "2"
    assert sample(body, f"{metrics.PREFIX}requests_waiting") == "3"
    assert sample(body, f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="0"}}') == "0.25"
    assert sample(body, f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="1"}}') == "0.1"


def test_cuda_health_folds_drafts_and_drops_running_when_the_request_ends():
    app = SimpleNamespace()
    out: list[int] = []
    with health.of(app).running(5, out) as request:
        assert sample(metrics.render(app), f"{metrics.PREFIX}requests_running") == "1"
        out.extend([7, 8, 9])
        request.saw()
        request.stats = {"drafted": 4, "accepted": 2}
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}requests_running") == "0"
    assert sample(body, f"{metrics.PREFIX}prompt_tokens_total") == "5"
    assert sample(body, f"{metrics.PREFIX}generation_tokens_total") == "3"
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == "4"
    assert sample(body, f"{metrics.PREFIX}mtp_accepted_total") == "2"
    assert sample(body, f"{metrics.PREFIX}time_to_first_token_seconds_count") == "1"
    assert health.of(app).snapshot(app)["requests_running"] == 0
    assert health.of(app).snapshot(app)["drafted_total"] == 4


def test_mac_finish_request_counts_once():
    app = SimpleNamespace()
    metrics.begin(app, 6, time.perf_counter())
    metrics.tokens(3, time.perf_counter())
    metrics.bind(SimpleNamespace(stream=SimpleNamespace(drafted=5, accepted=2)))
    metrics.finish_request()
    metrics.finish_request()
    body = metrics.render(app)
    assert sample(body, f"{metrics.PREFIX}prompt_tokens_total") == "6"
    assert sample(body, f"{metrics.PREFIX}generation_tokens_total") == "3"
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == "5"
    assert sample(body, f"{metrics.PREFIX}mtp_accepted_total") == "2"
    assert sample(body, f"{metrics.PREFIX}request_latency_seconds_count") == "1"
    assert sample(body, f"{metrics.PREFIX}time_to_first_token_seconds_count") == "1"


def test_a_mac_chat_counts_the_reply_the_http_thread_returns():
    pytest.importorskip("mlx.core")
    from tests.test_lane_server import make_app

    app = make_app(lanes=1, use_proposer=False)
    try:
        reply = app.chat([{"role": "user", "content": "hi"}], max_tokens=4)
        body = metrics.render(app)
        drafted = int((reply.get("speculative") or {}).get("drafted", 0))
        accepted = int((reply.get("speculative") or {}).get("accepted", 0))
    finally:
        app.close()
    assert sample(body, f"{metrics.PREFIX}requests_running") == "0"
    assert sample(body, f"{metrics.PREFIX}requests_waiting") == "0"
    assert sample(body, f"{metrics.PREFIX}prompt_tokens_total") == str(reply["prompt_tokens"])
    assert sample(body, f"{metrics.PREFIX}generation_tokens_total") == str(reply["completion_tokens"])
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == str(drafted)
    assert sample(body, f"{metrics.PREFIX}mtp_accepted_total") == str(accepted)
    assert sample(body, f"{metrics.PREFIX}request_latency_seconds_count") == "1"
    assert int(reply["completion_tokens"]) > 0
    assert reply["runtime"]["time_to_first_token"] is not None
    assert sample(body, f"{metrics.PREFIX}time_to_first_token_seconds_count") == "1"


def test_a_live_request_is_running_and_the_next_one_is_waiting(tmp_path):
    import pytest

    pytest.importorskip("jinja2")
    from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, WAIT, app_for, post, serving, until

    engine = PacedEngine(hold_at=0)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        assert sample(get(port, "/metrics")[2], f"{metrics.PREFIX}requests_running") == "0"
        box: dict = {}

        def run(key: str, tokens: int) -> None:
            box[key] = post(port, {"messages": MESSAGES, "max_tokens": tokens})

        first = threading.Thread(target=run, args=("first", 4))
        first.start()
        assert engine.held.wait(WAIT)
        during = get(port, "/metrics")[2]
        assert sample(during, f"{metrics.PREFIX}requests_running") == "1"
        assert sample(during, f"{metrics.PREFIX}requests_waiting") == "0"
        second = threading.Thread(target=run, args=("second", 2))
        second.start()
        until(lambda: getattr(app, "turns", None) is not None and app.turns.parked == 1,
              "the second request to wait")
        waited = get(port, "/metrics")[2]
        assert sample(waited, f"{metrics.PREFIX}requests_running") == "1"
        assert sample(waited, f"{metrics.PREFIX}requests_waiting") == "1"
        engine.release.set()
        first.join(WAIT)
        second.join(WAIT)
        assert box["first"][0] == 200 and box["second"][0] == 200, box
        done = get(port, "/metrics")[2]
        assert sample(done, f"{metrics.PREFIX}requests_running") == "0"
        assert sample(done, f"{metrics.PREFIX}requests_waiting") == "0"
        assert sample(done, f"{metrics.PREFIX}generation_tokens_total") == "6"
        assert int(sample(done, f"{metrics.PREFIX}prompt_tokens_total")) > 0
        assert sample(done, f"{metrics.PREFIX}request_latency_seconds_count") == "2"
        assert sample(done, f"{metrics.PREFIX}time_to_first_token_seconds_count") == "2"


def test_the_vllm_mirror_names_carry_the_same_readings():
    # A vLLM dashboard filled by swapping the "tensorfold:" prefix must read identical values.
    idle = SimpleNamespace()
    body = metrics.render(idle)
    for native, mirror in (("requests_running", "num_requests_running"),
                           ("requests_waiting", "num_requests_waiting"),
                           ("kv_cache_usage_ratio", "kv_cache_usage_perc"),
                           ("mtp_drafted_total", "spec_decode_num_draft_tokens_total"),
                           ("mtp_accepted_total", "spec_decode_num_accepted_tokens_total")):
        assert sample(body, f"{metrics.PREFIX}{native}") == sample(body, f"{metrics.PREFIX}{mirror}")
    assert sample(body, f'{metrics.PREFIX}kv_cache_usage_perc{{stream="0"}}') == "0"
    app = SimpleNamespace(scheduler=SimpleNamespace(active=1, waiting=2, filling=[object()]),
                          context_window=80,
                          engine=SimpleNamespace(_live=[(SimpleNamespace(cache_len=40, finished=False), None)],
                                                 context_window=0))
    metrics.note(app, prompt=4, generation=1, drafted=3, accepted=1, latency=0.2, ttft=0.02)
    body = metrics.render(app)
    assert (sample(body, f"{metrics.PREFIX}requests_running")
            == sample(body, f"{metrics.PREFIX}num_requests_running") == "2")
    assert (sample(body, f"{metrics.PREFIX}requests_waiting")
            == sample(body, f"{metrics.PREFIX}num_requests_waiting") == "2")
    assert sample(body, f'{metrics.PREFIX}kv_cache_usage_ratio{{pool="0"}}') == \
        sample(body, f'{metrics.PREFIX}kv_cache_usage_perc{{stream="0"}}') == "0.5"
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == \
        sample(body, f"{metrics.PREFIX}spec_decode_num_draft_tokens_total") == "3"
    assert sample(body, f"{metrics.PREFIX}mtp_accepted_total") == \
        sample(body, f"{metrics.PREFIX}spec_decode_num_accepted_tokens_total") == "1"
    assert bucket(body, "e2e_request_latency_seconds", "+Inf") == bucket(body, "request_latency_seconds", "+Inf") == "1"


def test_the_event_counters_are_read_where_the_server_keeps_them():
    mac = SimpleNamespace(scheduler=SimpleNamespace(active=0, waiting=1, filling=[],
                                                    cancelled=2, preemptions=3, failed_rounds=1))
    body = metrics.render(mac)
    assert sample(body, f"{metrics.PREFIX}client_disconnections_total") == "2"
    assert sample(body, f"{metrics.PREFIX}preemptions_total") == "3"
    assert f"{metrics.PREFIX}request_failures_total" not in body

    class Queue:
        def qsize(self) -> int:
            return 1

    cuda = SimpleNamespace(engine=SimpleNamespace(
        scheduler=SimpleNamespace(decoder=SimpleNamespace(live=lambda: 0, streams={}, filling=()),
                                 waiting=Queue(), held=None, yields=4),
        context_window=100))
    assert sample(metrics.render(cuda), f"{metrics.PREFIX}preemptions_total") == "4"
    assert f"{metrics.PREFIX}client_disconnections_total" not in metrics.render(cuda)


def test_both_http_layers_serve_the_mirrored_and_event_families(tmp_path):
    pytest.importorskip("jinja2")
    from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, WAIT, app_for, post, serving, until

    pairs = (("requests_running", "num_requests_running"), ("requests_waiting", "num_requests_waiting"),
             ("kv_cache_usage_ratio", "kv_cache_usage_perc"),
             ("mtp_drafted_total", "spec_decode_num_draft_tokens_total"),
             ("mtp_accepted_total", "spec_decode_num_accepted_tokens_total"))
    mac_app = SimpleNamespace(served_name="test", model_ids=["test"], max_batch_size=1,
                              scheduler=SimpleNamespace(active=1, waiting=2, filling=[], cancelled=1,
                                                        preemptions=2, failed_rounds=0),
                              context_window=80)
    httpd, thread = serve(mac_app)
    try:
        body = get(httpd.server_port, "/metrics")[2]
        for native, mirror in pairs:
            assert sample(body, f"{metrics.PREFIX}{native}") == sample(body, f"{metrics.PREFIX}{mirror}")
        assert sample(body, f"{metrics.PREFIX}client_disconnections_total") == "1"
        assert sample(body, f"{metrics.PREFIX}preemptions_total") == "2"
        alias = get(httpd.server_port, "/v1/metrics")
        assert alias[0] == 200 and alias[2] == body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)

    engine = PacedEngine(hold_at=0)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        idle = get(port, "/metrics")[2]
        for native, mirror in pairs:
            assert sample(idle, f"{metrics.PREFIX}{native}") == sample(idle, f"{metrics.PREFIX}{mirror}")

        box: dict = {}

        def run(key: str, tokens: int) -> None:
            box[key] = post(port, {"messages": MESSAGES, "max_tokens": tokens})

        first = threading.Thread(target=run, args=("first", 4))
        first.start()
        assert engine.held.wait(WAIT)
        during = get(port, "/metrics")[2]
        for native, mirror in pairs:
            assert sample(during, f"{metrics.PREFIX}{native}") == sample(during, f"{metrics.PREFIX}{mirror}")
        assert sample(during, f"{metrics.PREFIX}num_requests_running") == "1"
        second = threading.Thread(target=run, args=("second", 2))
        second.start()
        until(lambda: getattr(app, "turns", None) is not None and app.turns.parked == 1,
              "the second request to wait")
        waited = get(port, "/metrics")[2]
        assert sample(waited, f"{metrics.PREFIX}num_requests_running") == "1"
        assert sample(waited, f"{metrics.PREFIX}num_requests_waiting") == "1"
        engine.release.set()
        first.join(WAIT)
        second.join(WAIT)
        assert box["first"][0] == 200 and box["second"][0] == 200, box
