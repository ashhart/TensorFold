"""Issue #110: the CUDA server's GET /metrics — vLLM's metric names, the server's own numbers."""

from contextlib import contextmanager
import http.client
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tests.prometheus_format import check_histograms, parse, series_value
from tests.test_cuda_server_disconnect import (CountingTurns, PacedDecoder, PacedEngine, SchedulerEngine, app_for,
                                                leave, post, send, serving, until)
from tensorfold.cuda import health
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.metrics import TIME_TO_FIRST_TOKEN_BUCKETS, E2E_REQUEST_LATENCY_BUCKETS, Histogram

WAIT = 10


def get(port, path="/metrics"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    headers = {name.lower(): value for name, value in response.getheaders()}
    connection.close()
    return response.status, headers, body


USED_PORTS: set[int] = set()


@contextmanager
def fresh_serving(app):
    """The shared serving helper, retried while its port is one an earlier test's server used: with SO_REUSEADDR a
    closing server's port can still accept connections for a moment, and a second test binding that same port would
    have half its requests answered by the first test's dying app. Ports stay retired for the session."""

    for _ in range(8):
        with serving(app) as port:
            if port not in USED_PORTS:
                USED_PORTS.add(port)
                yield port
                return
        time.sleep(0.02)                 # let the earlier server's close free the port
    raise AssertionError("every port the server got was still held by an earlier test's server")


# the names and HELP texts are vLLM's, verbatim, so a vLLM dashboard scrapes /metrics with a prefix swap
EXPECTED = {
    "tensorfold:num_requests_running": ("gauge", "Number of running requests"),
    "tensorfold:num_requests_waiting": ("gauge", "Number of requests waiting to be processed"),
    "tensorfold:prompt_tokens_total": ("counter", "Number of prefill tokens processed"),
    "tensorfold:generation_tokens_total": ("counter", "Number of generation tokens processed"),
    "tensorfold:spec_decode_num_draft_tokens_total": ("counter", "SpecDecoding: Number of draft tokens"),
    "tensorfold:spec_decode_num_accepted_tokens_total": ("counter", "SpecDecoding: Number of accepted tokens"),
    "tensorfold:time_to_first_token_seconds": ("histogram", "Latency until first output"),
    "tensorfold:e2e_request_latency_seconds": ("histogram", "E2E request latency"),
}


# -- the endpoint ---------------------------------------------------------------------------

def test_metrics_serves_vllm_names_and_helps(tmp_path):
    app = app_for(tmp_path, PacedEngine())
    with fresh_serving(app) as port:
        status, headers, body = get(port)
        assert status == 200
        assert headers["content-type"] == "text/plain; version=0.0.4"
        parsed = parse(body)
        for name, (kind, help) in EXPECTED.items():
            assert name in parsed, f"{name} missing from the exposition"
            assert parsed[name]["type"] == kind, f"{name} is {parsed[name]['type']}, vLLM's is {kind}"
            assert parsed[name]["help"] == help, f"{name}'s help is {parsed[name]['help']!r}, vLLM's is {help!r}"
        # the engine this app runs has no drafter and no pool of its own: no kv_cache_usage_perc, and
        # spec_decode_num_drafts_total stays out because a draft step is not tracked separately from rounds
        assert "tensorfold:kv_cache_usage_perc" not in parsed
        assert "tensorfold:spec_decode_num_drafts_total" not in parsed
        check_histograms(parsed)
        status, headers, body = get(port, "/v1/metrics")
        assert status == 200
        assert headers["content-type"] == "text/plain; version=0.0.4"
        parse(body)


# -- counters: the /health totals, counted as requests end -----------------------------------

def test_counters_fold_each_finished_request(tmp_path):
    engine = PacedEngine("Hi")
    app = app_for(tmp_path, engine)
    with fresh_serving(app) as port:
        first_prompt = len(app.tok.encode("user:Hi;assistant:").ids)
        status, body = post(port, {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3})
        assert status == 200
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == first_prompt
        assert series_value(parsed, "tensorfold:generation_tokens_total") == 3
        assert series_value(parsed, "tensorfold:time_to_first_token_seconds_count") == 1
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 1

        status, body = post(port, {"messages": [{"role": "user", "content": "there!"}], "max_tokens": 3})
        assert status == 200
        second_prompt = len(app.tok.encode("user:there!;assistant:").ids)
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == first_prompt + second_prompt
        assert series_value(parsed, "tensorfold:generation_tokens_total") == 6
        assert series_value(parsed, "tensorfold:time_to_first_token_seconds_count") == 2
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 2
        check_histograms(parsed)


# -- gauges: running and waiting track the engine and the queue ------------------------------

def test_gauges_track_running_and_waiting(tmp_path):
    engine = PacedEngine("ab", hold_at=1)
    app = app_for(tmp_path, engine)
    app.turns = CountingTurns()
    with fresh_serving(app) as port:
        first = send(port, {"messages": [{"role": "user", "content": "first"}], "max_tokens": 3})
        assert engine.held.wait(WAIT)
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:num_requests_running") == 1
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 0
        # a running request has not ended, so nothing has been observed into the latencies yet
        assert series_value(parsed, "tensorfold:time_to_first_token_seconds_count") == 0
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 0

        second = send(port, {"messages": [{"role": "user", "content": "second"}], "max_tokens": 3})
        until(lambda: app.turns.waiting == 1, "the second request to queue behind the first")
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:num_requests_running") == 1
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 1

        engine.release.set()
        response = http.client.HTTPResponse(first)
        response.begin()
        response.read()
        first.close()
        leave(second)
        until(lambda: not app.turns.busy and not app.turns.waiting, "the held request to finish and the queue to drain")
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:num_requests_running") == 0
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 0
        assert series_value(parsed, "tensorfold:time_to_first_token_seconds_count") == 2
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 2
        check_histograms(parsed)


def test_concurrent_engine_reports_admitted_streams_and_queue(tmp_path):
    # a sequential post waits for its reply, so the three requests go out on three threads at once
    engine = SchedulerEngine()
    app = app_for(tmp_path, engine)
    with fresh_serving(app) as port:
        posts = [threading.Thread(target=post, args=(port, {"messages": [{"role": "user", "content": content}],
                                                              "max_tokens": 300}))
                 for content in ("one", "two", "three")]
        for started in posts:
            started.start()
        deadline = time.monotonic() + WAIT
        running = waiting = -1
        while time.monotonic() < deadline:
            parsed = parse(get(port)[2])
            running = series_value(parsed, "tensorfold:num_requests_running")
            waiting = series_value(parsed, "tensorfold:num_requests_waiting")
            if running == 2 and waiting == 1:
                break
            time.sleep(0.005)
        assert (running, waiting) == (2, 1), f"the admission window passed: saw running={running}, waiting={waiting}"
        for finished in posts:
            finished.join(WAIT)
        until(lambda: len(engine.decoder.streams) == 0 and engine.scheduler.waiting.qsize() == 0,
              "every stream to finish and the queue to drain")
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:num_requests_running") == 0
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 0
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 3
        # the fake decoder gives its streams no pool state of their own, so the KV gauge stays out of the body
        assert "tensorfold:kv_cache_usage_perc" not in parsed
        check_histograms(parsed)


# -- the KV cache gauge: each stream's own pool ----------------------------------------------

def test_serial_engine_reports_its_pool_while_a_request_runs(tmp_path):
    class PooledEngine(PacedEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.e = SimpleNamespace(st=SimpleNamespace(pos=0, capacity=100))

    engine = PooledEngine("ab", hold_at=1)
    app = app_for(tmp_path, engine)
    with fresh_serving(app) as port:
        assert "tensorfold:kv_cache_usage_perc" not in parse(get(port)[2])     # no request, no pool in use
        first = send(port, {"messages": [{"role": "user", "content": "first"}], "max_tokens": 3})
        assert engine.held.wait(WAIT)
        engine.e.st.pos = 10
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:kv_cache_usage_perc", {"stream": "0"}) == 0.1
        engine.release.set()
        response = http.client.HTTPResponse(first)
        response.begin()
        response.read()
        first.close()
        until(lambda: len(health.of(app).live) == 0, "the held request to finish")
        assert "tensorfold:kv_cache_usage_perc" not in parse(get(port)[2])     # the pool is back behind the idle engine


class HoldingPacedDecoder(PacedDecoder):
    """The fake decoder, holding every round while ``held`` is set and giving each stream a 64-row pool."""

    def __init__(self):
        super().__init__()
        self.held, self.release = threading.Event(), threading.Event()

    def admit(self, stream):
        stream.st = SimpleNamespace(pos=len(stream.prompt), limit=64)
        super().admit(stream)

    def round(self):
        if self.held.is_set():
            if self.release.wait(0.05):
                self.held.clear()
                return []
            time.sleep(0.01)
            return []
        done = super().round()
        for stream in self.streams:
            if stream.st is not None and not stream.done:
                stream.st.pos += 1
        return done


def test_concurrent_engine_reports_each_streams_pool(tmp_path):
    decoder = HoldingPacedDecoder()
    engine = SchedulerEngine.__new__(SchedulerEngine)
    engine.eos = (0,)
    engine.concurrent = True
    engine.decoder = decoder
    engine.scheduler = Scheduler(decoder, max_streams=2)
    app = app_for(tmp_path, engine)
    with fresh_serving(app) as port:
        first = send(port, {"messages": [{"role": "user", "content": "one"}], "max_tokens": 60})
        until(lambda: len(decoder.seen) >= 1, "the first stream to be admitted")
        decoder.held.set()
        stream = decoder.seen[0]
        stream.st.pos = 32
        parsed = parse(get(port)[2])
        assert series_value(parsed, "tensorfold:kv_cache_usage_perc", {"stream": str(stream.sid)}) == 0.5
        assert series_value(parsed, "tensorfold:num_requests_running") == 1
        decoder.release.set()
        until(lambda: len(decoder.streams) == 0 and not decoder.held.is_set(), "the held stream to finish")
        response = http.client.HTTPResponse(first)
        response.begin()
        response.read()
        first.close()
        parsed = parse(get(port)[2])
        assert "tensorfold:kv_cache_usage_perc" not in parsed
        assert series_value(parsed, "tensorfold:num_requests_running") == 0


# -- the histogram buckets: vLLM's, and they bucket honestly ----------------------------------

def test_ttft_buckets_land_observations_where_vllm_would():
    histogram = Histogram(TIME_TO_FIRST_TOKEN_BUCKETS)
    for value in (0.001, 0.0025, 0.003, 1.0, 61.0):
        histogram.observe(value)
    counts = list(histogram._counts)
    index = {bound: i for i, bound in enumerate(TIME_TO_FIRST_TOKEN_BUCKETS)}
    assert counts[index[0.0025]] == 2          # 0.001 and 0.0025 (the edge belongs to its own bucket)
    assert counts[index[0.005]] == 3           # 0.003 rounds up
    assert counts[index[1.0]] == 4             # 1.0 is on an edge
    assert counts[-1] == 5                      # 61.0 only reaches +Inf
    assert histogram.count == 5 and histogram.sum == 0.001 + 0.0025 + 0.003 + 1.0 + 61.0

    e2e = Histogram(E2E_REQUEST_LATENCY_BUCKETS)
    for value in (0.04, 0.05, 60.0, 61.0):
        e2e.observe(value)
    counts = list(e2e._counts)
    index = {bound: i for i, bound in enumerate(E2E_REQUEST_LATENCY_BUCKETS)}
    assert counts[index[0.05]] == 2            # both sub-first-bucket and on-edge observations
    assert counts[index[1.0]] == 2             # 60.0 and 61.0 pass the 0.05..1.0 buckets
    assert counts[index[40.0]] == 2
    assert counts[index[60.0]] == 3            # 61.0 outgrows 60.0
    assert counts[-1] == 4
