from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from tensorfold.control.telemetry import (Client, Sample, Rates, base_url, normalize, numeric, parse_metrics)
from tensorfold.control.safety import ControlError


@contextmanager
def endpoint(route):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            code, body, headers = route(self.path, self.headers)
            self.send_response(code)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://u:p@localhost:8", "http://",
    "http://localhost?token=x", "http://localhost/#frag", "http://localhost:99999", "http://a\r\nHost: b"])
def test_reject_bad_endpoint(url):
    with pytest.raises(ControlError):
        base_url(url)


def test_endpoint_normalization():
    assert base_url("http://[::1]:8080/v1/") == "http://[::1]:8080"
    assert base_url("https://example.invalid/api/v1") == "https://example.invalid/api"


def test_real_http_mac_contract_and_alias_deduplication():
    metrics = b'''# TYPE tensorfold:requests_running gauge
    tensorfold:requests_running 3
    tensorfold:num_requests_running 3
    tensorfold:requests_waiting 1
    tensorfold:generation_tokens_total 1000
    vllm:generation_tokens_total 1000
    tensorfold:prompt_tokens_total 8000
    tensorfold:mtp_drafted_total 100
    tensorfold:mtp_accepted_total 81
    tensorfold:time_to_first_token_seconds_sum 4.5
    tensorfold:time_to_first_token_seconds_count 3
    tensorfold:kv_cache_usage_ratio{pool="a"} 0.2
    tensorfold:kv_cache_usage_ratio{pool="b"} 0.7
    '''
    metrics = b"\n".join(line.strip() for line in metrics.splitlines())
    requests = []
    def route(path, headers):
        requests.append(path)
        if path == "/health":
            return 200, json.dumps({"status": "ok", "warming": False, "model": "Fixture", "memory":
                                    {"active": 2000, "cache": 100, "peak": 2200}}).encode(), {}
        return 200, metrics, {}
    with endpoint(route) as url:
        sample = Client(url).sample()
    assert requests == ["/health", "/metrics"]
    assert sample.online and sample.running == 3 and sample.waiting == 1
    assert sample.counters["generation"] == 1000
    assert sample.acceptance == 0.81 and sample.ttft_mean == 1.5
    assert sample.kv_ratio == 0.7 and sample.memory == 2000
    assert sample.sources["generation"] == "completed requests"


def test_cuda_live_counter_wins_over_finished_metrics():
    sample = normalize(1, {"ok": True, "completion_tokens_total": 100, "requests_running": 2},
                       {"tensorfold:generation_tokens_total": [70]})
    assert sample.counters["generation"] == 100
    assert sample.sources["generation"] == "live counter"


def test_missing_stats_not_zero():
    sample = normalize(1, {"status": "ok"}, {})
    assert sample.counters == {}
    assert all(v is None for v in [sample.running, sample.waiting, sample.memory, sample.acceptance])


def test_bad_metrics_and_escaped_labels():
    metrics = parse_metrics('''# HELP example test
bad NaN
bad2 +Inf
bad3 1e999
bad4 -20
valid{pool="a,b\\\"c"} 2.5e2
valid{pool="z"} 1 12345
junk invalid text
''')
    assert metrics == {"valid": [250.0, 1.0]}


def test_http_unauthorized_not_healthy():
    with endpoint(lambda *_: (401, b"{}", {})) as url:
        sample = Client(url).sample()
    assert sample.phase == "unauthorized" and not sample.online


def test_redirect_never_forwards_credentials():
    calls = []
    def route(path, headers):
        calls.append((path, headers.get("Authorization")))
        return 302, b"", {"Location": "/capture"}
    with endpoint(route) as url:
        assert not Client(url, "private").sample().online
    assert calls == [("/health", "Bearer private")]


def test_body_limit_and_missing_metrics():
    with endpoint(lambda *_: (200, b"x" * ((1 << 20) + 1), {})) as url:
        assert "exceeds" in Client(url).sample().error
    def route(path, _):
        return (200, b'{"status":"ok"}', {}) if path == "/health" else (404, b"{}", {})
    with endpoint(route) as url:
        sample = Client(url).sample()
    assert sample.online and sample.counters == {} and "404" in sample.warning


def test_warming_is_distinct_from_ready():
    def route(path, _):
        return (200, b'{"status":"ok","warming":true}', {}) if path == "/health" else (404, b"{}", {})
    with endpoint(route) as url:
        sample = Client(url).sample()
    assert sample.online and sample.phase == "warming"


def counter(at, value, source="live counter"):
    return Sample(at, True, "ready", counters={"generation": value}, sources={"generation": source})


def test_first_sample_is_baseline_not_rate():
    rates = Rates()
    assert rates.update(counter(1, 9999))["generation"] is None
    assert rates.update(counter(3, 10019))["generation"] == 10


@pytest.mark.parametrize("second", [counter(3, 1), counter(20, 101), counter(1, 101),
                                    counter(3, 101, "completed requests")])
def test_reset_gap_clock_and_source_changes(second):
    rates = Rates()
    rates.update(counter(2, 100))
    assert rates.update(second)["generation"] is None


def test_failed_sample_invalidates_rates():
    rates = Rates()
    rates.update(counter(1, 100))
    rates.update(Sample(3, error="offline"))
    assert rates.update(counter(5, 500))["generation"] is None


def test_idle_and_rolling_window():
    rates = Rates(window=10)
    rates.update(counter(0, 0))
    for t in range(1, 20):
        result = rates.update(counter(t, min(10, t) * 20))
    assert result["generation"] >= 0
    for t in range(20, 32):
        result = rates.update(counter(t, 200))
    assert result["generation"] == 0


@pytest.mark.parametrize("value", [True, False, "1", -1, float("nan"), float("inf"), None])
def test_numeric_unknown(value):
    assert numeric(value) is None
