"""Benchmark HTTP tests use synthetic output and reserved public model labels."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tensorfold.benchmark import client
from tensorfold.benchmark.client import stream_request
from tensorfold.benchmark.protocol import ERROR_CODES, FIXTURES, SAMPLE_FIELDS, SUITE_SHA256, summary
from tensorfold.benchmark.runner import run_suite


def _event(value, *, multiline=False):
    if isinstance(value, str):
        return f"data: {value}\n\n".encode()
    text = json.dumps(value, indent=2) if multiline else json.dumps(value)
    return "".join(f"data: {line}\r\n" for line in text.splitlines()).encode() + b"\r\n"


def _reply(*, completion_tokens=4, usage=True, reasoning=False, done=True, runtime=True, batch=False):
    first = {"choices": [{"index": 0, "delta": {"reasoning_content" if reasoning else "content": "synthetic"}}]}
    second = {"choices": [{"index": 0, "delta": {"content": " output"}}]}
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    if runtime:
        final["tensorfold"] = {
            "decode_tps": 42.0, "decode_s": 0.2, "prefill_s": 0.03, "cached": 0,
            "token_sha": "0123456789ab",
        }
    usage_event = {"choices": [], "usage": {
        "prompt_tokens": 24, "completion_tokens": completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
    }}
    blocks = [b": keepalive\n\n" + _event(first, multiline=True), _event(second), _event(final)]
    if usage:
        blocks.append(_event(usage_event))
    if done:
        blocks.append(_event("[DONE]"))
    if batch:
        return [(0, b"".join(blocks))]
    return [(0, blocks[0]), (0.012, blocks[1]), (0, b"".join(blocks[2:]))]


@contextlib.contextmanager
def _server(make_reply, *, status=200, content_type="text/event-stream"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            requests.append({"path": self.path, "body": body})
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            try:
                for delay, data in make_reply(body, len(requests)):
                    if delay:
                        time.sleep(delay)
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _run(base, *, fixture=FIXTURES[0], **kwargs):
    return stream_request(
        base, "example/model", fixture, tokens=4, temperature=1.0, seed=1234, repeat=0, **kwargs,
    )


def test_stream_multiline_usage_final_and_reported_runtime():
    with _server(lambda *_: _reply(reasoning=True)) as (base, requests):
        sample = _run(base, fixture=FIXTURES[1], serial=True)
    assert sample["status"] == "ok"
    assert sample["prompt_tokens"] == 24
    assert sample["completion_tokens"] == 4
    assert sample["cached_tokens"] == 0
    assert sample["delivery_seconds"] > 0
    assert sample["delivery_tps"] == pytest.approx(3 / sample["delivery_seconds"])
    assert sample["ttft_seconds"] < sample["end_to_end_seconds"]
    assert sample["server_decode_tps"] == 42
    assert sample["server_decode_seconds"] == 0.2
    assert sample["prefill_seconds"] == 0.03
    assert sample["token_sha"] == "0123456789ab"
    assert set(sample) == SAMPLE_FIELDS
    assert requests[0]["path"] == "/v1/chat/completions"
    body = requests[0]["body"]
    assert body["draft"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "ignore_eos" not in body
    assert body["top_k"] == 20 and body["top_p"] == 0.95
    assert "synthetic" not in json.dumps(sample)
    assert base not in json.dumps(sample)


def test_batch_delivery_is_not_a_fabricated_engine_rate():
    with _server(lambda *_: _reply(batch=True)) as (base, _):
        sample = _run(base)
    assert sample["status"] == "unmeasured"
    assert sample["error_code"] == "unmeasured_delivery"
    assert sample["delivery_seconds"] is None and sample["delivery_tps"] is None
    assert sample["server_decode_tps"] == 42


def test_missing_server_telemetry_stays_unmeasured():
    with _server(lambda *_: _reply(runtime=False)) as (base, _):
        sample = _run(base)
    assert sample["status"] == "ok"
    assert sample["delivery_tps"] > 0
    assert sample["server_decode_tps"] is None
    assert sample["server_decode_seconds"] is None
    assert sample["prefill_seconds"] is None
    assert sample["token_sha"] is None


@pytest.mark.parametrize("kwargs,code", [
    ({"usage": False}, "missing_usage"),
    ({"done": False}, "truncated_stream"),
    ({"completion_tokens": -1}, "invalid_usage"),
    ({"completion_tokens": True}, "invalid_usage"),
    ({"completion_tokens": 0}, "empty_output"),
    ({"completion_tokens": 5}, "token_count_mismatch"),
])
def test_unusable_streams_are_retained(kwargs, code):
    with _server(lambda *_: _reply(**kwargs)) as (base, _):
        sample = _run(base)
    assert sample["status"] == "error"
    assert sample["error_code"] == code
    assert sample["error_code"] in ERROR_CODES
    assert set(sample) == SAMPLE_FIELDS


def test_early_eos_is_not_rankable():
    with _server(lambda *_: _reply(completion_tokens=2)) as (base, _):
        sample = _run(base)
    assert sample["status"] == "early_eos"
    assert sample["completion_tokens"] == 2
    assert sample["delivery_tps"] > 0
    rows = [{**sample, "repeat": repeat} for repeat in range(5)]
    cell = summary(rows)[0]
    assert cell["protocol_complete"] is False
    assert cell["delivery_tps"] is None
    assert cell["status_counts"] == {"early_eos": 5}


@pytest.mark.parametrize("data,code", [
    (_event({"error": {"message": "private upstream URL must not escape"}}), "server_error"),
    (_event("{invalid JSON}"), "invalid_stream"),
    (b"data: [DONE]", "truncated_stream"),
    (_event("[DONE]"), "unexpected_finish"),
    (_event({"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 4}}) + _event("[DONE]"), "empty_output"),
])
def test_errors_have_fixed_codes_and_no_private_messages(data, code):
    with _server(lambda *_: [(0, data)]) as (base, _):
        sample = _run(base)
    assert sample["status"] == "error" and sample["error_code"] == code
    assert "private" not in json.dumps(sample)


def test_http_errors_and_wrong_content_type():
    with _server(lambda *_: [(0, b"not public")], status=503, content_type="application/json") as (base, _):
        sample = _run(base)
    assert sample["error_code"] == "http_error"
    with _server(lambda *_: [(0, b"{}")], content_type="application/json") as (base, _):
        sample = _run(base)
    assert sample["error_code"] == "invalid_stream"


def test_timeout_is_bounded_even_when_server_trickles_bytes():
    def trickle(*_):
        return [(0.004, b"x") for _ in range(40)]
    with _server(trickle) as (base, _):
        started = time.perf_counter()
        sample = _run(base, timeout=0.04)
        elapsed = time.perf_counter() - started
    assert sample["error_code"] == "timeout"
    assert elapsed < 0.3


def test_cancelled_request_is_a_sample_and_stops_suite(monkeypatch):
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(client.urllib.request, "urlopen", cancel)
    samples = run_suite("http://127.0.0.1:1", "example/model", tokens=4, repetitions=2)
    assert len(samples) == 1
    assert samples[0]["repeat"] == -1
    assert samples[0]["status"] == "error"
    assert samples[0]["error_code"] == "cancelled"


def test_suite_settings_warmup_and_repeat_seeds():
    progress = []
    with _server(lambda *_: _reply()) as (base, requests):
        samples = run_suite(base + "/v1", "example/model", tokens=4, repetitions=2, progress=progress.append)
    assert len(samples) == 8 and len(requests) == 12 and len(progress) == 12
    assert all(sample["repeat"] != -1 for sample in samples)
    for group in range(4):
        bodies = [request["body"] for request in requests[group * 3:group * 3 + 3]]
        assert [body["seed"] for body in bodies] == [1234, 1234, 1235]
        assert all(body["max_tokens"] == 4 for body in bodies)
    assert [sample["temperature"] for sample in samples] == [1.0] * 4 + [0.0] * 4
    assert all("top_k" not in request["body"] for request in requests[6:])
    assert len(SUITE_SHA256) == 64


def test_failed_warmup_prevents_misleading_complete_summary():
    def replies(body, number):
        return [(0, _event({"error": {"message": "warmup failed"}}))] if number == 1 else _reply()
    with _server(replies) as (base, _):
        samples = run_suite(base, "example/model", tokens=4, temperatures=(0.0,))
    first_cell = [sample for sample in samples if sample["fixture_id"] == "code"]
    assert len(first_cell) == 6 and first_cell[0]["repeat"] == -1
    cell = summary(first_cell)[0]
    assert cell["protocol_complete"] is False
    assert cell["status_counts"] == {"error": 1, "ok": 5}
    assert cell["delivery_tps"] is None


def test_summary_requires_every_repeat_and_optional_metric():
    with _server(lambda *_: _reply()) as (base, _):
        sample = _run(base)
    rows = [{**sample, "repeat": repeat, "delivery_tps": float(repeat + 1)} for repeat in range(5)]
    cell = summary(rows)[0]
    assert cell["protocol_complete"] is True
    assert cell["delivery_tps"] == {"median": 3, "min": 1, "max": 5, "spread": 4}
    rows[0]["server_decode_tps"] = None
    assert summary(rows)[0]["server_decode_tps"] is None
    rows[0]["status"] = "error"
    assert summary(rows)[0]["delivery_tps"] is None


def test_custom_repetitions_summary_never_mislabels_partial_results():
    with _server(lambda *_: _reply()) as (base, _):
        sample = _run(base)
    cell = summary([sample], expected_repetitions=1)[0]
    assert cell["protocol_complete"] is False
    assert cell["delivery_tps"]["median"] == sample["delivery_tps"]
    assert summary([sample], expected_repetitions=2)[0]["delivery_tps"] is None


def test_sse_byte_boundaries_and_utf8_preserve_event_content():
    event = 'data: {"choices": [],\r\ndata: "marker": "λ"}\r\n\r\n'.encode()

    class FragmentedResponse:
        def __init__(self):
            self.blocks = iter(bytes([byte]) for byte in event)

        def read1(self, _size):
            return next(self.blocks, b"")

    events = list(client._events(FragmentedResponse(), time.perf_counter() + 1))
    assert len(events) == 1
    assert json.loads(events[0][0]) == {"choices": [], "marker": "λ"}


def test_runtime_hash_does_not_substitute_output_and_ignores_invalid_timings():
    values = client._runtime_fields({
        "token_sha": "synthetic output", "decode_s": float("nan"),
        "decode_tps": float("inf"), "prefill_s": -1,
    })
    assert all(value is None for value in values.values())


@pytest.mark.parametrize("kwargs", [
    {"tokens": 0}, {"tokens": True}, {"repetitions": 0}, {"timeout": 0},
    {"timeout": float("nan")}, {"temperatures": ()}, {"temperatures": (0, 0)},
    {"temperatures": (float("inf"),)}, {"temperatures": (-1,)},
])
def test_invalid_configuration_fails_before_network(kwargs):
    with pytest.raises(ValueError):
        run_suite("http://127.0.0.1:1", "example/model", **kwargs)


@pytest.mark.parametrize("url", ["file:///tmp/model", "http://test:password@example.test", "https://example.test?key=x"])
def test_server_urls_require_supported_transport_without_credentials(url):
    with pytest.raises(ValueError):
        _run(url)
