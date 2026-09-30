"""Prometheus exposition and MLX scheduler accounting."""

from io import BytesIO
from types import SimpleNamespace

from tensorfold.server import metrics
from tensorfold.server.http import make_handler
from tensorfold.server.prompt_fill import Filling
from tensorfold.server.scheduler import ChatJob, Scheduler
from tensorfold.engine.lane_engine import LaneEngine
from tests.lane_fakes import FakeEngine


def get(app, path="/metrics"):
    incoming = f"GET {path} HTTP/1.0\r\n\r\n".encode()

    class Connection:
        def __init__(self):
            self.input, self.output = BytesIO(incoming), bytearray()

        def makefile(self, mode, buffering=None):
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    make_handler(app)(connection, ("127.0.0.1", 0), None)
    headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
    return headers.decode(), body.decode()


def test_metrics_endpoint_exposes_gauges_counters_histograms_and_mlx_pool():
    scheduler = SimpleNamespace(
        lanes=2,
        metrics_snapshot=lambda: {"running": 1, "waiting": 3, "tokens": 50},
    )
    app = SimpleNamespace(scheduler=scheduler, context_window=100)
    metrics.of(app).observe(
        prompt_tokens=12,
        generation_tokens=4,
        latency=2.0,
        ttft=0.25,
        mtp_drafted=3,
        mtp_accepted=2,
    )

    headers, body = get(app, "/v1/metrics/?scrape=1")

    assert "200 OK" in headers
    assert f"Content-Type: {metrics.CONTENT_TYPE}" in headers
    assert "tensorfold:requests_running 1" in body
    assert "tensorfold:requests_waiting 3" in body
    assert "tensorfold:prompt_tokens_total 12" in body
    assert "tensorfold:generation_tokens_total 4" in body
    assert "tensorfold:mtp_drafted_total 3" in body
    assert "tensorfold:mtp_accepted_total 2" in body
    assert 'tensorfold:kv_cache_usage_ratio{pool="mlx"} 0.25' in body
    assert 'tensorfold:request_latency_seconds_bucket{le="2.5"} 1' in body
    assert "tensorfold:request_latency_seconds_count 1" in body
    assert 'tensorfold:time_to_first_token_seconds_bucket{le="0.25"} 1' in body
    assert body.endswith("\n")


def test_metrics_are_isolated_per_app_and_omit_unknown_kv_capacity():
    first, second = SimpleNamespace(), SimpleNamespace()
    metrics.of(first).observe(prompt_tokens=7, generation_tokens=2, latency=1)

    first_body = metrics.of(first).render(running=0, waiting=0).decode()
    second_body = metrics.of(second).render(running=0, waiting=0).decode()

    assert "tensorfold:prompt_tokens_total 7" in first_body
    assert "tensorfold:prompt_tokens_total 0" in second_body
    assert "tensorfold:kv_cache_usage_ratio{" not in second_body


def test_scheduler_terminal_outcomes_count_emitted_tokens_and_mtp_only():
    service = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset())
    service.metrics = metrics.Metrics()
    job = ChatJob("request", [1, 2, 3], 8, 0.0)
    job.started_at = job.received_at = job.submitted_at
    job.first_token_at = job.submitted_at + 0.1
    job.prompt_tokens_processed = len(job.prompt_ids)
    job.stream = SimpleNamespace(emitted=[4, 5], mtp_drafted=3, mtp_accepted=1)

    service._finish(job)

    counters, histograms = service.metrics.snapshot()
    assert counters == {
        "prompt_tokens_total": 3,
        "generation_tokens_total": 2,
        "mtp_drafted_total": 3,
        "mtp_accepted_total": 1,
    }
    assert histograms["request_latency_seconds"].count == 1
    assert histograms["time_to_first_token_seconds"].count == 1


def test_preempted_scheduler_attempt_does_not_double_count_the_logical_request():
    service = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset())
    service.metrics = metrics.Metrics()
    job = ChatJob("preempted", [1, 2], 8, 0.0, preempted=True)
    job.started_at = job.submitted_at
    job.stream = SimpleNamespace(emitted=[3], mtp_drafted=0, mtp_accepted=0)

    service._finish(job)

    counters, histograms = service.metrics.snapshot()
    assert not any(counters.values())
    assert histograms["request_latency_seconds"].count == 0


def test_unprocessed_and_internal_jobs_do_not_pollute_request_metrics():
    service = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset())
    service.metrics = metrics.Metrics()
    unprocessed = ChatJob("cancelled", [1, 2, 3], 8, 0.0)
    unprocessed.started_at = unprocessed.submitted_at
    unprocessed.stream = SimpleNamespace(emitted=[], mtp_drafted=0, mtp_accepted=0)
    internal = ChatJob("warm", [4, 5], 1, 0.0, internal=True, prompt_tokens_processed=2)
    internal.started_at = internal.submitted_at
    internal.stream = SimpleNamespace(emitted=[6], mtp_drafted=0, mtp_accepted=0)

    service._finish(unprocessed)
    service._finish(internal)

    counters, histograms = service.metrics.snapshot()
    assert not any(counters.values())
    assert histograms["request_latency_seconds"].count == 1


def test_mlx_kv_snapshot_uses_materialized_prefill_position():
    service = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset())
    job = ChatJob("filling", list(range(20)), 8, 0.0)
    job.stream = SimpleNamespace(context=list(range(20)), finished=False)
    service._filling = Filling(job, iter(()), set(), position=7)

    snapshot = service.metrics_snapshot()

    assert snapshot["running"] == 1
    assert snapshot["tokens"] == 7


def test_only_head_drafts_increment_the_mtp_stream_totals(monkeypatch):
    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    monkeypatch.setattr(lane_tree, "accept_path", lambda *args: [0, 1])
    monkeypatch.setattr(lane_tree, "tree_paths", lambda parents: ([0, 1, 2], []))
    engine = LaneEngine.__new__(LaneEngine)
    engine.drafted = engine.accepted = 0
    engine._observe_depth = lambda *args: None

    def stream():
        return SimpleNamespace(
            force=[],
            drafted=0,
            accepted=0,
            mtp_drafted=0,
            mtp_accepted=0,
            proposer=None,
            rounds=0,
            commit=lambda committed: list(committed),
            finished=False,
            cache_len=0,
            pending=[],
            think_cut=lambda committed: None,
            call_gate=None,
        )

    head, copied = stream(), stream()
    engine._conclude(head, "head", [], [11, 12, 13], [10, 11, 12], [-1, 0, 1])
    engine._conclude(copied, "copy", [], [11, 12, 13], [10, 11, 12], [-1, 0, 1])

    assert (head.mtp_drafted, head.mtp_accepted) == (2, 1)
    assert (copied.mtp_drafted, copied.mtp_accepted) == (0, 0)
