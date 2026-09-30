"""``--dashboard``: the 5-minute decode sampler, the prefill average, and the two read-only routes that serve them."""

import json
from io import BytesIO
from types import SimpleNamespace

from tensorfold.server import live
from tensorfold.server.http import make_handler
from tensorfold.server.scheduler import ChatJob
from tensorfold.server.stats import PrefillAverage, RollingRate, StatsCollector

from tests.test_lane_server import make_app


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def get(app, path):
    """One GET through the handler, the way tests/http_fakes.py posts: status, headers and body come back."""

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(f"GET {path} HTTP/1.0\r\n\r\n".encode())

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    make_handler(app)(connection, ("127.0.0.1", 0), None)   # type: ignore[arg-type]
    headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(headers.split()[1]), headers.decode(), response.decode()


def test_rolling_rate_gives_min_max_mean_and_drops_samples_past_the_window():
    clock = Clock()
    rolling = RollingRate(window=300.0, clock=clock)
    for rate in (50.0, 70.0, 30.0):
        rolling.add(rate)
        clock.now += 10.0
    now = rolling.snapshot()
    assert (now["min"], now["max"], now["mean"]) == (30.0, 70.0, 50.0)
    assert now["count"] == 3 and len(now["history"]) == 3
    clock.now = 490.0                        # every sample is past the 5-minute window
    assert rolling.snapshot()["count"] == 0 and rolling.snapshot()["min"] is None
    rolling.add(30.0)                        # a new reading starts the window over
    assert rolling.snapshot()["count"] == 1 and rolling.snapshot()["min"] == 30.0


def test_prefill_average_means_completed_chunks_and_keeps_the_newest():
    chunks = PrefillAverage(capacity=3)
    chunks.add(2048, 1.6)                    # 1280 tok/s
    chunks.add(1024, 1.0)                    # 1024 tok/s
    chunks.add(0, 1.0)                       # a chunk that prefilled nothing counts as nothing
    assert chunks.snapshot() == {"avg_tok_s": 1152.0, "last_tok_s": 1024.0, "count": 2}
    for tokens, seconds in ((512, 1.0), (256, 1.0), (128, 1.0)):
        chunks.add(tokens, seconds)
    assert chunks.snapshot()["count"] == 3    # the ring holds the last three rates only


def hold_a_stream(scheduler) -> None:
    """One fake live stream in the scheduler's bookkeeping, so the context tile has something to add up."""

    job = ChatJob(job_id="live", prompt_ids=[1] * 40, max_tokens=8, temperature=0.0)
    job.cached_tokens = 16
    job.stream = SimpleNamespace(cache_len=52, cached_tokens=16, emitted=[7])  # type: ignore[assignment]
    scheduler._jobs["live"] = job


def test_the_snapshot_carries_every_tile_and_the_schedulers_context_adds_up():
    app = make_app(context_window=1024)
    clock = Clock()
    app.scheduler.decoded = live.Meter(window=2.0, clock=clock)   # a meter a fake second cannot outrun
    collector = StatsCollector(app, clock=lambda: 123.0).start()
    try:
        app.scheduler.decoded.add(16)
        hold_a_stream(app.scheduler)
        snap = collector.sample()                               # one tick of the sampler, on the spot
        assert snap["model"] == "fake-27b" and snap["ts"] == 123.0
        assert snap["live"]["decode_tok_s"] > 0
        assert snap["decode_5m"]["min"] == snap["decode_5m"]["max"]      # the one live reading, sampled
        assert snap["context"]["window_tokens"] == app.context_window
        assert snap["context"]["used_tokens"] == 52 and snap["context"]["cached_tokens"] == 16
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()


def test_stats_answers_with_the_snapshot_and_dashboard_with_the_page_only_under_a_collector():
    app = make_app()
    collector = None
    try:
        assert json.loads(get(app, "/stats")[2])["error"]["message"]           # off unless --dashboard starts it
        assert get(app, "/dashboard")[0] == 404
        collector = StatsCollector(app).start()
        status, _, body = get(app, "/v1/stats")                                 # the /v1 prefix is tolerated
        assert status == 200 and json.loads(body)["model"] == "fake-27b"
        status, headers, body = get(app, "/dashboard")
        assert status == 200 and "<canvas id=\"gauge\"" in body and "text/html" in headers
    finally:
        if collector is not None:
            collector.stop()
            app.scheduler.stats = None
        app.close()


def test_a_serving_request_is_sampled_into_the_decode_window():
    app = make_app()
    collector = StatsCollector(app).start()
    try:
        app.chat([{"role": "user", "content": "count the tokens as they land"}], max_tokens=8)
        collector.sample()
        assert collector.snapshot()["decode_5m"]["count"] >= 1     # the metered decode reached the page
        assert collector.snapshot()["decode_5m"]["max"] > 0
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()


def test_a_recorded_prefill_chunk_lands_in_the_average():
    # the fill loop calls record_prefill with each chunk's tokens and seconds (prompt_fill._fill); the fake
    # engine exposes no prefill_tokens counter, so the call itself is covered here, not through a fake serve
    app = make_app()
    collector = StatsCollector(app).start()
    try:
        assert app.scheduler.stats is collector               # start() handed itself to the fill loop
        collector.record_prefill(2048, 1.0)
        assert collector.sample()["prefill"] == {"avg_tok_s": 2048.0, "last_tok_s": 2048.0, "count": 1}
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()


def test_the_dashboards_own_polls_stay_out_of_the_access_log(capsys):
    # one poll a second would drown the terminal in ``GET /stats 200`` lines; chat traffic still logs
    app = make_app()
    collector = StatsCollector(app).start()
    try:
        get(app, "/stats")
        get(app, "/dashboard")
        assert capsys.readouterr().out == ""
        get(app, "/v1/models")
        assert "GET /v1/models" in capsys.readouterr().out
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()


def test_a_finished_request_lands_in_the_pages_recent_list():
    app = make_app()
    collector = StatsCollector(app).start()
    try:
        app.chat([{"role": "user", "content": "count the tokens as they land"}], max_tokens=8)
        requests = collector.snapshot()["requests"]
        assert requests and requests[0]["job_id"].startswith("req-")
        row = requests[0]
        assert row["prompt_tokens"] > 0 and row["completion_tokens"] >= 8
        assert row["cache"] in {"hit", "miss", "off"}
        # the fake engine finishes short replies inside the prefill, so decode_seconds may be 0 and the
        # request's own tok/s None; the burst readings (min/max) exist either way
        if row["decode_seconds"]:
            assert row["tok_s"] > 0 and row["tok_s_min"] <= row["tok_s"] <= row["tok_s_max"]
        else:
            assert row["tok_s"] is None and row["tok_s_max"] > 0
        assert row["total_seconds"] > 0 and row["rounds"] > 0
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()


def test_the_snapshot_carries_the_live_streams_speculative_state():
    app = make_app()
    collector = StatsCollector(app).start()
    try:
        hold_a_stream(app.scheduler)
        app.scheduler._jobs["live"].stream.rounds = 3
        app.scheduler._jobs["live"].stream.drafted = 10
        app.scheduler._jobs["live"].stream.accepted = 7
        spec = collector.sample()["speculative"]
        assert (spec["rounds"], spec["drafted"], spec["accepted"]) == (3, 10, 7)
        assert spec["acceptance_rate"] == 0.7
    finally:
        collector.stop()
        app.scheduler.stats = None
        app.close()