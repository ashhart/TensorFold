"""The Mac server's live line: connections and decode/prefill tok/s on one terminal line, never inside a log line."""

import io
import sys
from types import SimpleNamespace

from tensorfold.server import live
from tensorfold.server.live import CLEAR, ChunkRate, LiveLine, Meter, snapshot, status

from tests.test_lane_server import make_app


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_decode_rate_averages_the_last_window():
    clock = Clock()
    meter = Meter(window=2.0, clock=clock)
    meter.add(100)
    clock.now = 1.5
    meter.add(50)
    assert meter.rate() == 75.0
    clock.now = 2.6                      # the first round left the window
    assert meter.rate() == 25.0


def test_prefill_rate_is_the_newest_chunks_and_fades():
    clock = Clock()
    chunks = ChunkRate(window=2.0, clock=clock)
    chunks.add(2048, 1.6)
    assert chunks.rate() == 1280.0
    clock.now = 2.5
    assert chunks.rate() == 0.0


def test_status_counts_running_prefilling_and_waiting_requests():
    clock = Clock()
    decoded, prefilled = Meter(clock=clock), ChunkRate(clock=clock)
    decoded.add(284)
    prefilled.add(2420, 2.0)
    sched = SimpleNamespace(active=2, filling=[object(), object()], waiting=1, decoded=decoded,
                            prefilled=prefilled)
    assert status(sched) == "[tensorfold] 5 connections (1 waiting) · decode 142 tok/s · prefill 1,210 tok/s"
    idle = SimpleNamespace(active=1, filling=[], waiting=0, decoded=Meter(),
                           prefilled=ChunkRate())
    assert status(idle) == "[tensorfold] 1 connection · decode 0 tok/s · prefill 0 tok/s"
    assert snapshot(sched) == {"connections": 5, "waiting": 1, "decode_tokens_per_second": 142.0,
                               "prefill_tokens_per_second": 1210.0}


def test_a_log_line_clears_the_live_line_and_a_partial_line_is_never_split():
    out, real = io.StringIO(), io.StringIO()
    line = LiveLine(lambda: "[tensorfold] 1 connection", out, every=3600)
    line.draw()
    assert out.getvalue() == CLEAR + "[tensorfold] 1 connection"
    line.write("[tensorfold] done req-1\n", real)
    assert out.getvalue().endswith(CLEAR) and real.getvalue() == "[tensorfold] done req-1\n"
    line.write("half a line", real)
    before = out.getvalue()
    line.draw()                          # mid-line: no redraw
    assert out.getvalue() == before
    line.write(" ends\n", real)
    line.draw()
    assert out.getvalue().endswith(CLEAR + "[tensorfold] 1 connection")


def test_install_routes_prints_through_the_line_and_stop_restores_the_streams(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    line = LiveLine(lambda: "status", out, every=3600).install()
    try:
        line.draw()
        print("a log line")
        assert out.getvalue() == CLEAR + "status" + CLEAR + "a log line\n"
    finally:
        line.stop()
    assert sys.stdout is out and sys.stderr is err


def test_it_stays_off_without_a_terminal_or_when_turned_off(monkeypatch):
    sched = SimpleNamespace()
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert live.start(sched) is None
    monkeypatch.setenv("TENSORFOLD_NO_LIVE", "1")
    assert live.start(sched) is None


def test_the_scheduler_meters_a_served_request():
    app = make_app(lanes=2)
    try:
        app.chat([{"role": "user", "content": "count the tokens as they land"}], max_tokens=16)
        assert app.scheduler.decoded.rate() > 0
        assert app.scheduler.engine.prefill_tokens > 0
        assert status(app.scheduler).startswith("[tensorfold] 0 connections · decode ")
    finally:
        app.close()
