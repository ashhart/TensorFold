"""A finished stream leaves per-stream tables behind (a long-running server gains an entry a request)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from test_family_streams import SPECS, StreamsModel, _stream  # noqa: E402

from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402


def test_finished_streams_leave_no_per_stream_state():
    engine = LaneEngine(StreamsModel(gpu_tokens=True))
    for i, spec in enumerate(SPECS * 4):
        stream = _stream(i, spec)
        engine.add_stream(stream)
        while engine.active_count:
            engine.step()
    tables = {"_depth_state": engine._depth_state, "_mode": engine._mode, "_next": engine._next,
              "_inflight": engine._inflight, "_served": engine._served, "_granted": engine._granted}
    assert {name: len(t) for name, t in tables.items() if t} == {}


def _tables(engine):
    return [getattr(engine, name) for name in ("_depth_state", "_mode", "_next", "_inflight", "_served", "_granted")]


@pytest.mark.parametrize("head", [False, True])
def test_prefill_finish_releases_state(head):
    engine = LaneEngine(StreamsModel(gpu_tokens=True, head=head))
    stream = _stream(0, ([8, 2], 1, 1, True))
    engine.add_stream(stream)
    assert stream.finished
    assert not any(_tables(engine))


def test_reset_releases_failed_streams_and_can_accept_new_work():
    engine = LaneEngine(StreamsModel(gpu_tokens=True))
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS[:2])]
    for stream in streams:
        engine.add_stream(stream)
    engine.step()
    assert engine._depth_state
    engine.reset()
    assert engine.streams == []
    assert engine.active_count == 0
    assert not any(_tables(engine))
    stream = _stream(0, SPECS[1])
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert stream.finished
    assert not any(_tables(engine))


def test_shared_pipelined_finish_releases_modes():
    engine = LaneEngine(StreamsModel(gpu_tokens=True, head=False))
    for i, spec in enumerate(SPECS[:2]):
        engine.add_stream(_stream(i, spec))
    engine.step()
    assert engine._mode
    while engine.active_count:
        engine.step()
    assert not any(_tables(engine))
