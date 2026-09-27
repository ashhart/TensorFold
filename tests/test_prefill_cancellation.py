import socket
import sys
from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.app import ChatJob, Scheduler
from tensorfold.server.cancellation import Cancellation, PrefillGuard, RequestCancelled, socket_cancellation
from tests.lane_fakes import FakeEngine, fake_serial


def test_cancellation_after_evaluated_first_chunk_prevents_next_model_chunk(monkeypatch):
    engine = FakeEngine()
    engine.prefill_step = 4
    token = Cancellation()
    engine.prefill_guard = PrefillGuard(token)
    calls, evaluated = [], []
    monkeypatch.setattr(sys.modules["mlx.core"], "eval", lambda *args: evaluated.append(True))
    forward = engine.model.hidden

    def canceled_forward(rows, cache, parents=None):
        calls.append(int(rows.size))
        token.cancel()
        return forward(rows, cache, parents)

    engine.model.hidden = canceled_forward
    cache = engine.model.make_cache()
    with pytest.raises(RequestCancelled):
        engine._family_feed(list(range(12)), cache, engine.prompt_chunks(range(12)).between(0, 12))
    assert calls == [4] and cache[0].rows[0] == [0, 1, 2, 3]
    assert evaluated == [True]


def test_guard_keeps_memory_admission_and_checkpoint_policy():
    calls = []
    memory = SimpleNamespace(before_chunk=lambda *args: calls.append("before"),
                             after_chunk=lambda *args: calls.append("after"),
                             allow_checkpoint=lambda *args: False)
    guard = PrefillGuard(Cancellation(), memory)
    guard.before_chunk([], 4)
    guard.after_chunk([], 4)
    assert calls == ["before", "after"] and not guard.allow_checkpoint([])
    guard.cancellation.cancel()
    with pytest.raises(RequestCancelled):
        guard.before_chunk([], 4)
    assert calls == ["before", "after"]


def test_canceling_held_job_wakes_waiter_without_prefill_or_harming_active_job():
    engine = FakeEngine()
    full = SimpleNamespace(admits=lambda *args: False)      # memory for one stream only: the second is held
    scheduler = Scheduler(engine, lanes=2, admission=full, eos_ids=frozenset({-1}))
    first, held = ChatJob("first", [1, 2], 8, 0), ChatJob("held", [3, 4], 8, 0)
    scheduler.submit(first)
    scheduler.submit(held)
    scheduler._admit()
    assert scheduler._held is held and len(engine.prefill_calls) == 1
    scheduler.cancel(held.cancellation)
    scheduler._cancel_active()
    assert held.done.is_set() and held.chunks.get_nowait() is None
    assert scheduler._held is None and first.error is None
    assert len(engine.prefill_calls) == 1


@pytest.mark.parametrize("joined", [False, True])
def test_engine_discards_only_canceled_row_and_preserves_survivor_tokens(joined):
    engine = FakeEngine()
    canceled = LaneStream("canceled", [1, 2], 8, eos_ids=frozenset({-1}))
    survivor = LaneStream("survivor", [3, 4], 8, eos_ids=frozenset({-1}))
    engine.add_stream(canceled)
    engine.add_stream(survivor)
    if joined:
        engine.step()
    engine.discard_stream(canceled)
    assert engine.active_count == 1 and canceled not in engine.streams
    while not survivor.finished:
        engine.step()
    assert survivor.emitted == fake_serial([3, 4], 8, {-1})
    assert all(s is not canceled for s, cache in engine._live)


def test_family_discard_releases_owned_rows_and_draft_state_only():
    engine = FakeEngine()
    canceled, survivor = LaneStream("canceled", [1], 8), LaneStream("survivor", [2], 8)
    caches = [object(), object()]
    engine.streams = [canceled, survivor]
    engine._live = [(canceled, caches[0]), (survivor, caches[1])]
    tables = (engine._inflight, engine._next, engine._mode, engine._depth_state, engine._served, engine._granted)
    for mapping in tables:
        mapping.update(canceled=object(), survivor=object())
    engine.discard_stream(canceled)
    assert engine._live == [(survivor, caches[1])]
    for mapping in tables:
        assert set(mapping) == {"survivor"}


def test_socket_probe_does_not_consume_readable_bytes_or_cancel_a_live_peer():
    left, right = socket.socketpair()
    try:
        token = socket_cancellation(left)
        assert not token.cancelled
        right.sendall(b"x")
        assert not token.cancelled
        assert left.recv(1) == b"x"
        right.close()
        with pytest.raises(RequestCancelled):
            token.check()
    finally:
        left.close()
        right.close()
