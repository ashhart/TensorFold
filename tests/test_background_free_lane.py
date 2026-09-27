"""Background work keeps its progress while a foreground request can use a free lane."""

import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_engine import LaneEngine, LaneStream
from tensorfold.server.scheduler import ChatJob, Scheduler


from tests.test_lane_server import SlowFakeEngine, expected_reply, make_app


def test_background_job_keeps_running_when_a_lane_is_free():
    app = make_app(lanes=2, engine_factory=SlowFakeEngine)
    try:
        batch = [{"role": "user", "content": "a long batch generation for training data"}]
        turn = [{"role": "user", "content": "the user's own question"}]
        results = {}
        deltas = []

        def batch_run():
            results["batch"] = app.chat(batch, max_tokens=60, on_delta=deltas.append,
                                        sampling={"priority": "background"})

        worker = threading.Thread(target=batch_run)
        worker.start()
        deadline = time.perf_counter() + 5
        while not deltas and time.perf_counter() < deadline:
            time.sleep(0.002)
        assert deltas
        results["turn"] = app.chat(turn, max_tokens=12)
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert results["batch"]["content"] == expected_reply(app, batch, 60)[1]
        assert app.scheduler.preemptions == 0, f"restarted {app.scheduler.preemptions}x with a free lane"
    finally:
        app.close()




def _scheduler(lanes, backgrounds):
    engine = LaneEngine(SimpleNamespace(lane_family=True))
    scheduler = Scheduler(engine, lanes=lanes, eos_ids=frozenset())
    for i in range(backgrounds):
        job = ChatJob(str(i), [1], 10, 0.0, background=True)
        job.stream = LaneStream(job.job_id, [1], 10)
        scheduler._jobs[job.job_id] = job
        engine.streams.append(job.stream)
        engine._live.append((job.stream, []))
    return scheduler


@pytest.mark.parametrize("held", [False, True])
def test_a_foreground_job_does_not_preempt_a_free_lane(held):
    scheduler = _scheduler(2, 1)
    foreground = ChatJob("foreground", [2], 10, 0.0)
    if held:
        scheduler._held = foreground
    else:
        scheduler.submit(foreground)
    scheduler._preempt_background()
    assert scheduler.preemptions == 0
    assert scheduler.engine.active_count == 1
    assert not scheduler._jobs["0"].preempted


@pytest.mark.parametrize("lanes", [1, 2])
def test_a_full_engine_frees_only_one_lane_for_a_waiting_job(lanes):
    scheduler = _scheduler(lanes, lanes)
    scheduler.submit(ChatJob("foreground", [2], 10, 0.0))
    scheduler._preempt_background()
    assert scheduler.preemptions == 1
    assert scheduler.engine.active_count == lanes - 1
    assert scheduler._jobs["0"].stream.finish_reason == "preempted"
    scheduler._preempt_background()
    assert scheduler.preemptions == 1


def test_background_work_does_not_preempt_other_background_work():
    scheduler = _scheduler(1, 1)
    scheduler.submit(ChatJob("waiting", [2], 10, 0.0, background=True))
    scheduler._preempt_background()
    assert scheduler.preemptions == 0


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("admission", [False, True])
@pytest.mark.parametrize("lanes,backgrounds,needed", [(2, 1, 1), (3, 2, 1), (2, 2, 2), (4, 3, 2)])
def test_memory_pressure_preempts_only_enough_background_work(lanes, backgrounds, needed, admission, held):
    class Cache:
        pass

    scheduler = _scheduler(lanes, backgrounds)
    refs = []

    def with_cache(entry):
        stream, _ = entry
        cache = Cache()
        refs.append(weakref.ref(cache))
        return stream, [cache]

    scheduler.engine._live = [with_cache(entry) for entry in scheduler.engine._live]
    remaining = backgrounds - needed
    if admission:
        scheduler.admission = SimpleNamespace(admits=lambda prompt, total, live: len(live) <= remaining)
    else:
        scheduler.prompt_memory = SimpleNamespace(
            would_fit=lambda prompt, reply: sum(ref() is not None for ref in refs) <= remaining)
    foreground = ChatJob("foreground", [2], 10, 0.0)
    if held:
        scheduler._held = foreground
    else:
        scheduler.submit(foreground)
    scheduler._preempt_background()
    assert scheduler.preemptions == needed
    assert scheduler.engine.active_count == remaining
    assert sum(ref() is not None for ref in refs) == remaining
    assert all(scheduler._jobs[str(i)].stream.finish_reason == "preempted" for i in range(needed))
    admitted = []
    scheduler._start_job = admitted.append
    scheduler._admit()
    assert admitted == [foreground] and scheduler._held is None


def test_waiting_foreground_bypasses_held_background_and_preserves_running_foreground():
    scheduler = _scheduler(3, 2)
    scheduler._jobs["0"].background = False
    scheduler.prompt_memory = SimpleNamespace(would_fit=lambda *args: len(scheduler.engine._live) < 2)
    background = ChatJob("held", [3], 10, 0.0, background=True)
    foreground = ChatJob("foreground", [2], 10, 0.0)
    scheduler._held = background
    scheduler.submit(foreground)
    scheduler._preempt_background()
    assert scheduler.preemptions == 1
    assert not scheduler._jobs["0"].stream.finished
    assert scheduler._jobs["1"].stream.finish_reason == "preempted"
    assert scheduler._held is background and scheduler._queue.get_nowait() is foreground
