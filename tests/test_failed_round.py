"""Failed rounds report errors to every client and release each stream."""

import http.client
import gc
import json
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_engine import LaneEngine, LaneStream
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.http_fakes import post
from tests.test_server_openai_compat import FakeApp


from tests.test_lane_server import SlowFakeEngine, make_app
from tests.test_server_openai_compat import serve_fake


def stream(server, text, out):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    conn.request("POST", "/v1/chat/completions",
                 body=json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 60,
                                  "stream": True}).encode(),
                 headers={"Content-Type": "application/json"})
    out[text] = conn.getresponse().read().decode()
    conn.close()


def test_a_failed_round_is_reported_as_an_error_and_releases_the_streams():
    app = make_app(lanes=2, engine_factory=SlowFakeEngine)
    step = app.engine.step

    def flaky():
        if app.engine.active_count == 2:
            raise RuntimeError("GPU fault in a shared round")
        return step()

    app.engine.step = flaky
    server = serve_fake(app)
    try:
        out = {}
        threads = [threading.Thread(target=stream, args=(server, text, out)) for text in ("first one", "second one")]
        for thread in threads:
            thread.start()
            time.sleep(0.01)
        for thread in threads:
            thread.join(timeout=30)
        assert all(not thread.is_alive() for thread in threads)
        assert len(out) == 2
        for body in out.values():
            events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
            assert events[-1]["error"]["message"] == "GPU fault in a shared round"
            assert all(choice["finish_reason"] is None for event in events for choice in event.get("choices", []))
            assert body.count("data: [DONE]") == 1
        assert not app.engine.streams, f"{len(app.engine.streams)} failed streams still held by the engine"
    finally:
        server.shutdown()
        server.server_close()
        app.close()




@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/completions"])
def test_handler_emits_an_error_event_without_a_success_finish(route):
    status, body = post(FakeApp(fail_stream=True), {"messages": [{"role": "user", "content": "hi"}],
                                                  "prompt": "hi", "stream": True}, route)
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    assert status == 200
    assert events[-1].get("error") == {"message": "boom", "type": "server_error"}
    assert all(c["finish_reason"] is None for e in events for c in e.get("choices", []))
    assert body.count("data: [DONE]") == 1


def test_scheduler_discards_each_failed_stream_before_reset():
    engine = LaneEngine(SimpleNamespace(lane_family=True))
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset())
    jobs = [ChatJob(str(i), [1], 10, 0.0) for i in range(2)]
    for job in jobs:
        job.stream = LaneStream(job.job_id, job.prompt_ids, job.max_tokens)
        engine.streams.append(job.stream)
        engine._live.append((job.stream, []))
        scheduler._jobs[job.job_id] = job
    discarded = []
    discard = engine.discard_stream
    fault = RuntimeError("failed round")

    def discard_stream(stream):
        discarded.append(stream.stream_id)
        discard(stream)

    def step():
        scheduler._stop.set()
        raise fault

    engine.discard_stream, engine.step = discard_stream, step
    scheduler._loop()
    assert discarded == [job.job_id for job in jobs]
    assert not engine.streams and not scheduler._jobs and not engine.active_count
    for job in jobs:
        assert str(job.error) == str(fault) and job.done.is_set()
        assert job.chunks.get_nowait() is None
        assert job.stream.finished and job.stream.finish_reason == "error"


@pytest.mark.parametrize("chained", [False, True])
def test_failed_round_caches_are_collectable_while_scheduler_stays_idle(chained, capsys):
    class Cache:
        pass

    engine = LaneEngine(SimpleNamespace(lane_family=True, exact_width=2, streams_exact=True,
                                       hidden_rows=lambda *args: None))
    scheduler = Scheduler(engine, lanes=2, eos_ids=frozenset())
    jobs, refs = [], []

    def add_job(index):
        job = ChatJob(str(index), [1], 10, 0.0)
        job.stream = LaneStream(job.job_id, [1], 10)
        cache = Cache()
        refs.append(weakref.ref(cache))
        engine.streams.append(job.stream)
        engine._live.append((job.stream, [cache]))
        scheduler._jobs[job.job_id] = job
        jobs.append(job)

    for index in range(2):
        add_job(index)

    def fail(entries):
        plans = [cache for _, cache in entries]
        try:
            raise ValueError("draft allocation failed")
        except ValueError as cause:
            error = RuntimeError("shared round failed")
            error.cache = plans[0]
            if chained:
                raise error from cause
            raise error

    idle = threading.Event()
    get = scheduler._queue.get

    def idle_get(*args, **kwargs):
        idle.set()
        return get(*args, **kwargs)

    engine._family_round_streams = fail
    scheduler._queue.get = idle_get
    scheduler._thread.start()
    try:
        assert all(job.done.wait(2) for job in jobs)
        assert idle.wait(2) and scheduler._thread.is_alive()
        gc.collect()
        assert all(ref() is None for ref in refs)
        for job in jobs:
            assert str(job.error) == "shared round failed"
            assert job.error.error_type == "RuntimeError"
            assert job.error.__traceback__ is job.error.__cause__ is job.error.__context__ is None
            assert not hasattr(job.error, "cache")
            assert job.chunks.get_nowait() is None
        assert jobs[0].error is not jobs[1].error
        report = capsys.readouterr().err
        assert "draft allocation failed" in report and "shared round failed" in report
    finally:
        scheduler.stop()
