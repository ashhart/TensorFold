"""Health metrics preserve prompt admission's in-flight workspace measurement."""

import http.client
import json
import sys
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.server.http import RequestError, make_handler
from tensorfold.server.prompt_memory import PromptMemory
from tests.test_prompt_memory import Runtime, populated


@pytest.fixture
def measured_runtime(monkeypatch):
    runtime = Runtime(resident=1000)
    cache = populated(256)
    runtime.caches.append(cache)
    core = ModuleType("mlx.core")
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory", "reset_peak_memory"):
        setattr(core, name, getattr(runtime, name))
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    model = SimpleNamespace(args=SimpleNamespace(num_attention_heads=1, head_dim=128))
    guard = PromptMemory(5300, model, runtime=runtime, overhead_bytes=0, bootstrap_bytes=0)
    return runtime, cache, guard


@contextmanager
def serving(guard=None):
    app = SimpleNamespace(served_name="test", model_ids=["test"], max_batch_size=1)
    if guard is not None:
        app.prompt_memory = guard
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def health(port, reset=True):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", "/health?reset_peak=1" if reset else "/health")
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())["memory"]
    finally:
        connection.close()


@pytest.mark.parametrize("reset", [False, True])
def test_health_during_prefill_preserves_workspace_without_waiting(measured_runtime, reset):
    runtime, cache, guard = measured_runtime
    guard.begin(256, 0)
    runtime.peak = runtime.get_active_memory() + 400
    peak = runtime.peak
    entered, release = threading.Event(), threading.Event()

    def prefill():
        entered.set()
        release.wait()

    prefill_thread = threading.Thread(target=prefill, daemon=True)
    with serving(guard) as port:
        prefill_thread.start()
        assert entered.wait(timeout=1)
        try:
            metrics = health(port, reset)
            assert prefill_thread.is_alive() and not release.is_set()
            assert metrics["peak"] == peak and runtime.peak == peak
        finally:
            release.set()
            prefill_thread.join(timeout=3)
    guard.after_chunk(cache, 256)
    assert guard.observed_work == 400
    guard.end()
    runtime.caches.clear()
    # The preserved workspace makes the next 512-token request exceed its budget.
    with pytest.raises(RequestError, match="fits up to"):
        guard.begin(512, 0)


def test_idle_health_reset_returns_peak_then_resets_it(measured_runtime):
    runtime, _, guard = measured_runtime
    runtime.peak = runtime.get_active_memory() + 400
    peak = runtime.peak
    with serving(guard) as port:
        metrics = health(port)
    assert metrics["peak"] == peak
    assert runtime.peak == runtime.get_active_memory()


def test_profiled_request_allows_reset_without_losing_learned_work(measured_runtime):
    runtime, cache, guard = measured_runtime
    guard.begin(256, 0)
    runtime.peak = runtime.get_active_memory() + 400
    guard.after_chunk(cache, 256)
    runtime.peak += 100
    peak = runtime.peak
    with serving(guard) as port:
        metrics = health(port)
    assert metrics["peak"] == peak and runtime.peak == runtime.get_active_memory()
    assert guard.workspace_profiled and guard.observed_work == 400


def test_legacy_health_app_keeps_metric_reset(measured_runtime):
    runtime, _, _ = measured_runtime
    runtime.peak = runtime.get_active_memory() + 400
    peak = runtime.peak
    with serving() as port:
        metrics = health(port)
    assert metrics["peak"] == peak and runtime.peak == runtime.get_active_memory()
