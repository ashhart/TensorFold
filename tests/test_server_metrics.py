"""Issue #110: the Mac server's GET /metrics — the scheduler's queue, its token counts and the request latencies."""

from contextlib import contextmanager
import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

from tensorfold.server.http import make_handler
from tensorfold.server.metrics import Metrics
from tensorfold.server.text import render_prompt_ids
from tests.lane_fakes import FakeEngine, fake_serial
from tests.prometheus_format import check_histograms, parse, series_value
from tests.test_lane_server import EOS, make_app

WAIT = 10
MESSAGES = [{"role": "user", "content": "hi"}]


USED_PORTS: set[int] = set()


@contextmanager
def serving(app):
    # retried while its port is one an earlier test's server used: with SO_REUSEADDR a closing server's port can still
    # accept connections for a moment, and a second server bound to it would have half its requests answered by the
    # first test's dying app. Ports stay retired for the session.
    for _ in range(8):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        if server.server_address[1] not in USED_PORTS:
            break
        server.server_close()
        time.sleep(0.02)
    else:
        raise AssertionError("every port the server got was still held by an earlier test's server")
    USED_PORTS.add(server.server_address[1])
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(WAIT)


def get(port, path="/metrics"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read().decode("utf-8")
    headers = {name.lower(): value for name, value in response.getheaders()}
    connection.close()
    return response.status, headers, body


def post(port, path, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    connection.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload

def install_fake_mlx(monkeypatch):
    """Point the engine's lazy ``import mlx.core`` at a numpy-backed stand-in so a real request can run off-Mac.

    The default chat is greedy (temperature 0), so the decode never touches the Metal sampling kernel: only
    ``array``/``concatenate``/``argmax``/``where`` and the eval no-ops, which numpy implements with the same result.
    """
    import numpy as np

    class MxArray(np.ndarray):
        def __new__(cls, value, dtype=None):
            base = value if isinstance(value, np.ndarray) else np.asarray(value, dtype=dtype)
            if dtype is not None:
                base = base.astype(dtype, copy=False)
            return base.view(cls)

    core = ModuleType("mlx.core")
    core.array = MxArray
    # numpy names a dtype by its byte count ("u4" is the 32-bit unsigned); bfloat16 has no numpy type and stores as f4
    core.uint32, core.int32, core.float16, core.bfloat16, core.float32 = "u4", "i4", "f2", "f4", "f4"
    core.eval = core.async_eval = lambda *args, **kwargs: None
    core.contiguous = lambda value: np.ascontiguousarray(value).view(MxArray)
    core.concatenate = lambda arrays, axis=0: np.concatenate([np.asarray(a) for a in arrays], axis=axis).view(MxArray)
    core.argmax = lambda a, axis=None: np.argmax(np.asarray(a), axis=axis).view(MxArray)
    core.where = lambda cond, a, b: np.where(np.asarray(cond), a, b).view(MxArray)
    core.zeros = lambda shape, dtype="f32": np.zeros(shape, dtype=dtype).view(MxArray)
    # g16 is the newest Apple GPU: the thread fitter's probe is for g14 and older, so nothing gets re-probed
    info = {"architecture": "applegpu_g16", "device_name": "off-Mac fake"}
    core.device_info = lambda: info
    metal = ModuleType("mlx.core.metal")
    metal.device_info = lambda: info
    metal.is_available = lambda: True
    metal.set_current_device = lambda *args: None
    core.metal = metal
    core.gpu = "gpu"
    core.default_device = lambda: "gpu"
    core.get_active_memory = lambda: 0
    core.get_cache_memory = lambda: 0
    core.get_peak_memory = lambda: 0
    core.reset_peak_memory = lambda: None
    core.set_cache_limit = core.set_memory_limit = core.set_wired_limit = core.synchronize = lambda *a, **k: None
    core.arange = lambda start, stop=None, **k: (np.arange(start, **k) if stop is None else np.arange(start, stop, **k)).view(MxArray)
    core.argpartition = lambda a, kth, axis=-1: np.argpartition(np.asarray(a), kth, axis=axis)
    core.array_equal = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))
    core.broadcast_to = lambda a, shape: np.broadcast_to(np.asarray(a), shape)
    core.pad = lambda a, width, mode="constant", **k: np.pad(np.asarray(a), width, mode=mode, **k)
    core.take = lambda a, indices, axis=None: np.take(np.asarray(a), np.asarray(indices), axis=axis)
    core.take_along_axis = lambda a, indices, axis: np.take_along_axis(np.asarray(a), np.asarray(indices), axis=axis)
    core.slice_update = lambda a, sl, value: (copy := np.copy(np.asarray(a)), copy.__setitem__(sl, np.asarray(value)), copy)[2]
    def logsumexp(a, axis=None):
        x = np.asarray(a, dtype=np.float64)
        m = x.max(axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(x - m), axis=axis)) + (m if axis is None else np.squeeze(m, axis))

    core.logsumexp = logsumexp
    core.bfloat, core.float, core.int, core.uint, core.bool_ = "f4", "f4", "i4", "u4", "b1"
    core.__version__ = "0.0 (numpy-backed test stand-in)"
    core.__getattr__ = lambda name: (lambda *a, **k: None)          # anything else the engine calls is a no-op
    core.fast = ModuleType("mlx.fast")
    core.fast.metal_kernel = lambda **k: (lambda **calls: [np.zeros(s, dtype=d)
                                               for s, d in zip(calls.get("output_shapes", []), calls.get("output_dtypes", []))])
    nn = ModuleType("mlx.nn")
    cache = ModuleType("mlx_lm.models.cache")

    class KVCache:
        """The base ``AlternatingKVCache`` subclasses; the fake's rows never claim this type, so it stays dormant."""

    models = ModuleType("mlx_lm.models")
    models.cache = cache
    cache.KVCache = KVCache
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.models = models
    mlx = ModuleType("mlx")
    mlx.core, mlx.nn, mlx.metal = core, nn, metal
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)


# -- the endpoint, on a real app and a real scheduler ------------------------------------------

def test_mac_metrics_serves_valid_exposition():
    app = make_app()
    with serving(app) as server:
        for path in ("/metrics", "/v1/metrics"):
            status, headers, body = get(server.server_port, path)
            assert status == 200
            assert headers["content-type"] == "text/plain; version=0.0.4"
            parsed = parse(body)
            check_histograms(parsed)
            for name in ("tensorfold:num_requests_running", "tensorfold:num_requests_waiting",
                         "tensorfold:prompt_tokens_total", "tensorfold:generation_tokens_total",
                         "tensorfold:time_to_first_token_seconds", "tensorfold:e2e_request_latency_seconds"):
                assert name in parsed, f"{name} missing from the exposition"
            # every lane engine carries the drafter counters: an engine with no active drafter reports zeros,
            # exactly as vLLM does with speculative decoding off. The KV pool stays out of the body, because on
            # the Mac each stream's cache grows from the memory budget rather than a fixed pool.
            assert "tensorfold:kv_cache_usage_perc" not in parsed
            assert series_value(parsed, "tensorfold:spec_decode_num_draft_tokens_total") == 0
            assert series_value(parsed, "tensorfold:spec_decode_num_accepted_tokens_total") == 0
        assert series_value(parse(get(server.server_port)[2]), "tensorfold:num_requests_running") == 0
        assert series_value(parse(get(server.server_port)[2]), "tensorfold:num_requests_waiting") == 0


def test_mac_counters_and_histograms_follow_a_real_request(monkeypatch, capsys):
    install_fake_mlx(monkeypatch)
    app = make_app()
    with serving(app) as server:
        status, reply = post(server.server_port, "/v1/chat/completions", {"model": "fake-27b", "messages": MESSAGES})
        assert status == 200
        usage = reply["usage"]

        # the prompt is the rendered messages; the reply is the fake target's own continuation
        prompt = render_prompt_ids(app.tokenizer, MESSAGES)
        tokens = fake_serial(prompt, app.default_max_tokens, {EOS})
        assert usage["prompt_tokens"] == len(prompt)

        metrics_body = get(server.server_port)[2]
        parsed = parse(metrics_body)
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == len(prompt)
        # the stream's emitted tokens include the EOS that ends the reply; the usage does not count it
        assert series_value(parsed, "tensorfold:generation_tokens_total") == len(tokens)
        assert series_value(parsed, "tensorfold:time_to_first_token_seconds_count") == 1
        assert series_value(parsed, "tensorfold:e2e_request_latency_seconds_count") == 1
        check_histograms(parsed)

        # the request has ended: the queue is empty again
        assert series_value(parsed, "tensorfold:num_requests_running") == 0
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 0

        # a scrape is not conversation: the live line stays clear
        capsys.readouterr()
        get(server.server_port)
        assert "GET /metrics" not in capsys.readouterr().out


def test_mac_a_request_cancelled_while_queued_counts_no_prompt_tokens(monkeypatch):
    # prompt_tokens_total is "prefill tokens processed": a job that leaves the queue before its prefill adds none,
    # while a job that is prefilled counts its whole prompt once
    install_fake_mlx(monkeypatch)
    from tensorfold.server.app import ChatJob, Scheduler

    metrics = Metrics()
    scheduler = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset({EOS}), metrics=metrics)
    left = ChatJob("left", [5, 6, 7], 4, 0)
    scheduler.submit(left)
    scheduler.cancel(left.cancellation)
    assert left.done.is_set()
    assert metrics.prompt_tokens.value == 0

    served = ChatJob("served", [1, 2], 4, 0)
    scheduler.submit(served)
    scheduler.start()
    try:
        assert served.done.wait(WAIT)
    finally:
        scheduler.stop()
    assert served.error is None
    assert metrics.prompt_tokens.value == 2


# -- what the gauges and the drafter counters report, on a fake app -----------------------------

class FakeScheduler:
    def __init__(self, metrics):
        self.metrics = metrics
        self.active = 0
        self._waiting = 0
        self.filling = None

    @property
    def waiting(self):
        return self._waiting


class FakeApp:
    """The handler's view of an app: the metrics builder, a scheduler-shaped object and the engine's totals."""

    served_name = "test"
    model_ids = ["test"]
    max_batch_size = 1

    def __init__(self, engine):
        self.scheduler = FakeScheduler(Metrics())
        self.engine = engine

    def metrics_text(self):
        running = self.scheduler.active + (1 if self.scheduler.filling is not None else 0)
        drafted = getattr(self.engine, "drafted", None)
        accepted = getattr(self.engine, "accepted", None)
        return "\n".join(self.scheduler.metrics.render(running, self.scheduler.waiting, drafted, accepted)) + "\n"


def test_mac_gauges_mirror_the_scheduler_queue():
    app = FakeApp(SimpleNamespace())
    with serving(app) as server:
        status, headers, body = get(server.server_port)
        assert status == 200
        assert headers["content-type"] == "text/plain; version=0.0.4"
        assert series_value(parse(body), "tensorfold:num_requests_running") == 0
        assert series_value(parse(body), "tensorfold:num_requests_waiting") == 0

        app.scheduler.active = 2
        app.scheduler.filling = object()
        app.scheduler._waiting = 3
        parsed = parse(get(server.server_port)[2])
        assert series_value(parsed, "tensorfold:num_requests_running") == 3     # admitted streams plus the prefill
        assert series_value(parsed, "tensorfold:num_requests_waiting") == 3
        check_histograms(parsed)


def test_mac_drafter_counters_follow_the_engine_and_stay_out_without_one():
    withdrafter = FakeApp(SimpleNamespace(drafted=7, accepted=3))
    with serving(withdrafter) as server:
        parsed = parse(get(server.server_port)[2])
        assert series_value(parsed, "tensorfold:spec_decode_num_draft_tokens_total") == 7
        assert series_value(parsed, "tensorfold:spec_decode_num_accepted_tokens_total") == 3
        check_histograms(parsed)

    plain = FakeApp(SimpleNamespace())
    with serving(plain) as server:
        parsed = parse(get(server.server_port)[2])
        assert "tensorfold:spec_decode_num_draft_tokens_total" not in parsed
        assert "tensorfold:spec_decode_num_accepted_tokens_total" not in parsed
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == 0


def test_mac_token_counters_accumulate():
    app = FakeApp(SimpleNamespace())
    with serving(app) as server:
        app.scheduler.metrics.prompt_tokens.add(10)
        app.scheduler.metrics.generation_tokens.add(3)
        parsed = parse(get(server.server_port)[2])
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == 10
        assert series_value(parsed, "tensorfold:generation_tokens_total") == 3
        app.scheduler.metrics.prompt_tokens.add(5)
        parsed = parse(get(server.server_port)[2])
        assert series_value(parsed, "tensorfold:prompt_tokens_total") == 15
