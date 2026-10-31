"""The Bench against the mock engine: a run, its exactness check, the store, the receipt and the routes."""

import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import bench, power
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import State as MockState
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=2000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _runner(store, target):
    runner = bench.Bench(store, target)
    runner.power = False                   # the machine running the tests is not what they measure
    return runner


def _wait(runner: bench.Bench, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while runner.busy() and time.time() < end:
        time.sleep(0.05)
    assert not runner.busy()


def test_suites_load_with_their_passages():
    data = bench.load_suites()
    ids = [s["id"] for s in data["suites"]]
    assert ids == ["prose_en", "code", "chat", "rag", "reasoning", "multilingual", "agent"]
    rag = next(s for s in data["suites"] if s["id"] == "rag")
    assert all("{passage}" not in m["content"] and "Harlow Ford" in m["content"]
               for p in rag["prompts"] for m in p["messages"] if m["role"] == "user")
    assert [p["id"] for _, p in bench.pick(data, "quick")] == ["sky", "csv_totals", "fr_letter"]
    assert len(bench.pick(data, "profile")) == 14 and len(bench.pick(data, "profile", ["code"])) == 2


# The digest of each published version of suites.json. Editing a prompt or a passage changes the digest: raise the
# file's "version" and add its new digest here, so runs on different prompts are never compared as if they were the same.
SUITE_DIGESTS = {1: "d691a57cc2d9"}


def test_the_suites_version_follows_their_content():
    data = bench.load_suites()
    assert SUITE_DIGESTS.get(data["version"]) == data["digest"], \
        "suites.json changed: raise its version and record the new digest in SUITE_DIGESTS"


@pytest.fixture
def hybrid():
    """The mock as a hybrid model: it reasons unless its template's off switch is sent."""

    saved = (MockState.thinks, set(MockState.refuses), set(MockState.off_by))
    MockState.thinks = True
    yield MockState
    MockState.thinks, MockState.refuses, MockState.off_by = saved


def test_thinking_is_turned_off_the_first_way_the_engine_accepts(tmp_path, mock, hybrid):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))
    runner.start("quick", exactness=False)
    _wait(runner)
    run = store.run(runner.current["id"])
    assert run["status"] == "done" and run["thinking"]["off"] is True
    assert run["thinking"]["sent"] == bench.THINKING_OFF[0][0] and run["thinking"]["refused"] == []
    assert run["suites_version"] == {"version": 1, "digest": SUITE_DIGESTS[1]}


def test_a_refused_way_is_followed_by_another_explicit_off(tmp_path, mock, hybrid):
    # the template refuses thinking_mode, and only its own "thinking" field turns reasoning off: dropping fields
    # would leave it reasoning, so the Bench must go on until a way that is both accepted and off
    hybrid.refuses, hybrid.off_by = {"thinking_mode"}, {"thinking"}
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))
    runner.start("quick", exactness=False)
    _wait(runner)
    run = store.run(runner.current["id"])
    assert run["status"] == "done"
    assert run["thinking"] == {"sent": bench.THINKING_OFF[2][0], "off": True, "refused": [bench.THINKING_OFF[0][0]]}
    assert not any("reasoned" in n for n in run["notes"])


def test_thinking_that_stays_on_is_said(tmp_path, mock, hybrid):
    hybrid.off_by = {"nothing_turns_it_off"}
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))
    runner.start("quick", exactness=False)
    _wait(runner)
    run = store.run(runner.current["id"])
    assert run["status"] == "done" and run["thinking"]["off"] is False
    assert run["thinking"]["sent"] == bench.THINKING_OFF[0][0]
    assert any("thinking stayed on" in n for n in run["notes"])
    # the reply reasons before it answers: the first answer token comes after the first token
    assert run["summary"]["overall"]["ttft_answer"] is not None


def test_percentile():
    assert bench.percentile([], 0.5) is None
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert bench.percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)


def test_a_quick_run_measures_and_checks_exactness(tmp_path, mock):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))
    run = runner.start("quick", engine_name="TensorFold (mock)")
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "done" and len(got["results"]) == 6
    on = [r for r in got["results"] if r["draft"]]
    assert all(r["m"]["timing"] == "engine" and r["m"]["sha"] and r["m"]["tokens"] > 0 for r in on)
    assert all(r["m"]["runtime"]["acceptance_rate"] == 0.75 for r in on)   # read from `speculative`, beside the block
    assert all(r["same"] is True and r["compared_by"] == "token_sha" for r in got["results"] if not r["draft"])
    overall = got["summary"]["overall"]
    assert overall["checked"] == 3 and overall["exact"] == 3 and overall["gain"] == pytest.approx(1 / 0.7 - 1)
    assert set(got["summary"]["suites"]) == {"prose_en", "code", "multilingual"}
    assert got["memory"]["samples"] >= 1 and got["memory"]["engine_peak_bytes"] > 0
    assert "hostname" not in got["machine"]
    assert store.runs()[0]["id"] == run["id"]
    r = bench.receipt(got)
    assert r["kind"] == "tensorfold-studio-bench-receipt" and r["protocol"]["temperature"] == 0.0


def test_a_run_needs_a_model_and_runs_alone(tmp_path, mock):
    store = bench.Store(tmp_path / "bench.sqlite")
    with pytest.raises(RuntimeError, match="no model"):
        _runner(store, lambda: (mock, None, None)).start("quick")
    runner = _runner(store, lambda: (mock, None, "m"))
    with pytest.raises(ValueError, match="unknown suites"):
        runner.start("profile", suites=["nope"])
    runner.start("profile")
    with pytest.raises(RuntimeError, match="already"):
        runner.start("quick")
    assert runner.cancel() and runner.current["progress"]["stopping"] is True
    _wait(runner)
    assert store.runs()[0]["status"] == "cancelled"


def test_a_failed_warm_up_is_kept_and_said(tmp_path):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: ("http://127.0.0.1:9", None, "m"))
    run = runner.start("quick")
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "failed" and got["results"] == []
    assert got["error"].startswith("The engine does not answer") and "warm-up" in got["error_detail"]
    assert got["progress"]["phase"] == "failed" and got["progress"]["eta_s"] is None


def test_a_run_left_going_by_a_stopped_studio_is_marked_interrupted(tmp_path):
    store = bench.Store(tmp_path / "bench.sqlite")
    store.save_run({"id": "x", "created": 1.0, "mode": "quick", "status": "running", "model": "m"})
    assert bench.Store(tmp_path / "bench.sqlite").runs()[0]["status"] == "interrupted"


def test_bench_routes(tmp_path, monkeypatch, mock):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    app = create_app(settings=s, engine=Engine(s), catalog=cat)
    client = TestClient(app)
    suites = client.get("/api/bench/suites").json()
    assert {x["id"] for x in suites["suites"]} >= {"code", "agent"} and "quick" in suites["modes"]
    assert client.post("/api/bench/runs", json={"mode": "fast"}).status_code == 400
    end = time.time() + 10                     # the mock says it is warming for its first seconds
    while client.get("/api/engine").json()["status"] != "running" and time.time() < end:
        time.sleep(0.1)
    run = client.post("/api/bench/runs", json={"mode": "profile", "suites": ["code"]}).json()
    assert run["engine"] == "TensorFold" and run["model"] == "Qwen3.8-27B"
    _wait(app.state.bench)
    got = client.get(f"/api/bench/runs/{run['id']}").json()
    assert got["status"] == "done" and got["summary"]["overall"]["exact"] == 2
    rec = client.get(f"/api/bench/runs/{run['id']}/receipt")
    assert "attachment" in rec.headers["content-disposition"] and rec.json()["results"]
    assert got["findings"][0].startswith("All 2 replies came out identical") and "Qwen3.8-27B" in got["share_text"]
    same = client.get(f"/api/bench/compare?a={run['id']}&b={run['id']}").json()
    assert same["speed"]["median_change"] == 0 and same["exactness"]["checked"] == 2
    assert client.get(f"/api/bench/compare?a={run['id']}&b=nope").status_code == 404
    assert "attachment" in client.get(f"/api/bench/compare/report?a={run['id']}&b={run['id']}").headers["content-disposition"]
    page = client.get(f"/api/bench/runs/{run['id']}/report")
    assert page.headers["content-type"].startswith("text/html") and "attachment" in page.headers["content-disposition"]
    assert client.delete(f"/api/bench/runs/{run['id']}").status_code == 200
    assert client.get(f"/api/bench/runs/{run['id']}").status_code == 404


def test_a_prefill_sweep_stays_below_the_window_and_misses_the_cache(tmp_path, mock):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))
    run = runner.start("prefill")
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "done" and got["window"] == 32768 and not got["exactness"]
    sweep = got["summary"]["sweep"]
    assert [x["size"] for x in sweep] == [1024, 2048, 4096, 8192, 16384]        # 32768 needs room for the reply
    assert all(x["cached"] == 0 and x["prefill_tps"] > 0 for x in sweep)
    assert all(abs(x["prompt_tokens"] - x["size"]) < x["size"] * 0.1 for x in sweep)
    assert any("window" in n for n in got["notes"])
    small = runner.start("prefill", max_context=4096)
    _wait(runner)
    assert [x["size"] for x in store.run(small["id"])["summary"]["sweep"]] == [1024, 2048, 4096]


class FakePower(power.Source):
    name, scope, whole_machine = "fake", "whole machine", True

    def __init__(self):
        self.watts = 5.0

    def read(self):
        return power.Reading(watts=self.watts, cpu_c=50.0, gpu_c=60.0)


def test_energy_counts_the_measured_part_above_idle():
    src = FakePower()
    s = bench.Sampler(lambda: 1000, source=src)
    s.series = [[0.0, 5.0, 40.0, 41.0, 1000], [1.0, 5.0, 40.0, 41.0, 1000],    # idle, before measuring
                [2.0, 25.0, 70.0, 80.0, 2000], [4.0, 25.0, 72.0, 85.0, 2000], [6.0, 25.0, 71.0, 83.0, 2000]]
    s.idle_watts, s.measured_from = 5.0, 2.0
    r = s.report(output_tokens=200)
    assert r["energy_j"] == 100.0 and r["avg_watts"] == 25.0 and r["energy_above_idle_j"] == 80.0
    assert r["j_per_token"] == 0.5 and r["j_per_token_above_idle"] == 0.4
    assert r["peak_gpu_c"] == 85.0 and r["peak_cpu_c"] == 72.0 and r["estimate"] is False
    assert bench.Sampler(lambda: None).report(10) is None                       # no source, no figures


def test_a_run_with_a_power_source_reports_energy(tmp_path, mock, monkeypatch):
    monkeypatch.setattr(power, "detect", lambda: (FakePower(), "fake"))
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = bench.Bench(store, lambda: (mock, None, "m"))
    run = runner.start("quick", exactness=False)
    _wait(runner)
    hw = store.run(run["id"])["hardware"]
    assert hw["source"] == "fake" and hw["idle_watts"] == 5.0 and hw["output_tokens"] > 0
    assert hw["series"] and hw["peak_gpu_c"] == 60.0


def test_power_gaps_are_left_out_and_said():
    s = bench.Sampler(lambda: None, source=FakePower())
    s.series = [[0.0, 20.0, None, None, None, False], [2.0, 20.0, None, None, None, False],
                [17.0, 20.0, None, None, None, True], [19.0, 20.0, None, None, None, True]]     # 15 s without a reading
    s.measured_from = 0.0
    s.reasons = {"thermal (hardware)"}
    r = s.report(output_tokens=100)
    assert r["measured_s"] == 4.0 and r["window_s"] == 19.0 and r["energy_j"] == 80.0
    assert r["coverage"] == pytest.approx(4 / 19)
    assert r["throttle"] == {"reported": True, "throttled": True, "seconds": 2.0, "reasons": ["thermal (hardware)"]}


def test_energy_cost_at_the_price_set():
    hw = {"energy_j": 3.6e6, "j_per_token": 0.36, "j_per_token_above_idle": 0.18}
    c = bench.energy_cost(hw, 0.25, "EUR")
    assert c["run"] == pytest.approx(0.25) and c["per_million_tokens"] == pytest.approx(0.025)
    assert c["per_million_tokens_above_idle"] == pytest.approx(0.0125) and c["currency"] == "EUR"
    assert bench.energy_cost(hw, None, "EUR") is None


def test_nvidia_throttle_reasons_with_either_field_name(monkeypatch):
    class Done:
        def __init__(self, out, err=""):
            self.stdout, self.stderr = out, err

    asked = []

    def run(cmd, **_):
        asked.append(cmd[1])
        if "clocks_event_reasons" in cmd[1]:                  # an older driver: the field is unknown, and says so
            return Done("", 'Field "clocks_event_reasons.active" is not a valid field to query.')
        return Done("30.5, 61, 0x0000000000000024\n")         # power cap and software thermal

    monkeypatch.setattr(power.subprocess, "run", run)
    src = power.NvidiaSmi()
    r = src.read()
    assert r.watts == 30.5 and r.gpu_c == 61.0 and r.throttled is True
    assert r.reasons == ("power cap", "thermal (software)")
    src.read()
    assert sum("clocks_event_reasons" in a for a in asked) == 1      # the unknown field is asked once


def test_a_run_costs_its_energy_at_the_price_set(tmp_path, monkeypatch):
    monkeypatch.setattr(power, "detect", lambda: (FakePower(), "fake"))
    server = mock_serve(port=0, rate_tps=40.0)            # slow enough for a few power readings while measuring
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = bench.Bench(store, lambda: (f"http://127.0.0.1:{server.server_address[1]}", None, "m"))
    runner.price = lambda: (0.30, "EUR")
    run = runner.start("quick", exactness=False)
    try:
        _wait(runner, timeout=60)
    finally:
        server.shutdown()
    hw = store.run(run["id"])["hardware"]
    assert hw["cost"]["price_per_kwh"] == 0.30 and hw["cost"]["run"] > 0
    assert hw["throttle"]["reported"] is False and hw["coverage"] == pytest.approx(1.0, abs=0.05)


def test_energy_price_and_currency_are_checked():
    assert settings_mod.validate({"energy_price": 0.2516, "currency": " usd "}) == {"energy_price": 0.2516, "currency": "USD"}
    for bad in ({"energy_price": 0}, {"energy_price": True}, {"energy_price": 50}, {"currency": "euro"}):
        with pytest.raises(ValueError):
            settings_mod.validate(bad)


def test_two_starts_at_once_make_one_run(tmp_path, mock):
    import threading as th

    runner = _runner(bench.Store(tmp_path / "bench.sqlite"), lambda: (mock, None, "m"))
    outcomes: list[str] = []

    def go():
        try:
            runner.start("quick", exactness=False)
            outcomes.append("started")
        except RuntimeError:
            outcomes.append("refused")

    threads = [th.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _wait(runner)
    assert sorted(outcomes) == ["refused"] * 3 + ["started"]


def test_a_run_says_its_phase_and_time_left(tmp_path, mock):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "m"))
    run = runner.start("quick")
    _wait(runner)
    p = store.run(run["id"])["progress"]
    assert p["phase"] == "done" and p["measuring_since"] >= run["created"] and p["eta_s"] is None


def test_time_left_in_a_sweep_grows_with_the_sizes_and_stays_within_the_limit():
    assert bench.sweep_eta(10.0, 2, 0.0) == pytest.approx(10 * 2.2 + 10 * 2.2 ** 2)
    assert bench.sweep_eta(100.0, 5, 800.0) == pytest.approx(bench.SWEEP_LIMIT_S - 800.0)
    assert bench.sweep_eta(10.0, 0, 50.0) == 0.0


def test_errors_are_said_plainly():
    assert bench.plain_error("ConnectError: [Errno 111] Connection refused").startswith("The engine does not answer")
    assert "ten minutes" in bench.plain_error("ReadTimeout: timed out")
    assert bench.plain_error("something else") == "something else"


def test_time_left_in_a_suite_run_times_replies_with_and_without_drafts_apart():
    work = [("s", "a"), ("s", "b"), ("s", "c")]
    done = [{"draft": True, "m": {"wall_s": 4.0}}, {"draft": False, "m": {"wall_s": 12.0}}]
    assert bench.suites_eta(done, work, exactness=True) == pytest.approx(2 * 4.0 + 2 * 12.0)
    assert bench.suites_eta(done[:1], work, exactness=True) == pytest.approx(2 * 4.0 + 3 * 5.6)
    assert bench.suites_eta(done[:1], work, exactness=False) == pytest.approx(2 * 4.0)


@pytest.fixture
def other_engine():
    """The mock as another engine: replies without TensorFold's block, so no prefill time and no token digest."""

    MockState.plain = True
    yield MockState
    MockState.plain = False


def test_another_engine_is_measured_without_the_drafts_off_check(tmp_path, mock, other_engine):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (None, None, None))                 # nothing served here: only the target
    run = runner.start("quick", engine_name="Ollama 0.35.1", target=(mock, None, "Qwen3.8-27B"))
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "done" and got["exactness"] is False and len(got["results"]) == 3
    assert got["target"] == {"kind": "address", "url": mock} and got["engine"] == "Ollama 0.35.1"
    assert any("drafts-off check" in n for n in got["notes"])
    assert all(r["m"]["timing"] == "studio" for r in got["results"])


def test_a_sweep_on_an_engine_without_prefill_time_estimates_it(tmp_path, mock, other_engine):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (None, None, None))
    run = runner.start("prefill", engine_name="llama.cpp", target=(mock, None, "Qwen3.8-27B"), max_context=4096)
    _wait(runner, timeout=60)
    got = store.run(run["id"])
    sweep = got["summary"]["sweep"]
    assert got["status"] == "done" and [p["size"] for p in sweep] == [1024, 2048, 4096]
    assert all(p["estimated"] and p["prefill_tps"] for p in sweep)
    assert got["prefill_overhead_s"] >= 0 and any("no prefill time" in n for n in got["notes"])
    assert all(r["m"]["repeats"] == 3 for r in got["results"])


def test_a_load_sweep_sends_waves_and_says_the_total_and_each_requests_speed(tmp_path):
    from tensorfold_web import report

    server = mock_serve(port=0, rate_tps=400.0)        # each request at its own pace: the total grows with the waves
    try:
        store = bench.Store(tmp_path / "bench.sqlite")
        runner = _runner(store, lambda: (f"http://127.0.0.1:{server.server_address[1]}", None, "Qwen3.8-27B"))
        run = runner.start("load", max_concurrency=4)
        _wait(runner, timeout=90)
    finally:
        server.shutdown()
    got = store.run(run["id"])
    levels = got["summary"]["load"]
    assert got["status"] == "done" and got["exactness"] is False and got["max_concurrency"] == 4
    assert [lv["concurrency"] for lv in levels] == [1, 2, 4] and all(lv["requests"] == 8 for lv in levels)
    assert levels[2]["total_tps"] > 2.5 * levels[0]["total_tps"]
    assert levels[0]["per_request_tps"] == pytest.approx(levels[2]["per_request_tps"], rel=0.2)
    found = report.findings(got)
    assert found[0].startswith("Serving several at once raises the total from")
    assert "Requests at once" in report.html_report(got)
    with pytest.raises(ValueError):
        bench.Bench(store, lambda: (None, None, "m")).start("burst")


@pytest.mark.parametrize("events,error", [
    (['{"choices":[{"delta":{"content":"Hi"}}]}', '{"choices":[{"delta":{},"finish_reason":"stop"}]}', "[DONE]"], None),
    (['{"choices":[{"delta":{"content":"Hi"}}]}', '{"error":{"message":"CUDA error: out of memory"}}'], "out of memory"),
    (['{"choices":[{"delta":{"content":"Hi"}}]}'], "ended before"),                     # cut: no finish, no [DONE]
])
def test_stream_errors_and_cut_streams_are_not_successes(events, error):
    import threading

    import httpx

    body = "".join(f"data: {e}\n\n" for e in events)
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, text=body)))
    rep = bench.stream_once(client, "http://engine/v1/chat/completions", {}, {}, threading.Event())
    assert rep.text == "Hi"
    assert (rep.error is None) if error is None else (error in rep.error)


def test_an_engine_elsewhere_is_judged_by_its_address():
    import socket

    assert bench.on_this_machine("127.0.0.1") and bench.on_this_machine("localhost")
    assert bench.on_this_machine(socket.gethostname()) in (True, False)      # resolves or not; never raises
    assert not bench.on_this_machine("192.0.2.10")                           # TEST-NET-1: never this machine's
    assert not bench.on_this_machine("no-such-host.invalid")


def test_a_served_engine_that_is_not_tensorfold_gets_no_vacuous_exactness(tmp_path, mock, other_engine):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = _runner(store, lambda: (mock, None, "Qwen3.8-27B"))         # served here, but Settings name Ollama
    run = runner.start("quick", engine_name="Ollama 0.35.1")
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "done" and got["exactness"] is False
    assert not any(r.get("same") for r in got["results"]) and got["summary"]["overall"]["checked"] == 0


def test_a_power_source_that_cannot_open_does_not_leave_a_run_running(tmp_path, mock, monkeypatch):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = bench.Bench(store, lambda: (mock, None, "Qwen3.8-27B"))
    runner.power = True
    monkeypatch.setattr(power, "detect", lambda: (_ for _ in ()).throw(AttributeError("no SMC symbol")))
    run = runner.start("quick", engine_name="TensorFold")
    _wait(runner)
    got = store.run(run["id"])
    assert got["status"] == "done" and got["finished"] and not runner.busy()
    assert any("power source failed to open" in n for n in got["notes"])


def test_a_cancel_ends_a_read_the_engine_never_answers(tmp_path):
    import socket
    import threading

    import httpx

    from tensorfold_web import kl

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def hold():                                                          # accepts and never answers, like a stall
        held = []
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
                held.append(c)
            except TimeoutError:
                continue
        for c in held:
            c.close()
    t = threading.Thread(target=hold, daemon=True)
    t.start()
    try:
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            rep = bench.stream_once(client, f"http://127.0.0.1:{port}/v1/chat/completions", {}, {"x": 1}, cancel)
        assert rep.error == "cancelled" and time.time() - t0 < 5
        cancel2 = threading.Event()
        threading.Timer(0.3, cancel2.set).start()
        t0 = time.time()
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client, pytest.raises(InterruptedError):
            kl.ask(client, kl.Target(f"http://127.0.0.1:{port}/v1/chat/completions", None, "m", {}),
                   [{"role": "user", "content": "hi"}], 5, cancel=cancel2)
        assert time.time() - t0 < 5
    finally:
        stop.set()
        t.join(2)
        srv.close()


def test_a_pair_the_engine_could_not_confirm_is_not_a_checked_one():
    m = {"tps": 10.0, "tokens": 5, "ttft": 0.1, "ttft_answer": 0.1, "prefill_s": 0.05, "gap_p50": 0.01, "gap_p95": 0.02,
         "runtime": {}}
    rows = [{"suite": "s", "prompt": "a", "draft": True, "m": m}, {"suite": "s", "prompt": "a", "draft": False, "m": m, "same": True},
            {"suite": "s", "prompt": "b", "draft": True, "m": m},
            {"suite": "s", "prompt": "b", "draft": False, "m": m, "same": None, "compared_by": "unverified"}]
    block = bench.summarize(rows)["overall"]
    assert (block["checked"], block["exact"]) == (1, 1)         # not "1 of 2 replies differed"
