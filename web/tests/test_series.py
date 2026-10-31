import time

import pytest

from tensorfold_web import bench, series
from tensorfold_web.mock_engine import serve as mock_serve


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=2000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _setup(tmp_path, mock, fail_on=None, slow_on=None):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = bench.Bench(store, lambda: (mock, None, "Qwen3.8-27B"))
    runner.power = False
    calls = []

    def switch(c, cancelled=lambda: False):
        calls.append(dict(c))
        if c.get("model") == slow_on:                           # a model that loads for a long time
            end = time.time() + 5
            while not cancelled() and time.time() < end:
                time.sleep(0.05)
            if cancelled():
                raise RuntimeError("cancelled while the engine was loading")
        if c.get("model") == "Org/Running":                     # putting the engine back takes a while
            time.sleep(0.3)
            assert s.doc["status"] == "running"                 # and the comparison is not said done before it
        if fail_on and c.get("model") == fail_on:
            raise RuntimeError("out of memory while loading")
        return {"model": c.get("model"), "context_tokens": 17408}

    before = {"model": "Org/Running", "context": None, "budget_fraction": None, "settings_model": "Org/Running"}
    s = series.Series(store.path, runner, switch, lambda: dict(before), lambda: "TensorFold 0.6.5")
    return s, store, calls


def _wait(s, timeout=60):
    end = time.time() + timeout
    while s.busy() and time.time() < end:
        time.sleep(0.05)
    assert not s.busy()


def test_configurations_are_checked():
    for bad, said in (([{"model": "a/b"}], "2 to 4"), ([{"model": "x"}, {"model": "a/b"}], "owner/name"),
                      ([{"model": "a/b", "context": 100}, {"model": "a/c"}], "1,024"),
                      ([{"model": "a/b"}, {"model": "a/b"}], "the same")):
        with pytest.raises(ValueError, match=said):
            series.validate(bad)
    assert series.label({"model": "Org/Qwen-27B", "context": 16384, "drafts": False}) == "Qwen-27B, 16k context, drafts off"


def test_a_comparison_measures_each_configuration_and_puts_the_engine_back(tmp_path, mock):
    s, store, calls = _setup(tmp_path, mock)
    doc = s.start([{"model": "Org/A"}, {"model": "Org/A", "drafts": False}], mode="profile", exactness=True)
    _wait(s)
    got = s.get(doc["id"])
    assert got["status"] == "done" and all(got["runs"]) and got["restored"]["model"] == "Org/Running"
    assert "restore_error" not in got
    assert [c["model"] for c in calls] == ["Org/A", "Org/A", "Org/Running"]
    first, second = store.run(got["runs"][0]), store.run(got["runs"][1])
    assert first["tags"] == {"series": doc["id"], "config": 0, "label": "A"} and first["exactness"] is True
    assert second["drafts"] is False and second["exactness"] is False
    m = series.matrix(got, store.run)
    assert m["rows"][0]["vs_first"] is None
    vs = m["rows"][1]["vs_first"]["speed"]
    assert vs["median_change"] == pytest.approx(-0.3) and not vs["within_noise"]      # the mock without drafts: 0.7x
    assert vs["b_slower"] == 14 and vs["p"] < 0.001                                    # 14 prompts: a sign test can tell


def test_a_configuration_that_does_not_start_is_said_and_the_next_goes_on(tmp_path, mock):
    s, _store, calls = _setup(tmp_path, mock, fail_on="Org/Huge")
    doc = s.start([{"model": "Org/Huge"}, {"model": "Org/B"}], mode="quick", exactness=False)
    _wait(s)
    got = s.get(doc["id"])
    assert got["runs"][0] is None and "out of memory" in got["errors"][0] and got["runs"][1]
    assert calls[-1]["model"] == "Org/Running"


def test_one_thing_at_a_time_and_a_cancel(tmp_path, mock):
    s, _store, calls = _setup(tmp_path, mock)
    doc = s.start([{"model": "Org/A"}, {"model": "Org/B"}, {"model": "Org/C"}], mode="profile", exactness=True)
    with pytest.raises(RuntimeError, match="is going"):
        s.start([{"model": "Org/A"}, {"model": "Org/B"}])
    assert s.gate.holder == "a comparison of configurations"
    time.sleep(0.3)
    assert s.cancel()
    _wait(s)
    got = s.get(doc["id"])
    assert got["status"] == "cancelled" and calls[-1]["model"] == "Org/Running"
    assert s.gate.holder is None                   # given back once the engine is put back
    with pytest.raises(ValueError):                # a refused start does not keep it either
        s.start([{"model": "Org/A"}], mode="nonsense")
    assert s.gate.holder is None


def test_a_comparison_can_measure_quality(tmp_path, mock):
    s, store, _calls = _setup(tmp_path, mock)
    doc = s.start([{"model": "Org/A"}, {"model": "Org/B"}], mode="quality", suites=["instructions"])
    _wait(s)
    m = series.matrix(s.get(doc["id"]), store.run)
    assert [r["items"] for r in m["rows"]] == [20, 20]
    assert m["rows"][1]["vs_first"]["quality"]["checked"] == 20


def test_two_starts_at_the_same_instant_start_one_comparison(tmp_path, mock):
    import threading

    s, _store, _calls = _setup(tmp_path, mock)
    gate, out = threading.Barrier(2), []

    def go():
        gate.wait()
        try:
            out.append(s.start([{"model": "Org/A"}, {"model": "Org/B"}], mode="quick", exactness=False)["id"])
        except RuntimeError:
            out.append("refused")

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(x == "refused" for x in out) == [False, True]
    s.cancel()
    _wait(s)


def test_the_gate_is_given_back_whatever_fails(tmp_path, mock):
    s, _store, _calls = _setup(tmp_path, mock)

    def broken():
        raise RuntimeError("cannot read the engine")
    s.current = broken                                   # fails before anything else
    s.start([{"model": "Org/A"}, {"model": "Org/B"}], mode="quick", exactness=False)
    _wait(s)
    assert s.gate.holder is None and s.get(s.doc["id"])["status"] == "failed"

    s2, _store2, calls = _setup(tmp_path / "b", mock)
    saves = {"n": 0}
    real = s2._save

    def flaky():
        saves["n"] += 1
        if s2.doc and (s2.doc.get("progress") or {}).get("phase") == "restoring":
            raise OSError("disk full")
        real()
    s2._save = flaky
    s2.start([{"model": "Org/A"}, {"model": "Org/B"}], mode="quick", exactness=False)
    _wait(s2)
    assert s2.gate.holder is None
    assert calls[-1]["model"] == "Org/Running"          # the engine was still put back


def test_a_cancel_while_the_engine_loads_stops_the_wait(tmp_path, mock):
    s, _store, calls = _setup(tmp_path, mock, slow_on="Org/A")
    doc = s.start([{"model": "Org/A"}, {"model": "Org/B"}], mode="quick", exactness=False)
    time.sleep(0.3)
    t0 = time.time()
    assert s.cancel()
    _wait(s)
    assert time.time() - t0 < 2                                 # not the five seconds the load would have taken
    got = s.get(doc["id"])
    assert got["status"] == "cancelled" and got["runs"] == [None, None] and got["errors"] == [None, None]
    assert calls[-1]["model"] == "Org/Running"                  # the engine is put back all the same
