"""A Fleet machine as the Bench's target or KL reference, resolved by this TensorFold Web; and KL on a fixed text."""

import json
import math
import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import bench, engines, fleet, kl
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import MODEL, State
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app

TOKEN = "s3cret-peer-token"
TEXT = ("The lighthouse stood at the end of a narrow spit of rock, and for forty years the same family had kept its "
        "lamp burning. Every evening the keeper climbed the hundred and twelve steps, trimmed the wick, wound the clock "
        "that turned the lens, and wrote the weather in a book. Ships passing in the night never saw him, only the "
        "light, sweeping once every ten seconds across the water. When the lamp was finally made automatic, the last "
        "keeper left the book on the table, open at a page that read: wind from the west, sea calm, all well.")

# TensorFold 1.0.9 on a DGX Spark, `prompt_logprobs: 2` on "The cat sat": the shape `kl._scored` reads
REAL = {"choices": [{"index": 0, "text": " on", "prompt_logprobs": [
    None,
    {"7990": {"logprob": -11.6, "rank": 14013, "decoded_token": " cat"},
     "1044": {"logprob": -3.2, "rank": 1, "decoded_token": ","}, "1039": {"logprob": -3.7, "rank": 2, "decoded_token": "'"}},
    {"1408": {"logprob": -0.05, "rank": 1, "decoded_token": " sat"},
     "2613": {"logprob": -5.1, "rank": 2, "decoded_token": ".\""}}]}],
    "tensorfold": {"drafts": False}}


def T(ident: str, **top: float) -> kl.Step:
    return kl.Step(ident.encode(), {k.encode(): v for k, v in top.items()}, ident, {k.encode(): k for k in top})


def test_the_engines_answer_is_read_position_by_position():
    steps = kl._scored(REAL)
    assert [s.ident for s in steps] == ["7990", "1408"] and steps[0].token == b" cat"
    assert steps[0].top[b"1044"] == pytest.approx(math.exp(-3.2)) and len(steps[0].top) == 3
    assert steps[1].said[b"2613"] == '."'
    with pytest.raises(RuntimeError, match="no prompt_logprobs"):
        kl._scored({"choices": [{"text": "x"}]})


def test_a_fixed_text_compares_every_position_even_after_a_different_best_token():
    ref = [T("a", a=0.9), T("b", b=0.5, c=0.4), T("d", d=0.8)]
    same = kl.pair_text(ref, list(ref))
    assert same["positions"] == 3 and same["identical"] and max(same["kl"]) == 0 and same["differs_at"] is None
    other = [T("a", a=0.9), T("b", c=0.6, b=0.3), T("d", d=0.8)]       # the best token differs at the second
    p = kl.pair_text(ref, other)
    assert p["positions"] == 3 and p["agree"] == 2 and not p["identical"] and p["kl"][1] > 0
    assert p["worst"][0] == {"at": 2, "kl": p["kl"][1], "token": "b", "ref": "b", "cand": "c", "same_top": False}
    # a position without probability mass on one side is left unscored, the others still compare
    gone = [T("a", a=0.9), kl.Step(b"b", {b"b": 0.0}, "b"), T("d", d=0.8)]
    assert kl.pair_text(ref, gone)["unscored"] == 1 and kl.pair_text(ref, gone)["positions"] == 2
    with pytest.raises(RuntimeError, match="different tokens"):
        kl.pair_text(ref, [T("a", a=0.9), T("x", x=0.5)])


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=4000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _wait(runner, run):
    end = time.time() + 60
    while runner.busy() and time.time() < end:
        time.sleep(0.05)
    return runner.store.run(run["id"])


def _runner(tmp_path, base, key=None):
    runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: (base, key, "mock"))
    runner.power = False
    return runner


def test_a_kl_run_on_a_fixed_text(tmp_path, mock):
    runner = _runner(tmp_path, mock)
    done = _wait(runner, runner.start("kl", exactness=False, kl_text=TEXT))
    assert done["status"] == "done", done.get("error")
    assert done["kl"]["positions"] == len(TEXT.split(" ")) - 1 and done["kl"]["mean_kl"] == 0.0
    assert done["kl"]["identical"] == 1.0 and done["kl_text"] == TEXT and done["suites"] == ["text"]
    assert done["kl_reference"] == {"kind": "drafts-off", "drafts_off": True}
    assert done["suites_version"]["digest"] == bench.text_digest(TEXT)
    other = _wait(runner, runner.start("kl", exactness=False, kl_text="One two three four.", reference=(mock, None, "mock")))
    assert other["status"] == "done" and other["kl"]["positions"] == 3
    from tensorfold_web import report

    said = report.kl_line(done) + " ".join(report.findings(done)) + report.html_report(done)
    assert "of a fixed text" in said and "Every position of the fixed text" in said and "Same distributions" in said
    assert "replies identical" not in said and "first token where the two replies differ" not in said


def test_a_fixed_text_is_checked_before_the_run(tmp_path, mock):
    runner = _runner(tmp_path, mock)
    for mode, text in (("kl", ""), ("kl", "  "), ("kl", "x" * (kl.TEXT_MAX + 1)), ("quick", "a text")):
        with pytest.raises(ValueError, match="fixed text"):
            runner.start(mode, exactness=False, kl_text=text)


def test_an_engine_without_prompt_logprobs_is_said_plainly(tmp_path, mock, monkeypatch):
    monkeypatch.setattr(State, "no_prompt_scores", True)          # as TensorFold on Metal or its Python line
    runner = _runner(tmp_path, mock)
    done = _wait(runner, runner.start("kl", exactness=False, kl_text=TEXT))
    assert done["status"] == "failed"
    assert done["error"].startswith("the reference engine: it does not score a text's positions")
    assert "1.0.8 on CUDA" in done["error"] and "not supported by this engine" in done["error"]


def test_a_target_reached_with_headers_of_its_own(tmp_path):
    # a Fleet machine's relay takes its TensorFold Web's token as a header: the target carries it, the run never keeps it
    server = mock_serve(port=0, rate_tps=4000.0, api_key="needed")
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        runner = _runner(tmp_path, url)
        done = _wait(runner, runner.start("kl", exactness=False, kl_text="a b c", target=(url, None, "mock", {"x-api-key": "needed"})))
        assert done["status"] == "done", done.get("error")
        assert "needed" not in json.dumps(done)
    finally:
        server.shutdown()


def _app(port, token):
    s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return create_app(settings=s, engine=Engine(s), catalog=cat, token=token)


@pytest.fixture
def two(tmp_path, monkeypatch):
    """This TensorFold Web and another one (token TOKEN, in the fleet here) behind http://peer.test, both on a mock engine."""

    server = mock_serve(port=0)
    port = server.server_address[1]
    (tmp_path / "here").mkdir()
    (tmp_path / "there").mkdir()
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "there")
    peer = _app(port, TOKEN)
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "here")
    app = _app(port, "mine")
    here = TestClient(app, headers={"x-studio-token": "mine"})
    monkeypatch.setattr(fleet, "client", lambda: TestClient(peer, base_url="http://peer.test"))
    monkeypatch.setattr(fleet, "test", lambda *a, **k: {"ok": True, "checks": []})
    pid = here.post("/api/fleet", json={"url": "http://peer.test", "token": TOKEN, "name": "Node 2"}).json()["id"]
    yield here, app, pid
    server.shutdown()


def test_a_fleet_machine_is_identified_through_its_studio(two):
    here, _, pid = two
    r = here.post("/api/bench/identify", json={"peer": pid})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["url"] == "http://peer.test/engine" and got["models"] == [MODEL] and got["engine"] == "TensorFold"
    assert TOKEN not in r.text
    assert here.post("/api/bench/identify", json={"peer": "nobody"}).status_code == 404
    assert here.post("/api/bench/identify", json={"peer": 3}).status_code == 400
    assert here.post("/api/bench/identify", json={}).status_code == 400


def test_a_fleet_machine_as_target_and_as_reference_is_resolved_here(two, monkeypatch):
    here, app, pid = two
    seen = {}

    def start(mode, **kw):
        seen.update(kw, mode=mode)
        return {"id": "run-1", "mode": mode}
    monkeypatch.setattr(app.state.bench, "start", start)
    r = here.post("/api/bench/runs", json={"mode": "kl", "exactness": False, "target": {"peer": pid},
                                           "reference": {"peer": pid, "model": MODEL}, "kl_text": "a b c"})
    assert r.status_code == 200, r.text
    # the peer's relay, with its TensorFold Web's token (its engine has no key), kept on this side only
    assert seen["target"] == ("http://peer.test/engine", None, MODEL, {fleet.TOKEN_HEADER: TOKEN})
    assert seen["reference"] == ("http://peer.test/engine", None, MODEL, {fleet.TOKEN_HEADER: TOKEN})
    assert seen["tags"] == {"peer": "Node 2", "reference_peer": "Node 2"} and seen["kl_text"] == "a b c"
    assert TOKEN not in r.text
    assert here.post("/api/bench/runs", json={"mode": "kl", "reference": {"peer": "nobody"}}).status_code == 404
    assert here.post("/api/bench/runs", json={"mode": "quick", "target": {"peer": None}}).status_code == 400
    bad = here.post("/api/bench/runs", json={"mode": "kl", "reference": {"peer": pid, "model": "other"}})
    assert bad.status_code == 400 and "Node 2 does not serve" in bad.json()["detail"]


def test_a_fleet_machine_with_an_engine_key_is_reached_with_that_key(two, monkeypatch):
    here, app, pid = two
    monkeypatch.setattr(fleet, "engine_of", lambda peer: ("http://peer.test/engine", "engine-key", {}))
    seen = {}
    monkeypatch.setattr(app.state.bench, "start", lambda mode, **kw: seen.update(kw) or {"id": "run-2"})
    # the key is read from the peer's TensorFold Web by this one: identify goes through it, the page never sees it
    monkeypatch.setattr(engines, "identify", lambda url, key=None, client=None:
                        engines.Endpoint(url, models=[MODEL], engine="TensorFold"))
    r = here.post("/api/bench/runs", json={"mode": "quick", "target": {"peer": pid}})
    assert r.status_code == 200 and seen["target"] == ("http://peer.test/engine", "engine-key", MODEL, {})
    assert "engine-key" not in r.text
