"""The 22 findings of Codex's review of 13 October 2026, one test each,
numbered as in the review."""

import json
import os
import stat
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorfold_web import bench, catalog, compare, conversations, kl, native, pull, recipes, replay, report, service
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.machine import Gpu, Machine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=4000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_01_a_change_of_engine_mid_poll_sends_no_key_to_the_old_one():
    s = settings_mod.Settings(engine_url="http://old.test:8080", remote=True, api_key="KEY-OLD")
    eng = Engine(s)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/health" and request.url.host == "old.test":
            with eng.target_lock:                         # Settings changed while this poll runs, as the route does
                s.engine_url, s.api_key = "http://new.test:8080", "KEY-NEW"
                eng.retargeted()
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model": "new-model"})
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    eng._client = httpx.Client(transport=httpx.MockTransport(handler))
    h = eng.health()
    old = [(host, auth) for host, _, auth in seen if host == "old.test"]
    assert old and all(auth == "Bearer KEY-OLD" for _, auth in old)          # the new key never went to the old engine
    assert all(auth == "Bearer KEY-NEW" for host, _, auth in seen if host == "new.test")
    assert h.model == "new-model"                         # the reading is the new engine's, read again


def test_02_learning_flags_are_the_studios_alone():
    flags = {"--slide": {"value": False}, "--slide-graph": {"value": True}, "--verbose": {"value": False}}
    for extra in (["--slide"], ["--slide-graph", "/tmp/g.json"]):
        with pytest.raises(ValueError, match="Teach turns learning on"):
            native.check_extra(extra, flags)
        with pytest.raises(ValueError):
            settings_mod.validate({"extra_args": extra})
    assert native.safe_extra(["--slide", "--slide-graph", "g", "--verbose"]) == (["--verbose"], ["--slide", "--slide-graph"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes")
def test_03_the_settings_folder_is_private_from_the_first_start(tmp_path, monkeypatch):
    conf = tmp_path / "conf"
    monkeypatch.setattr(settings_mod, "config_dir", lambda: conf)
    conf.mkdir(mode=0o755)
    (conf / "chat.sqlite").write_text("x")
    os.chmod(conf / "chat.sqlite", 0o644)
    settings_mod.secure_config_dir()
    assert stat.S_IMODE(conf.stat().st_mode) == 0o700
    assert stat.S_IMODE((conf / "chat.sqlite").stat().st_mode) == 0o600


def _chats(tmp_path, messages, settings=None):
    chats = conversations.Conversations(tmp_path / "chat.sqlite")
    doc = chats.create({"messages": messages, **({"settings": settings} if settings else {})})
    return chats, doc


def test_04_a_replay_address_with_a_password_is_refused_before_anything_is_kept(tmp_path):
    chats, doc = _chats(tmp_path, [{"role": "user", "content": "hi"}])
    r = replay.Replays(chats, lambda: ("http://127.0.0.1:9", None, "m", "TensorFold"))
    with pytest.raises(ValueError, match="user or password"):
        r.start(doc["id"], {"kind": "address", "url": "http://alice:secret@127.0.0.1:9", "model": "m"})
    assert len(chats.list()) == 1 and "secret" not in json.dumps(chats.list())


def test_05_drafts_off_needs_the_engines_word():
    assert bench.drafts_off_confirmed({"drafts": False})
    assert bench.drafts_off_confirmed({"drafted": 0})
    assert not bench.drafts_off_confirmed({"token_sha": "abc"})              # a digest alone says nothing of drafts
    assert not bench.drafts_off_confirmed({"drafts": True, "drafted": 0})    # contradictory: not trusted
    assert not bench.drafts_off_confirmed({})


def _conv(cid, text, stats, drafts=None):
    return {"id": cid, "title": cid, "settings": {"target": {"drafts": drafts}} if drafts is not None else {},
            "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": text, "outcome": "done", "stats": stats}]}


def test_06_a_comparison_without_evidence_says_unknown_not_different_nor_same():
    a, b = _conv("a", "same", {"tps": 10}), _conv("b", "same", {"tps": 9})
    c = replay.compare_replies(a, b)
    assert c["turns"][0]["same_tokens"] is None and c["checked"] == 0 and c["same_text"] == 1
    on = _conv("a", "same", {"sha": "x", "drafts": True})
    off_unconfirmed = _conv("b", "same", {"sha": "x", "drafts": True}, drafts=False)
    assert replay.compare_replies(on, off_unconfirmed)["turns"][0]["same_tokens"] is None
    off = _conv("b", "same", {"sha": "x", "drafts": False}, drafts=False)
    assert replay.compare_replies(on, off)["turns"][0]["same_tokens"] is True


def test_07_stream_events_are_not_counted_as_tokens():
    m = bench.measure(bench.Reply(text="three words here", events=2, sent_at=0.0, first_at=0.1, end_at=1.1, finish="stop"))
    assert m["tokens"] is None and m["tps"] is None and m["events"] == 2
    m = bench.measure(bench.Reply(text="x", events=2, usage={"completion_tokens": 10}, sent_at=0.0, first_at=0.1,
                                  end_at=1.1, finish="stop"))
    assert m["tokens"] == 10 and m["tps"] == pytest.approx(10.0)


def test_08_the_kl_width_reaches_the_results_and_the_summary(tmp_path, mock):
    runner = bench.Bench(bench.Store(tmp_path / "bench.sqlite"), lambda: (mock, None, "mock"))
    runner.power = False
    run = runner.start("kl", suites=["code"], exactness=False)
    end = time.time() + 60
    while runner.busy() and time.time() < end:
        time.sleep(0.05)
    got = runner.store.run(run["id"])
    assert got["status"] == "done", got.get("error")
    assert all("width" in r["kl"] for r in got["results"])
    assert got["kl"]["width"] == min(w for r in got["results"] for w in r["kl"]["width"])


def test_09_a_failed_request_is_not_a_wrong_answer():
    m = {"tps": None, "tokens": None, "ttft": None, "ttft_answer": None, "prefill_s": None, "gap_p50": None,
         "gap_p95": None, "runtime": {}}
    results = [{"suite": "gsm8k", "prompt": "1", "draft": True, "m": m, "error": "503", "passed": None, "category": "gsm8k"},
               {"suite": "gsm8k", "prompt": "2", "draft": True, "m": m, "error": None, "passed": True, "category": "gsm8k"}]
    q = bench.summarize(results)["quality"]
    assert q["overall"] == {"items": 1, "passed": 1, "unscored": 1}
    lines = report.findings({"mode": "quality", "status": "done", "summary": bench.summarize(results), "results": results})
    assert any("1 of 1 items right" in x for x in lines) and any("not graded" in x for x in lines)


def test_10_load_sweeps_of_different_suites_are_not_compared():
    a = {"mode": "load", "suites": ["prose_en"], "results": []}
    b = {"mode": "load", "suites": ["multilingual"], "results": []}
    with pytest.raises(compare.Refused, match="not send the same suites"):
        compare.check(a, b)


def test_11_a_stopped_attempt_finishing_late_leaves_the_new_one_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(pull, "command", lambda repo: [sys.executable, "-c", "import time; time.sleep(30)"])
    dl = pull.Downloads()
    first = dl.start("org/name")
    assert dl.cancel("org/name")
    second = dl.start("org/name")
    assert second is not first and second.state == "running" and first.state == "cancelled"
    time.sleep(0.5)                                       # the first worker has finished by now
    assert dl.get("org/name") is second and second.state == "running"
    assert dl.cancel("org/name")


def test_12_two_replays_at_the_same_instant_start_one(tmp_path, monkeypatch):
    chats, doc = _chats(tmp_path, [{"role": "user", "content": "hi"}])
    r = replay.Replays(chats, lambda: ("http://127.0.0.1:9", None, "m", "TensorFold"))
    monkeypatch.setattr(r, "_go", lambda *a, **k: time.sleep(1.0))
    slow_get = chats.get
    monkeypatch.setattr(chats, "get", lambda cid: (time.sleep(0.2), slow_get(cid))[1])
    outcomes = []

    def go():
        try:
            r.start(doc["id"], {"kind": "served"})
            outcomes.append("started")
        except RuntimeError:
            outcomes.append("refused")
    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["refused", "started"]


def test_13_a_workshop_conversation_is_not_replayed_as_another_one(tmp_path):
    for messages, settings in (
            ([{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
              {"role": "tool", "content": "4", "tool_call_id": "1"}], None),
            ([{"role": "user", "content": "hi"}], {"workshop": {"tools": False, "output": "json"}})):
        chats, doc = _chats(tmp_path / str(len(messages)), messages, settings)
        r = replay.Replays(chats, lambda: ("http://127.0.0.1:9", None, "m", "TensorFold"))
        with pytest.raises(ValueError, match="tools or a reply format"):
            r.start(doc["id"], {"kind": "served"})


def test_14_a_card_whose_memory_is_unknown_gets_no_verdict():
    m = Machine(os="Linux", os_version="6", arch="x86_64", chip="RTX 3090", ram_bytes=128 * 1024**3,
                ram_available_bytes=100 * 1024**3, unified_memory=False, gpus=[Gpu("NVIDIA GeForce RTX 3090", None, "8.6")],
                backend="cuda")
    assert catalog.capacity_of(m) is None
    m.unified_memory = True                               # a GB10: the RAM is the GPU's
    assert catalog.capacity_of(m) is not None


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))


def test_15_a_conversation_titled_in_any_script_exports(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    cid = c.post("/api/conversations", json={"messages": [{"role": "user", "content": "x"}], "title": "你好"}).json()["id"]
    for fmt in ("md", "json"):
        r = c.get(f"/api/conversations/{cid}/export?format={fmt}")
        assert r.status_code == 200 and "filename*=UTF-8''%E4%BD%A0%E5%A5%BD" in r.headers["content-disposition"]


def test_16_an_old_pinned_conversation_stays_listed_and_old_ones_are_found(tmp_path):
    chats = conversations.Conversations(tmp_path / "chat.sqlite")
    old = chats.create({"messages": [{"role": "user", "content": "the unique old question"}], "title": "Old one"})
    chats.update(old["id"], {"pinned": True}, rev=old["rev"])
    found = chats.create({"messages": [{"role": "user", "content": "an older needle"}], "title": "Needle"})
    for i in range(30):
        chats.create({"messages": [{"role": "user", "content": f"filler {i}"}]})
    listed = chats.list(limit=10)
    assert listed[0]["id"] == old["id"] and listed[0]["pinned"]
    assert [x["id"] for x in chats.list("older needle", limit=10)] == [found["id"]]


def test_17_a_name_with_a_slash_is_refused_and_an_old_one_can_still_go(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="without a slash"):
        conversations.validate_preset({"name": "Work/2026"})
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/conversation-folders", json={"name": "Work/2026"}).status_code == 400
    chats = conversations.Conversations(tmp_path / "chat.sqlite")
    with chats._db() as db:                               # a preset kept before the rule
        db.execute("insert into presets (name, created, doc) values (?, ?, ?)",
                   ("Old/one", time.time(), json.dumps({"name": "Old/one", "system": ""})))
    assert c.delete("/api/chat-presets/Old%2Fone").status_code == 200


def test_18_positions_without_probabilities_are_left_out():
    empty = [kl.Step(b"a", {}), kl.Step(b"b", {})]
    p = kl.pair(empty, empty)
    assert p["positions"] == 0 and p["unscored"] == 2 and p["agree"] == 0
    assert kl.summary([{**p, "prompt": "s/p"}]) == {"positions": 0, "unscored": 2}


def test_19_energy_coverage_is_over_the_whole_measured_work():
    class Power(bench.power.Source):
        name, scope, whole_machine = "fake", "whole machine", True

        def read(self):
            return bench.power.Reading(watts=10.0, cpu_c=None, gpu_c=None)
    s = bench.Sampler(lambda: None, source=Power())
    s.measured_from = 0.0
    s.series = [[0.9, 10.0, None, None, None, None], [1.9, 10.0, None, None, None, None], [2.9, None, None, None, None, None]]
    r = s.report(output_tokens=100)
    assert r["coverage"] == pytest.approx(1.0 / 2.9, abs=0.01) and r["partial"] is True
    live = bench.Sampler(lambda: None, every=30, source=Power())
    with live:
        live.measuring()
        time.sleep(0.3)
        r = live.report(output_tokens=10)
    assert r["coverage"] == pytest.approx(1.0, abs=0.1) and r["partial"] is False


def test_20_a_check_that_could_not_be_made_is_not_a_pass():
    r = recipes._out("context", True, "the engine does not say its window; not checked", time.perf_counter(), skipped=True)
    assert r["skipped"] is True
    assert recipes._out("models", True, "m is listed", time.perf_counter())["skipped"] is False


def test_22_the_service_keeps_a_moved_settings_folder(tmp_path, monkeypatch):
    if service.platform_name() != "linux":
        pytest.skip("the systemd unit")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "my config"))
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "my config" / "tensorfold-web")
    p = service.plan()
    assert f'Environment="XDG_CONFIG_HOME={tmp_path / "my config"}"' in p.content
