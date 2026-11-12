import json
import threading
import time
import types

from fastapi.testclient import TestClient
from server_fixtures import _client, _mock

from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, parse_health
from tensorfold_web.server import create_app


def test_parse_health_mlx_shape():
    h = parse_health({"status": "ok", "model": "m", "model_ids": ["m"], "warming": False,
                      "live": {"connections": 2, "waiting": 1, "decode_tokens_per_second": 27.3,
                               "prefill_tokens_per_second": 1400.0}, "memory": {"active": 1}})
    assert h.ok and h.status == "running" and h.connections == 2 and h.decode_tps == 27.3

def test_parse_health_cuda_counters_and_errors():
    h = parse_health({"status": "ok", "requests_running": 3, "requests_waiting": 0, "context_length": 17408,
                      "requests_total": 10})
    assert h.ok and h.connections == 3 and h.context_length == 17408 and h.model is None
    e = parse_health({"status": "error"})
    assert not e.ok and e.status == "error"

def test_api_against_mock_engine(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        m = client.get("/api/machine").json()
        assert m["ram_bytes"] > 0 and "backend" in m and "basis" in m["budget"]
        time.sleep(0.1)
        e = client.get("/api/engine").json()
        assert e["model"] == "Qwen3.8-27B" and e["managed"] is False and e["owned"] is False
        c = client.get("/api/catalog").json()
        assert any(x["id"].endswith("Qwen3.8-27B-MLX-4bit") for x in c["models"])
        r = client.get("/engine/v1/models").json()
        assert r["data"][0]["id"] == "Qwen3.8-27B"
        snippets = client.get("/api/connect").json()["snippets"]
        assert any(sn["id"] == "claude-code" and f"http://127.0.0.1:{port}" in sn["snippet"] for sn in snippets)
        put = client.put("/api/settings", json={"budget_fraction": 0.85}).json()
        assert put["budget_fraction"] == 0.85 and "api_key" not in {k for k, v in put.items() if v and k == "api_key"} or put["api_key"] is None
        assert client.get("/api/pull").json() == {"jobs": []}
    finally:
        server.shutdown()

def test_sharing_generates_a_key_and_the_relay_sends_it(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-secret")
    try:
        client, s = _client(tmp_path, monkeypatch, port, api_key="tf-secret")
        assert client.get("/engine/v1/models").status_code == 200       # the relay added the key
        s.api_key = "wrong"
        assert client.get("/engine/v1/models").status_code == 401
        s.api_key = None
        put = client.put("/api/settings", json={"share": True}).json()
        assert put["api_key_set"] is True and put["api_key"] == "•" * 8
        conn = client.get("/api/connect").json()
        assert conn["api_key"] and conn["api_key"].startswith("tf-") and conn["api_key"] in conn["snippets"][0]["snippet"]
    finally:
        server.shutdown()

def test_refused_start_changes_no_setting(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, model="before")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
        r = client.post("/api/engine/start", json={"model": "org/other", "budget_fraction": 0.85})
        assert r.status_code == 409 and "already answers" in r.json()["detail"]
        assert s.model == "before" and s.budget_fraction is None
    finally:
        server.shutdown()

def test_startup_lines_are_read_as_numbers():
    s = settings_mod.Settings()
    e = Engine(s)
    e._emit("[tensorfold] memory budget 25.0 GiB: MLX's buffers up to 22.0 GiB, 3 GiB for the rest of the process")
    e._emit("[tensorfold] 18.3 GiB of weights kept resident")
    e._emit("[tensorfold] concurrency: up to 8 requests share each round; a stream 166 MB at 64 tokens, then 64.0 KB a token")
    e._emit("[tensorfold] context window 1,024 tokens: the most one request can use in the 25.0 GiB memory budget")
    e._emit("[tensorfold] serving X at http://127.0.0.1:8080/v1 (sampling: temperature 1.0; context: 1024; loaded in 78.3s)")
    assert e.startup == {"budget_gib": 25.0, "resident_gib": 18.3, "bytes_per_token": 64000.0,
                         "context_tokens": 1024.0, "loaded_s": 78.3}

def test_engine_key_opens_the_relay_but_not_the_studio(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-k")
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True, api_key="tf-k")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"))
        assert client.get("/engine/v1/models", headers={"authorization": "Bearer tf-k"}).status_code == 200
        assert client.get("/engine/v1/models", headers={"x-api-key": "tf-k"}).status_code == 200
        assert client.get("/engine/v1/models", headers={"authorization": "Bearer nope"}).status_code == 401
        assert client.get("/api/engine", headers={"authorization": "Bearer tf-k"}).status_code == 401
        assert client.get("/api/settings", headers={"x-api-key": "tf-k"}).status_code == 401
    finally:
        server.shutdown()

def test_rotation_keeps_the_running_engines_key_until_restart(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-old")
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, api_key="tf-old")
        eng = Engine(s)
        eng.proc = types.SimpleNamespace(alive=lambda: True)     # an owned process, started with the old key
        eng.active_key, eng.model = "tf-old", "org/model"
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=eng, catalog=cat))
        r = client.post("/api/settings/api-key").json()
        assert r["key_pending_restart"] is True and s.api_key != "tf-old"
        assert eng.key_in_use() == "tf-old"
        assert client.get("/engine/v1/models").status_code == 200          # the relay still opens the engine
        info = client.get("/api/engine").json()
        assert info["status"] == "running" and info["key_pending_restart"] is True and info["model_repo"] == "org/model"
        conn = client.get("/api/connect").json()
        # the page shows the new key for the relay at once: it opens it already, and the old one is not offered
        assert conn["key_pending_restart"] is True and conn["relay_key"] == s.api_key and conn["api_key"] == "tf-old"
    finally:
        server.shutdown()

def test_the_studio_is_not_pointed_elsewhere_while_it_runs_its_own_engine(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-local")
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, api_key="tf-local")
        eng = Engine(s)
        eng.proc = types.SimpleNamespace(alive=lambda: True)     # an owned process, with its own key
        eng.active_key, eng.model = "tf-local", "org/model"
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=eng, catalog=cat))
        r = client.put("/api/settings", json={"engine_url": "http://spark.local:8080", "remote": True})
        assert r.status_code == 409 and "stop it" in r.json()["detail"]
        assert s.engine_url == f"http://127.0.0.1:{port}" and s.remote is False   # nothing of the patch applied
    finally:
        server.shutdown()

def test_a_remote_engines_key_can_be_entered_and_a_wrong_one_is_reported(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-remote")
    try:
        client, s = _client(tmp_path, monkeypatch, port)
        assert client.get("/api/engine").json()["status"] == "unauthorized"       # answers, refuses: not "running"
        assert client.put("/api/settings", json={"api_key": ""}).status_code == 400
        put = client.put("/api/settings", json={"api_key": " tf-remote "}).json()
        assert put["api_key_set"] is True and s.api_key == "tf-remote"
        assert client.get("/api/engine").json()["status"] == "running"
        assert client.get("/engine/v1/models").status_code == 200
        assert client.put("/api/settings", json={"api_key": None}).json()["api_key_set"] is False
        assert client.get("/api/engine").json()["status"] == "unauthorized"
    finally:
        server.shutdown()

def test_engine_url_needs_a_host_and_a_numeric_port(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        for bad in ("http://", "http://spark.local:abc", "spark.local:8080"):
            assert client.put("/api/settings", json={"engine_url": bad}).status_code == 400, bad
    finally:
        server.shutdown()

def test_changing_the_engine_address_forgets_the_previous_one(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url="http://127.0.0.1:1", remote=True)
        eng = Engine(s)
        eng.startup["context_tokens"] = 16384.0             # read from an engine that is no longer the target
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=eng, catalog=cat))
        client.put("/api/settings", json={"engine_url": f"http://127.0.0.1:{port}"})
        assert client.get("/api/engine").json()["startup"] == {"context_tokens": 32768.0}   # the new one's own window
    finally:
        server.shutdown()

def test_metrics_stand_in_for_a_bare_health_behind_a_key(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-k", rate_tps=200.0)
    try:
        client, _ = _client(tmp_path, monkeypatch, port, api_key="tf-k")
        first = client.get("/api/engine").json()
        assert first["status"] == "running" and first["model"] == "Qwen3.8-27B" and first["source"] == "metrics"
        assert first["memory"]["footprint"] > 0 and first["decode_tps"] is None       # one reading: no rate yet
        client.post("/engine/v1/chat/completions", json={"model": "Qwen3.8-27B", "max_tokens": 5,
                                                         "messages": [{"role": "user", "content": "hi"}]})
        second = client.get("/api/engine").json()
        assert second["decode_tps"] and 100 < second["decode_tps"] < 400 and second["draft_acceptance"] == 0.5
        third = client.get("/api/engine").json()
        assert third["decode_tps"] == 0.0                                           # nothing new: a true zero
    finally:
        server.shutdown()

def test_relay_keeps_the_key_the_engine_still_uses_after_a_rotation(tmp_path, monkeypatch):
    server, port = _mock(api_key="tf-old")
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True, api_key="tf-new")
        eng = Engine(s)
        monkeypatch.setattr(eng, "key_in_use", lambda: "tf-old")      # an owned engine started with the old key
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=eng, catalog=cat, token="t0k"))
        assert client.get("/engine/v1/models", headers={"authorization": "Bearer tf-old"}).status_code == 200
        assert client.get("/engine/v1/models", headers={"authorization": "Bearer tf-new"}).status_code != 401
        assert client.get("/engine/v1/models", headers={"authorization": "Bearer nope"}).status_code == 401
    finally:
        server.shutdown()

def test_one_change_of_the_engine_at_a_time(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, model="before")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        app = create_app(settings=s, engine=Engine(s), catalog=cat)
        client = TestClient(app)
        app.state.gate.take("a comparison of configurations")      # as a comparison holds it for its whole length
        for method, path, body in (("post", "/api/engine/start", {"model": "org/x"}), ("post", "/api/engine/stop", None),
                                   ("put", "/api/settings", {"context": 8192}), ("post", "/api/settings/api-key", None),
                                   ("post", "/api/pull", {"repo": "org/x"}),
                                   ("post", "/api/bench/runs", {"mode": "quick"})):
            r = getattr(client, method)(path, **({"json": body} if body is not None else {}))
            assert r.status_code == 409 and "comparison" in r.json()["detail"], (path, r.text)
        assert s.context is None and s.model == "before"
        assert client.put("/api/settings", json={"energy_price": 0.2}).status_code == 200   # not the engine's
        app.state.gate.give()
        assert client.put("/api/settings", json={"context": 8192}).status_code == 200 and s.context == 8192
        assert app.state.gate.holder is None
    finally:
        server.shutdown()

def test_a_cancelled_stop_keeps_the_gate_until_the_engine_has_stopped(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    eng = Engine(s)
    release = threading.Event()
    eng.stop = lambda: release.wait(5)                          # a slow stop
    app = create_app(settings=s, engine=eng, catalog=cat)
    stop = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/engine/stop")

    async def go():
        task = asyncio.create_task(stop())
        await asyncio.sleep(0.1)
        task.cancel()                                           # the page went away
        await asyncio.sleep(0.1)
        assert app.state.gate.holder == "an engine stop"       # still held: the stop is still going
        release.set()
        for _ in range(50):
            if app.state.gate.holder is None:
                break
            await asyncio.sleep(0.05)
        assert app.state.gate.holder is None
    asyncio.run(go())

def test_a_comparison_starting_during_a_settings_save_never_reaches_the_disk(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False, model="mine")
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    app = create_app(settings=s, engine=Engine(s), catalog=cat)
    client = TestClient(app)
    real_save, in_save, go_on = settings_mod.save, threading.Event(), threading.Event()

    def slow_save(snapshot):
        in_save.set()
        go_on.wait(5)                                              # paused just before it writes
        real_save(snapshot)
    monkeypatch.setattr(settings_mod, "save", slow_save)
    saving = threading.Thread(target=lambda: client.put("/api/settings", json={"energy_price": 0.3}))
    saving.start()
    assert in_save.wait(5)
    starting = threading.Thread(target=lambda: app.state.temporarily(model="org/config-b"))
    starting.start()                                               # a comparison begins meanwhile
    go_on.set()
    saving.join(5)
    starting.join(5)
    assert json.loads((tmp_path / "settings.json").read_text())["model"] == "mine"
    assert s.model == "org/config-b"
    app.state.comparison_ended()

def test_an_app_with_the_engine_key_may_post_from_its_own_origin(tmp_path, monkeypatch):
    """Jan, built on a web view, sends its own origin with every call: its key makes it a client, not a page."""

    server, port = _mock(api_key="tf-key")
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True, api_key="tf-key")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"),
                            base_url="https://box.example.ts.net:8443")
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        app = {"origin": "http://tauri.localhost"}
        assert client.post("/engine/v1/chat/completions", json=body,
                           headers={**app, "authorization": "Bearer tf-key"}).status_code == 200
        # without the key, the same call is refused: by the origin, then by the missing token
        assert client.post("/engine/v1/chat/completions", json=body, headers=app).status_code == 403
        assert client.post("/engine/v1/chat/completions", json=body,
                           headers={**app, "authorization": "Bearer wrong"}).status_code == 403
        assert client.put("/api/settings", json={"theme": "dark"},          # TensorFold Web's own API is no relay
                          headers={**app, "authorization": "Bearer tf-key"}).status_code == 403
    finally:
        server.shutdown()
