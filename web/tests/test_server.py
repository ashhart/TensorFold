import time

from fastapi.testclient import TestClient

from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, parse_health
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


def _mock(**kw):
    server = mock_serve(port=0, **kw)
    return server, server.server_address[1]


def _client(tmp_path, monkeypatch, port, **settings):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True, **settings)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None       # offline in tests
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat)), s


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


def test_settings_are_validated(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        assert client.put("/api/settings", json={"budget_fraction": -1}).status_code == 400
        assert client.put("/api/settings", json={"remote": "false"}).status_code == 400
        assert client.put("/api/settings", json={"extra_args": "--vision"}).status_code == 400
        assert client.put("/api/settings", json={"engine_url": "spark.local"}).status_code == 400
        ok = client.put("/api/settings", json={"engine_url": f"http://127.0.0.1:{port}/", "thinking": "off"}).json()
        assert ok["engine_url"] == f"http://127.0.0.1:{port}" and ok["thinking"] == "off"
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


def test_page_never_serves_files_outside_its_folder(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        for p in ("/../server.py", "/..%2Fserver.py", "/assets/../../server.py", "/%2e%2e/pyproject.toml"):
            r = client.get(p)
            assert "fastapi" not in r.text and "[project]" not in r.text, p
        assert client.get("/api/nothing").status_code == 404
        assert client.get("/engine-ish").status_code in (200, 404)
    finally:
        server.shutdown()


def test_studio_token_gates_api_and_relay(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"))
        assert client.get("/api/engine").status_code == 401
        assert client.get("/engine/v1/models").status_code == 401
        assert client.get("/api/engine", headers={"x-studio-token": "t0k"}).status_code == 200
        assert client.get("/api/engine?token=t0k").status_code == 401      # the header only: an address ends in logs
        assert client.get("/").status_code == 200
        for p in ("/openapi.json", "/docs", "/redoc"):                      # no route map outside the token
            r = client.get(p)
            assert r.status_code == 404 or ('"paths"' not in r.text and "swagger" not in r.text.lower()), p
    finally:
        server.shutdown()


def test_streaming_chat_through_relay(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        with client.stream("POST", "/engine/v1/chat/completions",
                           json={"model": "Qwen3.8-27B", "stream": True, "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "hi"}]}) as r:
            assert r.status_code == 200
            text = "".join(r.iter_text())
        assert "data: [DONE]" in text and text.count("chat.completion.chunk") >= 5
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


# -- the follow-up review's scenarios (5 Oct 2026) -------------------------------------------
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tensorfold_web import engine as engine_mod
from tensorfold_web.engine import rates_from_counters


def _echo():
    """An engine stand-in that answers every GET with the headers and query it received."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path, _, query = self.path.partition("?")
            body = json.dumps({"status": "ok", "path": path, "query": query,
                               "headers": {k.lower(): v for k, v in self.headers.items()}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_relay_never_forwards_the_studio_token(tmp_path, monkeypatch):
    server, port = _echo()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"))
        seen = client.get("/engine/v1/models?token=t0k&x=1", headers={"x-studio-token": "t0k"}).json()
        assert seen["query"] == "x=1" and "x-studio-token" not in seen["headers"]
        # no key held: a client's own credentials pass through
        seen = client.get("/engine/v1/models", headers={"x-studio-token": "t0k", "authorization": "Bearer mine"}).json()
        assert seen["headers"]["authorization"] == "Bearer mine"
        # a key held: the relay's key, whatever the client sent
        s.api_key = "tf-real"
        seen = client.get("/engine/v1/models", headers={"x-studio-token": "t0k", "authorization": "Bearer local"}).json()
        assert seen["headers"]["authorization"] == "Bearer tf-real" and "x-api-key" not in seen["headers"]
    finally:
        server.shutdown()


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


def test_a_launch_error_reverts_the_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:1", remote=False, model="before", budget_fraction=0.7)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    monkeypatch.setattr(engine_mod, "tensorfold_command", lambda: ["tensorfold-that-does-not-exist"])
    monkeypatch.setattr(engine_mod, "memory_limit_env", lambda fraction: {})
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    r = client.post("/api/engine/start", json={"model": "org/other", "budget_fraction": 0.85})
    assert r.status_code == 500 and "could not launch" in r.json()["detail"]
    assert s.model == "before" and s.budget_fraction == 0.7
    assert client.post("/api/engine/start", json={"model": ""}).status_code == 400


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


def test_cuda_counters_become_rates_and_unknown_stays_unknown():
    h = parse_health({"ok": True, "status": "ok", "requests_running": 1, "completion_tokens_total": 100,
                      "decode_seconds_total": 2.0, "prompt_tokens_total": 1000, "prefill_seconds_total": 0.5})
    assert h.ok and h.counters["completion_tokens_total"] == 100 and h.decode_tps is None
    later = {"completion_tokens_total": 190.0, "decode_seconds_total": 4.0, "prompt_tokens_total": 1000.0,
             "prefill_seconds_total": 0.5}
    assert rates_from_counters(None, later) == (None, None)
    assert rates_from_counters(h.counters, later) == (45.0, 0.0)
    assert rates_from_counters(h.counters, {"completion_tokens_total": 190.0}) == (None, None)
    minimal = parse_health({"status": "ok"})                      # the MLX server once an API key is on
    assert minimal.ok and minimal.model is None and minimal.decode_tps is None and minimal.counters is None


def test_memory_limit_variable_follows_the_backend(monkeypatch):
    import tensorfold_web.catalog as catalog_mod
    from tensorfold_web import fit
    from tensorfold_web import machine as machine_mod

    machine = types.SimpleNamespace(backend="cuda", ram_bytes=64 * fit.GIB, unified_memory=False, gpu_ceiling_bytes=None,
                                    gpus=[types.SimpleNamespace(memory_bytes=24 * fit.GIB)])
    monkeypatch.setattr(machine_mod, "detect", lambda: machine)
    monkeypatch.setattr(catalog_mod, "capacity_of", lambda m: fit.cuda_capacity(24 * fit.GIB, unified=False))
    assert engine_mod.memory_limit_env(0.5) == {"TENSORFOLD_CUDA_MEMORY_LIMIT_GB": "12.0"}


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


def test_prompt_tokens_restored_from_the_cache_do_not_inflate_the_prefill_rate():
    before = {"prompt_tokens_total": 10125.0, "prefill_seconds_total": 1.43, "prompt_tokens_cached_total": 0.0}
    after = {"prompt_tokens_total": 30453.0, "prefill_seconds_total": 2.03, "prompt_tokens_cached_total": 16384.0}
    _, prefill = rates_from_counters(before, after)
    assert round(prefill) == round((20328 - 16384) / 0.6)


def test_manifest_and_icons_without_token(tmp_path, monkeypatch):
    """The browser fetches the app manifest and icons before any token: they are served, the API still is not."""

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"))
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    assert r.json()["start_url"] == "/#models"
    assert client.get("/icon-512.png").headers["content-type"] == "image/png"
    assert client.get("/api/engine").status_code == 401


def test_cross_site_writes_compare_host_and_port(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    open_studio = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat), base_url="http://localhost:8090")

    def put(client, origin, **headers):
        return client.put("/api/settings", json={"theme": "dark"}, headers={"origin": origin, **headers}).status_code

    assert put(open_studio, "http://localhost:3000") == 403          # another local server is another site
    assert put(open_studio, "http://localhost:8090") == 200
    assert put(open_studio, "null") == 403                           # no origin to check, no token to require
    with_token = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, token="t0k"),
                            base_url="http://box.example.ts.net", headers={"x-studio-token": "t0k"})
    assert put(with_token, "http://box.example.ts.net") == 200      # port 80 on both sides
    assert put(with_token, "https://box.example.ts.net:8443", host="box.example.ts.net:8443") == 200
    assert put(with_token, "https://box.example.ts.net:8443") == 403


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


def test_settings_sent_unchanged_do_not_slip_past_a_comparison(tmp_path, monkeypatch):
    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, model="before")
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        app = create_app(settings=s, engine=Engine(s), catalog=cat)
        client = TestClient(app)
        app.state.gate.take("a comparison of configurations")
        s.model = "set-by-the-comparison"                       # what the comparison installs meanwhile
        # the page sends its whole form, the model as it read it before
        r = client.put("/api/settings", json={"model": "set-by-the-comparison", "energy_price": 0.3})
        assert r.status_code == 200 and s.energy_price == 0.3 and s.model == "set-by-the-comparison"
        assert client.put("/api/settings", json={"model": "before"}).status_code == 409
        app.state.gate.give()
    finally:
        server.shutdown()


def test_a_cancelled_stop_keeps_the_gate_until_the_engine_has_stopped(tmp_path, monkeypatch):
    import asyncio
    import threading

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


def test_a_request_cancelled_before_its_worker_runs_takes_nothing(tmp_path, monkeypatch):
    import asyncio
    import concurrent.futures

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    eng = Engine(s)
    eng.stop = lambda: None
    app = create_app(settings=s, engine=eng, catalog=cat)
    stop = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/engine/stop")

    async def go():
        loop = asyncio.get_running_loop()
        busy = concurrent.futures.ThreadPoolExecutor(max_workers=1)    # the only worker is taken: the stop queues
        loop.set_default_executor(busy)
        block = asyncio.Event()
        hold = loop.run_in_executor(busy, lambda: asyncio.run_coroutine_threadsafe(block.wait(), loop).result())
        task = asyncio.create_task(stop())
        await asyncio.sleep(0.1)
        task.cancel()                                                  # gone before its worker ever ran
        await asyncio.sleep(0.05)
        block.set()
        await hold
        await asyncio.sleep(0.1)
        assert app.state.gate.holder is None
    asyncio.run(go())


def test_a_settings_save_during_a_comparison_writes_the_users_settings(tmp_path, monkeypatch):
    import json

    server, port = _mock()
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=False, model="mine", context=None)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        app = create_app(settings=s, engine=Engine(s), catalog=cat)
        client = TestClient(app)
        app.state.gate.take("a comparison of configurations")
        app.state.temporarily(model="org/config-b", context=8192)       # what the comparison runs for a while
        assert client.put("/api/settings", json={"energy_price": 0.25}).status_code == 200
        saved = json.loads((tmp_path / "settings.json").read_text())
        assert saved["model"] == "mine" and saved["context"] is None and saved["energy_price"] == 0.25
        assert s.model == "org/config-b"                                  # the engine still runs the comparison's
        app.state.comparison_ended()
        app.state.gate.give()
        assert s.model == "mine" and s.context is None
    finally:
        server.shutdown()


def test_a_comparison_starting_during_a_settings_save_never_reaches_the_disk(tmp_path, monkeypatch):
    import json
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


def test_the_served_model_is_named_as_the_catalog_names_it(tmp_path, monkeypatch):
    """The page says "Qwen3.8-27B", not the file the engine was given."""

    from tensorfold_web import mock_engine

    monkeypatch.setattr(mock_engine, "MODEL", "TensorFold/Qwen3.8-27B-MLX-4bit")
    server, port = _mock()
    try:
        client, _ = _client(tmp_path, monkeypatch, port)
        end = time.time() + 10
        while client.get("/api/engine").json()["status"] != "running" and time.time() < end:
            time.sleep(0.1)
        info = client.get("/api/engine").json()
        assert info["model"] == "TensorFold/Qwen3.8-27B-MLX-4bit" and info["model_name"] == "Qwen3.8-27B"
    finally:
        server.shutdown()


def test_one_model_is_recommended_when_one_runs_here(tmp_path, monkeypatch):
    from tensorfold_web.catalog import Catalog

    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    from tensorfold_web.machine import Machine

    machine = Machine(os="macOS", os_version="27.0", arch="arm64", chip="Apple M6", ram_bytes=32 * 1024**3,
                      ram_available_bytes=20 * 1024**3, unified_memory=True, backend="mlx")
    monkeypatch.setattr(cat, "native_here", lambda _e: False)
    rows = cat.view(machine, backend="mlx")
    picked = [m for m in rows if m.get("recommended")]
    assert len(picked) <= 1
    if picked:
        assert picked[0]["runs_here"] and picked[0]["verdict"]["level"] in ("fits", "tight")


def test_chat_preset_routes(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, 9)
    p = {"name": "Mine", "system": "", "temperature": 0.5, "top_p": 0.9}
    assert client.post("/api/chat-presets", json={"preset": p}).status_code == 200
    assert client.post("/api/chat-presets", json={"preset": p}).status_code == 409
    assert client.post("/api/chat-presets", json={"preset": {**p, "top_p": 2}, "replace": True}).status_code == 400
    assert client.put("/api/chat-presets/Mine", json={"name": "Yours"}).json()["top_p"] == 0.9
    assert [x["name"] for x in client.get("/api/chat-presets").json()["presets"]] == ["Yours"]
    assert client.delete("/api/chat-presets/Yours").status_code == 200
    assert client.delete("/api/chat-presets/Yours").status_code == 404


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


def test_makers_from_settings_through_the_api(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, 9)      # no engine needed
    assert client.get("/api/brands").json() == {"brands": []}
    r = client.put("/api/brands/falcon", json={"label": "Falcon", "kind": "model", "words": ["falcon"]})
    assert r.status_code == 200 and r.json()["words"] == ["falcon"]
    assert client.put("/api/brands/falcon", json={"label": "Falcon", "kind": "toaster"}).status_code == 400
    assert [b["id"] for b in client.get("/api/brands").json()["brands"]] == ["falcon"]
    assert client.delete("/api/brands/falcon").status_code == 200
    assert client.delete("/api/brands/falcon").status_code == 404
