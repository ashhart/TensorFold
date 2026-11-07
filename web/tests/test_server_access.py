import json

from fastapi.testclient import TestClient
from server_fixtures import _client, _echo, _mock

from tensorfold_web import engine as engine_mod
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, rates_from_counters
from tensorfold_web.server import create_app


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

def test_a_settings_save_during_a_comparison_writes_the_users_settings(tmp_path, monkeypatch):

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

def test_makers_from_settings_through_the_api(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, 9)      # no engine needed
    assert client.get("/api/brands").json() == {"brands": []}
    r = client.put("/api/brands/falcon", json={"label": "Falcon", "kind": "model", "words": ["falcon"]})
    assert r.status_code == 200 and r.json()["words"] == ["falcon"]
    assert client.put("/api/brands/falcon", json={"label": "Falcon", "kind": "toaster"}).status_code == 400
    assert [b["id"] for b in client.get("/api/brands").json()["brands"]] == ["falcon"]
    assert client.delete("/api/brands/falcon").status_code == 200
    assert client.delete("/api/brands/falcon").status_code == 404
