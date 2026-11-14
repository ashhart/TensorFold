import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from fastapi.testclient import TestClient

from tensorfold_web import catalog, native
from tensorfold_web import settings as settings_mod
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve
from tensorfold_web.server import create_app


def test_startup_and_local_views_do_not_request_remote_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(native, "find_binary", lambda: None)
    called = threading.Event()
    monkeypatch.setattr(catalog, "repo_size", lambda repo: called.set())
    monkeypatch.setattr(catalog, "installed", lambda *args: False)
    local = serve(port=0)
    try:
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{local.server_address[1]}", remote=True)
        with TestClient(create_app(settings=s, engine=Engine(s)), headers={"host": "127.0.0.1"}) as client:
            for path in ("/", "/api/machine", "/api/engine", "/api/catalog", "/api/conversations"):
                assert client.get(path).status_code == 200
            assert not called.wait(0.1), "startup contacted the model hub"
    finally:
        local.shutdown()
        local.server_close()


def test_loopback_engine_and_relay_bypass_an_active_system_proxy(tmp_path, monkeypatch):
    received = []
    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            self.send_response(503)
            self.end_headers()
        def log_message(self, *args):
            pass
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    local = serve(port=0)
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(native, "find_binary", lambda: None)
    monkeypatch.setattr(catalog.Catalog, "refresh_sizes_async", lambda self: None)
    proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
    engine_url = f"http://localhost:{local.server_address[1]}"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, proxy_url)
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "")
    try:
        with httpx.Client() as control:
            assert control.get(engine_url + "/v1/models").status_code == 503
        assert received == [engine_url + "/v1/models"], "the proxy control must actually intercept loopback"
        s = settings_mod.Settings(engine_url=engine_url, remote=True)
        eng = Engine(s)
        assert eng.status().status == "running"
        from tensorfold_web.engines import identify
        assert identify(engine_url).models
        with TestClient(create_app(settings=s, engine=eng), headers={"host": "127.0.0.1"}) as client:
            response = client.get("/engine/v1/models")
            assert response.status_code == 200, response.text
        assert received == [engine_url + "/v1/models"], "engine requests reached the proxy"
        eng._client.close()
    finally:
        local.shutdown()
        local.server_close()
        proxy.shutdown()
        proxy.server_close()


def test_discover_is_the_explicit_size_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(native, "find_binary", lambda: None)
    calls = []
    cat = catalog.Catalog()
    monkeypatch.setattr(cat, "refresh_sizes_async", lambda: calls.append("discover"))
    s = settings_mod.Settings()
    with TestClient(create_app(settings=s, engine=Engine(s), catalog=cat)) as client:
        assert client.get("/api/catalog").status_code == 200
        assert not calls
        assert client.post("/api/catalog/discover").status_code == 200
        assert calls == ["discover"]


def test_stand_in_reports_build_version_not_an_old_path_binary(tmp_path, monkeypatch):
    import re
    from pathlib import Path

    from tensorfold_web.mock_engine import native_version
    expected = re.search(r'\.version\s*=\s*"([^"]+)"', (Path(__file__).parents[2] / "build.zig.zon").read_text())[1]
    assert native_version() == expected
    local = serve(port=0)
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(native, "find_binary", lambda: None)
    from tensorfold_web import machine
    monkeypatch.setattr(machine, "_tensorfold_version", lambda: "1.1.6")
    machine._reset()
    try:
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{local.server_address[1]}", remote=True)
        eng = Engine(s)
        eng.mock_version = native_version()
        with httpx.Client(trust_env=False) as direct:
            assert direct.get(s.engine_url + "/version").json() == {"version": expected, "mock": True}
        with TestClient(create_app(settings=s, engine=eng)) as client:
            assert client.get("/api/machine").json()["tensorfold_version"] == expected
    finally:
        machine._reset()
        local.shutdown()
        local.server_close()
