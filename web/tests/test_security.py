from fastapi.testclient import TestClient

from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


def _client(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, **kw)), s


def test_without_a_token_only_this_machines_names_are_answered(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    assert client.get("/api/settings", headers={"host": "127.0.0.1:8090"}).status_code == 200
    assert client.get("/api/settings", headers={"host": "localhost:8090"}).status_code == 200
    # a site that points its own name at 127.0.0.1 (DNS rebinding) reaches TensorFold Web under that name: refused
    r = client.get("/api/settings", headers={"host": "evil.example:8090"})
    assert r.status_code == 421 and "--token" in r.json()["detail"]
    client2, _ = _client(tmp_path, monkeypatch, allowed_hosts=["studio.lan"])
    assert client2.get("/api/settings", headers={"host": "studio.lan:8090"}).status_code == 200


def test_a_change_asked_by_another_sites_page_is_refused(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    h = {"host": "127.0.0.1:8090"}
    assert client.post("/api/engine/stop", headers={**h, "origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/engine/stop", headers={**h, "sec-fetch-site": "cross-site"}).status_code == 403
    assert client.put("/api/settings", json={"theme": "dark"}, headers={**h, "origin": "http://127.0.0.1:8090"}).status_code == 200
    assert client.put("/api/settings", json={"theme": "light"}, headers=h).status_code == 200      # a tool sends no Origin
    assert client.get("/api/settings", headers={**h, "origin": "https://evil.example"}).status_code == 200  # reads are CORS's


def test_behind_a_proxy_with_a_token_the_page_works_under_the_proxys_name(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, token="t0k")
    h = {"host": "mac-mini.tail.ts.net", "x-studio-token": "t0k"}
    assert client.get("/api/settings", headers=h).status_code == 200
    assert client.put("/api/settings", json={"theme": "dark"},
                      headers={**h, "origin": "https://mac-mini.tail.ts.net"}).status_code == 200
    # a proxy that sets the backend's address as Host still says the page's in X-Forwarded-Host
    h2 = {"host": "127.0.0.1:8090", "x-forwarded-host": "mac-mini.tail.ts.net", "x-studio-token": "t0k"}
    assert client.put("/api/settings", json={"theme": "dark"},
                      headers={**h2, "origin": "https://mac-mini.tail.ts.net"}).status_code == 200


def test_the_access_posture(tmp_path, monkeypatch):
    client, s = _client(tmp_path, monkeypatch)
    s.engine_url, s.remote = "http://127.0.0.1:8080", False
    got = client.get("/api/access").json()
    assert got["posture"] == "ok" and got["studio"] == {"bind": "127.0.0.1", "local_only": True, "token": False, "via": None}
    proxied = client.get("/api/access", headers={"x-forwarded-host": "mac-mini.tail.ts.net"}).json()
    assert proxied["studio"]["via"] == "mac-mini.tail.ts.net" and proxied["posture"] == "warn"
    server = mock_serve(port=0, host="0.0.0.0") if "host" in mock_serve.__code__.co_varnames else None
    try:
        if server is not None:
            port = server.server_address[1]
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            s.engine_url, s.remote = f"http://{ip}:{port}", True
            got = client.get("/api/access").json()
            if got["engine"]["answers_without_key"] is not None:
                assert got["posture"] == "danger" and got["engine"]["local_only"] is False
    finally:
        if server is not None:
            server.shutdown()


def test_a_credential_with_non_ascii_text_is_refused_not_a_crash(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, token="right")
    raw = "é".encode("latin-1")                       # as curl sends it; the server reads headers as latin-1
    assert client.get("/api/settings", headers={"x-studio-token": raw}).status_code == 401
    assert client.get("/api/settings?token=%C3%A9").status_code == 401
    assert client.get("/engine/v1/models", headers={"authorization": b"Bearer " + raw}).status_code == 401
    assert client.post("/api/access/redeem", json={"code": "é"}).status_code == 403
    assert client.get("/living-view/", headers={"x-studio-token": raw}).status_code == 401


def test_managed_and_secret_flags_are_refused_whatever_engine_runs_them(tmp_path, monkeypatch):
    client, s = _client(tmp_path, monkeypatch)
    for bad in (["--host", "0.0.0.0"], ["--host=0.0.0.0"], ["--api-key", "x"], ["--port", "1"]):
        r = client.put("/api/settings", json={"extra_args": bad})
        assert r.status_code == 400 and ("set by TensorFold Web" in r.json()["detail"] or "secret" in r.json()["detail"]), bad
    assert s.extra_args == []


def test_the_log_stream_opens_with_its_own_cookie_and_nothing_in_its_address(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, token="right")
    assert client.post("/api/engine/logs/session").status_code == 401       # the cookie is given to the token only
    r = client.post("/api/engine/logs/session", headers={"x-studio-token": "right"})
    assert r.status_code == 200 and r.cookies.get("tfs_logs")
    cookie = r.cookies["tfs_logs"]
    assert client.get("/api/settings", cookies={"tfs_logs": cookie}).status_code == 401   # the stream's path only
    # a method the route has not: 401 is the middleware's answer, 405 the router's (the stream itself never ends)
    assert client.request("PATCH", "/api/engine/logs/stream", cookies={"tfs_logs": "x" * 64}).status_code == 401
    assert client.request("PATCH", "/api/engine/logs/stream", cookies={"tfs_logs": cookie}).status_code == 405
