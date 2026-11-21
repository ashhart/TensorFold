import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorfold_web import engine, gate, recipes, server_connect, server_settings, settings


@pytest.mark.parametrize("source", ["settings-api", "saved-on-disk"])
@pytest.mark.parametrize(("address", "prefix"), [
    ("http://fixture.test/proxy", "/proxy"),
    ("http://fixture.test/proxy/v1/", "/proxy"),
    ("http://fixture.test/proxy/v1/v1", "/proxy/v1"),
])
def test_compatible_engine_address_may_end_in_v1(tmp_path, monkeypatch, source, address, prefix):
    def respond(request):
        if request.url.path == prefix + "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == prefix + "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture"}]})
        return httpx.Response(404)

    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)
    if source == "saved-on-disk":
        settings.path().write_text(json.dumps({"engine_url": address, "remote": True}))
    cfg = settings.load()
    eng = engine.Engine(cfg)
    eng._client.close()
    eng._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        monkeypatch.setattr(recipes, "served_window", lambda _: None)
        app = FastAPI()
        server_settings.install(app, eng, gate.Gate(), cfg, lambda: settings.save(cfg), cfg.to_dict)
        server_connect.install(app, eng, cfg, None)
        with TestClient(app) as client:
            if source == "settings-api":
                saved = client.put("/api/settings", json={"engine_url": address, "remote": True})
                assert saved.status_code == 200
            assert settings.load().engine_url == (address.rstrip("/") if source == "settings-api" else address)
            health = eng.health()
            connect = client.get("/api/connect").json()
            root = "http://fixture.test" + prefix
            assert connect["engine_url"] == root
            jan = next(r for r in connect["snippets"] if r["id"] == "jan")
            assert (health.status, health.model, jan["snippet"].splitlines()[1]) == (
                "running", "fixture", "Base URL   " + root + "/v1"
            )
    finally:
        eng._client.close()


@pytest.mark.parametrize(("address", "expected"), [
    ("http://v1", "http://v1"),
    ("http://v1/v1/", "http://v1"),
    ("https://fixture.test/proxy/v10", "https://fixture.test/proxy/v10"),
    ("https://[::1]:8080/proxy/v1", "https://[::1]:8080/proxy"),
])
def test_engine_url_normalizes_the_api_path_without_changing_the_authority(address, expected):
    assert settings.engine_root(address) == expected
