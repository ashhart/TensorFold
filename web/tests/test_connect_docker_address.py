from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorfold_web import recipes, server_connect, settings


@pytest.mark.parametrize(("engine_url", "params", "expected"), [
    pytest.param("https://fixture.test:9443/proxy", {}, "https://fixture.test:9443/proxy/v1", id="remote-https-proxy"),
    pytest.param("http://fixture.test", {}, "http://fixture.test/v1", id="remote-http-default-port"),
    pytest.param("https://fixture.test/proxy", {}, "https://fixture.test/proxy/v1", id="remote-https-default-port"),
    pytest.param("http://[::1]:8080", {}, "http://host.docker.internal:8080/v1", id="ipv6-loopback"),
    pytest.param("https://fixture.test:9443/proxy", {"where": "away", "relay": "https://web.test"},
                 "https://web.test/engine/v1", id="away-relay"),
])
def test_connect_docker_recipes_keep_the_engine_address(monkeypatch, engine_url, params, expected):
    monkeypatch.setattr(recipes, "served_window", lambda _: 32768)
    engine = SimpleNamespace(
        last_health=SimpleNamespace(status="running", model="fixture"),
        key_in_use=lambda: None,
        key_pending_restart=lambda: False,
    )
    app = FastAPI()
    server_connect.install(app, engine, settings.Settings(engine_url=engine_url, remote=True), None)

    with TestClient(app) as client:
        response = client.get("/api/connect", params=params)

    assert response.status_code == 200
    snippets = {row["id"]: row["snippet"].splitlines() for row in response.json()["snippets"]}
    assert f'      baseURL: "{expected}"' in snippets["librechat"]
    assert f"GENERIC_OPEN_AI_BASE_PATH={expected}" in snippets["anythingllm"]
