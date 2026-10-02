"""With a server key, every CUDA route but /health wants it (as a bearer token or X-API-Key); without one, none do."""

import http.client
import json

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for, serving

WAIT = 10


def call(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        data = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


def test_a_server_key_guards_every_route_but_health(tmp_path):
    app = app_for(tmp_path, PacedEngine())
    app.api_key = "s3cret"
    with serving(app) as port:
        assert call(port, "GET", "/health")[0] == 200
        status, body = call(port, "GET", "/v1/models")
        assert status == 401 and body["error"]["code"] == "invalid_api_key"
        assert call(port, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"})[0] == 401
        request = {"messages": MESSAGES, "max_tokens": 2}
        assert call(port, "POST", "/v1/chat/completions", request)[0] == 401
        assert call(port, "GET", "/v1/models", headers={"Authorization": "Bearer s3cret"})[0] == 200
        assert call(port, "GET", "/v1/models", headers={"X-API-Key": "s3cret"})[0] == 200
        status, body = call(port, "POST", "/v1/chat/completions", request, {"Authorization": "bearer s3cret"})
        assert status == 200 and body["choices"][0]["message"]["content"] is not None


def test_no_server_key_means_open_routes(tmp_path):
    app = app_for(tmp_path, PacedEngine())
    with serving(app) as port:
        assert call(port, "GET", "/v1/models")[0] == 200


def test_tokenize_counts_the_rendered_prompt(tmp_path):
    app = app_for(tmp_path, PacedEngine())
    with serving(app) as port:
        status, body = call(port, "POST", "/tokenize", {"messages": MESSAGES, "add_generation_prompt": True})
        assert status == 200 and body["count"] == len(body["tokens"]) > 0
        assert call(port, "POST", "/v1/tokenize", {"prompt": "hello"})[0] == 200
        assert call(port, "POST", "/tokenize", {})[0] == 400
