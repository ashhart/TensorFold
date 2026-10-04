"""Authentication reaches the real HTTP handlers before application work."""

from contextlib import contextmanager
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server.http import Server, make_handler as mlx_handler


@contextmanager
def serving(backend, auth=None):
    app = SimpleNamespace(served="fixture", served_name="fixture", model_ids=["fixture"], max_batch_size=1)
    app.auth = auth
    server = Server(("127.0.0.1", 0), (cuda_handler if backend == "cuda" else mlx_handler)(app))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield app, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def request(port, path, headers=None, method="GET", body=None):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        client.close()


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_rejected_key_cannot_reach_model_routes(backend):
    policy = SimpleNamespace(enabled=True, metrics_open=False, match=lambda headers: None)
    with serving(backend, policy) as (_, port):
        code, headers, body = request(port, "/v1/models", {"Authorization": "Bearer fixture-wrong"})
    assert code == 401
    assert headers["WWW-Authenticate"] == "Bearer"
    assert json.loads(body)["error"]["code"] == "invalid_api_key"


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_gated_health_has_no_model_details(backend):
    policy = SimpleNamespace(enabled=True, metrics_open=False, match=lambda headers: None)
    with serving(backend, policy) as (_, port):
        code, _, body = request(port, "/health")
    assert code == 200 and json.loads(body) == {"status": "ok"}


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("path", ["/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/responses",
                                 "/v1/decisions", "/tokenize", "/detokenize", "/metrics", "/v1/new-route",
                                 "/chat/completions", "/decisions", "/messages", "/models",
                                 "/alternative/chat/completions", "/alternative/decisions", "/alternative/models"])
def test_every_inference_route_refuses_before_parsing(backend, path):
    from tensorfold.server.authentication import KeyStore

    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        code, headers, body = request(port, path, method="POST", body=b"not json")
    assert code == 401 and headers["WWW-Authenticate"] == "Bearer"
    error = json.loads(body)["error"]
    assert error.get("type") == "authentication_error" if path == "/messages" else error["code"] == "invalid_api_key"


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/count_tokens?x=1"])
def test_message_routes_use_their_authentication_error_shape(backend, path):
    from tensorfold.server.authentication import KeyStore

    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        code, headers, body = request(port, path, {"x-api-key": "fixture-wrong"}, "POST", b"{}")
    assert code == 401 and headers["WWW-Authenticate"] == "Bearer"
    payload = json.loads(body)
    assert payload["type"] == "error" and payload["error"]["type"] == "authentication_error"


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("header", ["Authorization", "x-api-key"])
def test_valid_keys_and_labels_reach_the_route(backend, header, tmp_path, capsys):
    from tensorfold.server.authentication import KeyStore

    path = tmp_path / "keys"
    path.write_text("client: fixture-key\n")
    path.chmod(0o600)
    value = "Bearer fixture-key" if header == "Authorization" else "fixture-key"
    with serving(backend, KeyStore(key_file=path)) as (_, port):
        assert request(port, "/v1/models", {header: value})[0] == 200
        status, _, body = request(port, "/metrics", {header: value})
    assert status == 200 and 'requests_total{key="client",status="200"} 1' in body
    output = capsys.readouterr().out
    assert "key=client" in output and "fixture-key" not in output


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_metrics_can_open_and_no_keys_preserve_open_routes(backend):
    from tensorfold.server.authentication import KeyStore

    for policy in (None, KeyStore(), KeyStore(["fixture-key"], metrics_open=True)):
        with serving(backend, policy) as (_, port):
            assert request(port, "/metrics")[0] == 200
            if policy is None or not policy.enabled:
                assert request(port, "/v1/models")[0] == 200


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_options_is_unaffected(backend):
    from tensorfold.server.authentication import KeyStore

    with serving(backend) as (_, port):
        before = request(port, "/v1/models", method="OPTIONS")
    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        after = request(port, "/v1/models", method="OPTIONS")
    assert before[0] == after[0] == 501
    assert before[2] == after[2]


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("reload_kind", ["mtime", "sighup"])
def test_rotation_changes_access_on_an_existing_server(backend, reload_kind, tmp_path):
    import os
    import signal
    from tensorfold.server.authentication import KeyStore

    now = [0.0]
    path = tmp_path / "keys"
    path.write_text("first: fixture-first\n")
    path.chmod(0o600)
    store = KeyStore(key_file=path, clock=lambda: now[0])
    with store.signals(), serving(backend, store) as (_, port):
        assert request(port, "/v1/models", {"x-api-key": "fixture-first"})[0] == 200
        path.write_text("next: fixture-next\n")
        if reload_kind == "sighup":
            os.kill(os.getpid(), signal.SIGHUP)
        else:
            now[0] = 1.1
        assert request(port, "/v1/models", {"x-api-key": "fixture-first"})[0] == 401
        assert request(port, "/v1/models", {"x-api-key": "fixture-next"})[0] == 200


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_a_pooled_connection_does_not_inherit_authentication(backend):
    from tensorfold.server.authentication import KeyStore

    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("GET", "/v1/models", headers={"x-api-key": "fixture-key"})
            first = connection.getresponse()
            assert first.status == 200
            first.read()
            connection.request("GET", "/v1/models")
            second = connection.getresponse()
            assert second.status == 401
            second.read()
        finally:
            connection.close()


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_authenticated_replies_and_translated_handlers_count_once(backend, path, tmp_path):
    from tensorfold.server.authentication import KeyStore
    from tensorfold.server import metrics
    from tests.test_server_openai_compat import FakeApp
    from tests.test_cuda_server_disconnect import PacedEngine, app_for

    app = FakeApp() if backend == "mlx" else app_for(tmp_path, PacedEngine())
    app.auth = KeyStore(["fixture-key"])
    httpd = Server(("127.0.0.1", 0), (mlx_handler if backend == "mlx" else cuda_handler)(app))
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    body = {"model": "fixture", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2}
    if path == "/v1/responses":
        body = {"input": "Hello", "max_output_tokens": 2, "store": False}
    try:
        code, _, content = request(httpd.server_port, path, {"Authorization": "Bearer fixture-key"},
                                   "POST", json.dumps(body))
        assert code == 200, content
        assert metrics.of(app).http_requests == {("cli-1", 200): 1}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_authentication_refuses_before_expect_continue(backend):
    from tensorfold.server.authentication import KeyStore

    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Content-Length", "100")
            connection.putheader("Expect", "100-continue")
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 401
            assert response.getheader("WWW-Authenticate") == "Bearer"
            response.read()
        finally:
            connection.close()
