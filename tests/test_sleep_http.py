"""Sleep control and whole-request admission over both real HTTP handlers, without device runtimes."""

from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server.http import make_handler as mac_handler
from tensorfold.server.lifecycle import Lifecycle
from tests.test_token_routes import Engine, MacApp, cuda_app

AUTH = {"Authorization": "Bearer test-sleep-secret"}
CHAT = {"model": "fake-model", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2}


@pytest.fixture(params=["cuda", "mac"])
def app(request, tmp_path):
    instance = cuda_app(tmp_path, window=128) if request.param == "cuda" else MacApp()
    instance.test_backend = request.param
    instance.reasoning_effort = None
    instance.default_thinking = False
    instance.context_window = 128
    instance.engine = Engine()
    instance.vision = None
    instance.sleep_token = "test-sleep-secret"
    instance.transitions = []

    def release():
        instance.transitions.append("sleep")
        instance.engine = instance.vision = None

    def restore():
        instance.transitions.append("wake")
        instance.engine = Engine()

    instance.lifecycle = Lifecycle(release=release, restore=restore, cleanup=release, drain_timeout=2)
    return instance


@contextmanager
def serving(app):
    from http.server import ThreadingHTTPServer

    handler = (cuda_handler if app.test_backend == "cuda" else mac_handler)(app)
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(3)


def call(server, path, *, method="POST", body=None, headers=AUTH):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        data = response.read().decode()
        return response.status, json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
    finally:
        connection.close()


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "HTTP lifecycle transition did not arrive"
        time.sleep(0.005)


def test_admin_routes_are_authenticated_idempotent_and_keep_discovery_available(app):
    with serving(app) as server:
        for path in ("/sleep?level=2", "/wake_up", "/is_sleeping"):
            method = "GET" if path == "/is_sleeping" else "POST"
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer "}):
                status, body = call(server, path, method=method, headers=headers)
                assert status == 401 and body["error"]["code"] == "invalid_sleep_token"
            assert call(server, path, method=method, headers={**AUTH, "Origin": "https://example.invalid"})[0] == 403
        assert app.transitions == []
        assert call(server, "/is_sleeping", method="GET")[1]["state"] == "awake"
        for _ in range(2):
            status, body = call(server, "/sleep?level=2", body={"ignored": True})
            assert status == 200 and body["is_sleeping"] and not body["ready"]
        assert app.transitions == ["sleep"]
        status, health = call(server, "/health", method="GET", headers={})
        assert status == 200 and health["lifecycle"]["state"] == "sleeping" and not health["ready"]
        assert call(server, "/v1/models", method="GET", headers={})[0] == 200
        status, metrics = call(server, "/metrics", method="GET", headers={})
        assert status == 200 and "tensorfold:model_ready 0" in metrics
        for _ in range(2):
            assert call(server, "/wake_up")[1]["state"] == "awake"
        assert app.transitions == ["sleep", "wake"]


@pytest.mark.parametrize("prefix", ["", "/v1"])
def test_sleep_credentials_stay_separate_from_api_keys(app, prefix):
    from tensorfold.control.telemetry import Client
    from tensorfold.server.authentication import KeyStore

    app.auth = KeyStore(["test-inference-secret"])
    inference = {"Authorization": "Bearer test-inference-secret"}
    with serving(app) as server:
        for route, method in (("/sleep", "POST"), ("/wake_up", "POST"), ("/is_sleeping", "GET")):
            for headers in ({}, inference):
                status, body = call(server, prefix + route, method=method, headers=headers)
                assert status == 401 and body["error"]["code"] == "invalid_sleep_token"
        assert app.transitions == []
        assert call(server, "/v1/models", method="GET")[0] == 401
        assert call(server, "/v1/models", method="GET", headers=inference)[0] == 200
        assert call(server, prefix + "/sleep", headers={**AUTH, "Origin": "https://example.invalid"})[0] == 403
        assert call(server, prefix + "/sleep")[1]["is_sleeping"]
        assert call(server, prefix + "/is_sleeping", method="GET")[1]["state"] == "sleeping"
        assert call(server, "/health", method="GET", headers={}) == (200, {"status": "ok"})
        sample = Client(f"http://127.0.0.1:{server.server_port}", "test-inference-secret").sample()
        assert sample.online and sample.phase == "sleeping"
        assert call(server, "/metrics", method="GET", headers={})[0] == 401
        status, metrics = call(server, "/metrics", method="GET", headers=inference)
        assert status == 200 and "tensorfold:model_ready 0" in metrics
        assert 'requests_total{key="cli-1",status="200"}' in metrics
        assert call(server, "/v1/chat/completions", body=CHAT, headers=inference)[0] == 503
        assert call(server, prefix + "/wake_up")[1]["ready"]
        assert app.transitions == ["sleep", "wake"]


@pytest.mark.parametrize("path", ["/sleep?level=1", "/sleep?level=", "/sleep?level=no", "/sleep?level=2&level=1"])
def test_bad_sleep_levels_leave_the_runtime_awake(app, path):
    with serving(app) as server:
        status, body = call(server, path)
        assert status == 400 and body["error"]["type"] == "invalid_request_error"
        assert app.lifecycle.snapshot()["state"] == "awake" and not app.transitions


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages",
                                  "/v1/messages/count_tokens", "/tokenize", "/detokenize", "/v1/decisions"])
def test_all_inference_posts_are_refused_before_preparation(app, path):
    app.lifecycle.sleep()
    with serving(app) as server:
        status, body = call(server, path, body={})
        assert status == 503 and body["error"]["code"] == "model_not_ready"


def test_disabled_controls_are_not_exposed(app):
    del app.lifecycle
    with serving(app) as server:
        assert call(server, "/sleep?level=2")[0] == 404
        assert call(server, "/wake_up")[0] == 404
        assert call(server, "/is_sleeping", method="GET")[0] == 404
        assert call(server, "/v1/chat/completions", body=CHAT)[0] == 200


def test_api_auth_still_gates_disabled_and_unknown_controls(app):
    from tensorfold.server.authentication import KeyStore

    app.auth = KeyStore(["test-inference-secret"])
    inference = {"Authorization": "Bearer test-inference-secret"}
    with serving(app) as server:
        for path in ("/v1/sleep/unknown", "/v1/is_sleeping"):
            assert call(server, path)[0] == 401
            assert call(server, path, headers=inference)[0] == 404
        del app.lifecycle
        for path in ("/v1/sleep", "/v1/wake_up"):
            assert call(server, path)[0] == 401
            assert call(server, path, headers=inference)[0] == 404
        assert not app.transitions


@pytest.mark.parametrize("path", ["/unknown", "/unknown/?unused=1"])
@pytest.mark.parametrize("framing", ["Content-Length: 1", "Transfer-Encoding: chunked"])
def test_unauthenticated_unknown_post_cannot_hold_sleep_admission(app, path, framing):
    from tensorfold.server.authentication import KeyStore

    app.auth = KeyStore(["test-inference-secret"])
    with serving(app) as server, socket.create_connection(("127.0.0.1", server.server_port), timeout=1) as client:
        client.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\n{framing}\r\n\r\n".encode())
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 404 and response.getheader("Connection") == "close"
        response.read()
        assert app.lifecycle.snapshot()["active_requests"] == 0
        assert call(server, "/sleep")[1]["is_sleeping"]
        assert app.transitions == ["sleep"]


def test_empty_configured_secret_never_authenticates(app):
    app.sleep_token = ""
    with serving(app) as server:
        assert call(server, "/sleep", headers={"Authorization": "Bearer "})[0] == 401
        assert not app.transitions


def test_admin_bad_framing_never_changes_state(app):
    with serving(app) as server, socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as client:
        client.sendall(b"POST /sleep?level=2 HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer test-sleep-secret\r\n"
                       b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n0\r\n\r\n")
        client.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 400 and response.getheader("Connection") == "close"
        assert json.loads(response.read())["error"]["type"] == "invalid_request_error"
        assert not app.transitions


def test_chunked_control_consumes_its_body_before_the_next_request(app):
    with serving(app) as server:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("POST", "/sleep?level=2", iter([b"{}"]), AUTH, encode_chunked=True)
            response = connection.getresponse()
            assert response.status == 200 and json.loads(response.read())["is_sleeping"]
            original = connection.sock
            connection.request("GET", "/is_sleeping", headers=AUTH)
            response = connection.getresponse()
            assert response.status == 200 and json.loads(response.read())["is_sleeping"]
            assert connection.sock is original and original is not None
        finally:
            connection.close()


def test_an_accepted_stream_finishes_before_sleep_and_new_requests_are_refused(app, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    name = "run" if app.test_backend == "cuda" else "chat"
    original = getattr(app, name)

    def paused(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(app, name, paused)
    results = {}
    with serving(app) as server:
        stream = threading.Thread(target=lambda: results.update(stream=call(
            server, "/v1/chat/completions", body={**CHAT, "stream": True})))
        sleeper = threading.Thread(target=lambda: results.update(sleep=call(server, "/sleep?level=2")))
        stream.start()
        try:
            assert entered.wait(3)
            sleeper.start()
            until(lambda: app.lifecycle.snapshot()["state"] == "draining")
            assert app.lifecycle.snapshot()["active_requests"] == 1
            assert app.engine is not None and not app.transitions
            assert call(server, "/v1/chat/completions", body=CHAT)[0] == 503
            assert call(server, "/sleep?level=2")[0] == 409
            assert call(server, "/wake_up")[0] == 409
            status, body = call(server, "/health", method="GET")
            assert status == 200 and body["lifecycle"]["state"] == "draining"
        finally:
            finish.set()
            stream.join(3)
            if sleeper.ident is not None:
                sleeper.join(3)
        assert results["stream"][0] == 200 and "data: [DONE]" in results["stream"][1]
        assert results["sleep"][0] == 200 and app.engine is None


@pytest.mark.parametrize("api", ["responses", "anthropic"])
def test_accepted_translation_can_enter_its_nested_chat_during_drain(app, monkeypatch, api):
    from tensorfold.server import anthropic, responses

    module = responses if api == "responses" else anthropic
    original = module.translate
    entered, finish = threading.Event(), threading.Event()

    def paused(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "translate", paused)
    body = {"input": "Hi", "max_output_tokens": 2} if api == "responses" else CHAT
    path = "/v1/responses" if api == "responses" else "/v1/messages"
    results = {}
    with serving(app) as server:
        requester = threading.Thread(target=lambda: results.update(reply=call(server, path, body=body)))
        sleeper = threading.Thread(target=lambda: results.update(sleep=call(server, "/sleep")))
        requester.start()
        try:
            assert entered.wait(3)
            sleeper.start()
            until(lambda: app.lifecycle.snapshot()["state"] == "draining")
            assert app.lifecycle.snapshot()["active_requests"] == 1
            assert call(server, path, body=body)[0] == 503
        finally:
            finish.set()
            requester.join(3)
            if sleeper.ident is not None:
                sleeper.join(3)
        assert results["reply"][0] == 200, results
        assert results["sleep"][0] == 200


def test_responses_history_and_metrics_survive_runtime_replacement(app):
    from tensorfold.server import metrics, responses

    metrics.note(app, prompt=7, generation=3)
    old_engine = app.engine
    with serving(app) as server:
        status, first = call(server, "/v1/responses", body={"input": "Hi", "max_output_tokens": 2})
        assert status == 200, first
        store = responses.store_for(app)
        before = metrics.of(app)
        assert call(server, "/sleep")[0] == 200
        status, stored = call(server, f"/v1/responses/{first['id']}", method="GET")
        assert status == 200 and stored == first
        assert call(server, "/wake_up")[0] == 200
        assert app.engine is not old_engine
        assert metrics.of(app) is before and responses.store_for(app) is store
        body = {"input": "Again", "previous_response_id": first["id"], "max_output_tokens": 2}
        status, second = call(server, "/v1/responses", body=body)
        assert status == 200 and second["previous_response_id"] == first["id"], second
        conversation = store.conversation(second["id"])
        assert len(conversation) == 4
        assert conversation[0]["content"] == "Hi" and conversation[2]["content"] == "Again"


def test_waking_is_visible_and_refuses_new_requests(app):
    entered, finish = threading.Event(), threading.Event()

    def restore():
        entered.set()
        assert finish.wait(3)
        app.engine = Engine()

    app.lifecycle = Lifecycle(release=lambda: setattr(app, "engine", None), restore=restore, cleanup=lambda: None)
    app.lifecycle.sleep()
    results = {}
    with serving(app) as server:
        waking = threading.Thread(target=lambda: results.update(wake=call(server, "/wake_up")))
        waking.start()
        try:
            assert entered.wait(3)
            assert call(server, "/health", method="GET")[1]["lifecycle"]["state"] == "waking"
            assert call(server, "/v1/chat/completions", body=CHAT)[0] == 503
            assert call(server, "/wake_up")[0] == 409
            assert call(server, "/sleep")[0] == 409
        finally:
            finish.set()
            waking.join(3)
        assert results["wake"][0] == 200


@pytest.mark.parametrize("route", ["/health", "/metrics"])
def test_runtime_observation_drains_before_release_and_never_runs_asleep(app, monkeypatch, route):
    from tensorfold.cuda import health
    from tensorfold.server import http, metrics

    entered, finish = threading.Event(), threading.Event()
    if route == "/metrics":
        owner, method = metrics, "_pools"
    elif app.test_backend == "cuda":
        owner, method = health.of(app), "_runtime"
    else:
        owner, method = http, "_memory"
    original = getattr(owner, method)

    def paused(*args, **kwargs):
        assert app.lifecycle.snapshot()["state"] != "sleeping"
        entered.set()
        assert finish.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, paused)
    results = {}
    with serving(app) as server:
        observing = threading.Thread(target=lambda: results.update(observe=call(server, route, method="GET")))
        sleeper = threading.Thread(target=lambda: results.update(sleep=call(server, "/sleep")))
        observing.start()
        try:
            assert entered.wait(3)
            sleeper.start()
            until(lambda: app.lifecycle.snapshot()["state"] == "draining")
            assert app.engine is not None and not app.transitions
            assert call(server, route, method="GET")[0] == 200
        finally:
            finish.set()
            observing.join(3)
            if sleeper.ident is not None:
                sleeper.join(3)
        assert results["observe"][0] == results["sleep"][0] == 200
        assert app.engine is None and call(server, route, method="GET")[0] == 200


def test_fatal_lifecycle_error_stops_the_actual_cuda_server_and_raises(monkeypatch):
    import signal
    from types import SimpleNamespace
    from tensorfold.cuda import http

    def fail():
        raise RuntimeError("injected teardown failure")

    app = SimpleNamespace(lifecycle=Lifecycle(release=fail, restore=lambda: None, cleanup=lambda: None),
                          sleep_token="test-sleep-secret")
    opened = threading.Event()
    results = {}
    original = http.Server

    def create(*args, **kwargs):
        results["server"] = original(*args, **kwargs)
        opened.set()
        return results["server"]

    monkeypatch.setattr(http, "Server", create)
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    def run():
        try:
            http.serve(app, "127.0.0.1", 0)
        except RuntimeError as exc:
            results["error"] = str(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert opened.wait(3)
    server = results["server"]
    try:
        status, body = call(server, "/sleep")
        assert status == 500 and body["error"]["type"] == "server_error"
        worker.join(3)
        assert not worker.is_alive()
        assert results["error"] == "model lifecycle failed; restart the server"
        assert server.socket.fileno() == -1
    finally:
        if worker.is_alive():
            server.shutdown()
            worker.join(3)


def test_lifecycle_routes_include_allocator_counters_when_supported(app):
    app.sleep_memory = lambda: {"allocated": 0 if app.engine is None else 100, "reserved": 100}
    with serving(app) as server:
        assert call(server, "/v1/is_sleeping", method="GET")[1]["memory"]["allocated"] == 100
        assert call(server, "/v1/sleep?level=2")[1]["memory"]["allocated"] == 0
        assert call(server, "/v1/wake_up")[1]["memory"]["allocated"] == 100


def test_lifecycle_routes_include_cache_status_when_supported(app):
    app.sleep_cache = lambda: {"enabled": True, "stored": app.engine is None}
    with serving(app) as server:
        assert call(server, "/v1/is_sleeping", method="GET")[1]["cache"] == {"enabled": True, "stored": False}
        assert call(server, "/v1/sleep?level=2")[1]["cache"] == {"enabled": True, "stored": True}
        assert call(server, "/v1/wake_up")[1]["cache"] == {"enabled": True, "stored": False}


def test_lifecycle_routes_omit_cache_status_without_a_callback(app):
    with serving(app) as server:
        for path, method in (("/is_sleeping", "GET"), ("/sleep", "POST"), ("/wake_up", "POST")):
            status, body = call(server, path, method=method)
            assert status == 200 and "cache" not in body


@pytest.mark.parametrize("mode, expected", [("unauthorized", 401), ("origin", 403), ("sleeping", 503)])
def test_refusal_closes_without_reading_a_stalled_body_or_dispatching_pipeline(app, mode, expected):
    if mode == "sleeping":
        app.lifecycle.sleep()
    headers = "Origin: https://example.invalid\r\n" if mode == "origin" else ""
    route = "/v1/chat/completions" if mode == "sleeping" else "/sleep"
    with serving(app) as server, socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as client:
        client.sendall(f"POST {route} HTTP/1.1\r\nHost: x\r\n{headers}Content-Length: 999999999\r\n\r\n"
                       "{}GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == expected and response.getheader("Connection") == "close"
        response.read()
        assert client.recv(1) == b""
        assert app.transitions == (["sleep"] if mode == "sleeping" else [])


def test_failed_restore_is_reported_and_a_retry_can_wake_the_same_server(app):
    attempts = []

    def restore():
        attempts.append("restore")
        if len(attempts) == 1:
            raise RuntimeError("injected load failure")
        app.engine = Engine()

    app.lifecycle = Lifecycle(release=lambda: setattr(app, "engine", None), restore=restore, cleanup=lambda: None)
    with serving(app) as server:
        assert call(server, "/sleep")[0] == 200
        status, body = call(server, "/wake_up")
        assert status == 503 and body["error"]["code"] == "model_wake_failed"
        assert call(server, "/is_sleeping", method="GET")[1]["state"] == "sleeping"
        assert not getattr(server, "lifecycle_failed", False)
        assert call(server, "/wake_up")[0] == 200
        assert call(server, "/v1/chat/completions", body=CHAT)[0] == 200


def test_normal_cuda_serve_interrupt_still_returns_cleanly(monkeypatch):
    import signal
    from types import SimpleNamespace
    from tensorfold.cuda import http

    class InterruptedServer:
        closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    server = InterruptedServer()
    monkeypatch.setattr(http, "Server", lambda *args: server)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    assert http.serve(SimpleNamespace(), "127.0.0.1", 0) is None
    assert server.closed
