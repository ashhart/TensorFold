import json
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from tensorfold.server.http import make_handler
from tests.lane_fakes import FakeEngine
from tests.test_lane_server import make_app


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("cancellation did not reach its safe boundary")


class AuditEngine(FakeEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.audit = []

    def add_stream(self, stream, **kwargs):
        self.audit.append(stream)
        return super().add_stream(stream, **kwargs)

    def _family_prefill(self, stream, **kwargs):
        self.entered.set()
        assert self.release.wait(3)
        if self.prefill_guard is not None:
            self.prefill_guard.before_chunk([], len(stream.prompt_ids))
        return super()._family_prefill(stream, **kwargs)

    def step(self):
        time.sleep(.004)
        return super().step()


def app():
    result = make_app(engine_factory=AuditEngine, lanes=1, checkpoint_slots=0)
    result.stop_ids = result.scheduler.eos_ids = frozenset({-1})
    return result


MESSAGES = [{"role": "user", "content": "Write a greeting."}]


def client(server, stream=True):
    body = json.dumps({"messages": MESSAGES, "max_tokens": 40, "stream": stream,
                       "tools": [{"type": "function", "function": {"name": "echo",
                                  "parameters": {"type": "object", "properties": {}}}}]}).encode()
    connection = socket.create_connection(server.server_address, timeout=2)
    headers = ("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
               f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
    connection.sendall(headers + body)
    if stream:
        response = b""
        while b"\r\n\r\n" not in response:
            response += connection.recv(4096)
        assert b"200 OK" in response
    return connection


def serve(instance):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(instance))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    return server


def test_callback_disconnect_stops_active_generation_and_recovers():
    instance = app()
    try:
        def disconnected(delta):
            raise BrokenPipeError("consumer disconnected")

        with pytest.raises(BrokenPipeError):
            instance.chat(MESSAGES, max_tokens=40, on_delta=disconnected)
        until(lambda: instance.scheduler.active == 0 and instance.engine.active_count == 0)
        assert len(instance.engine.audit[0].emitted) < 40
        assert instance.requests_completed == 0
        recovered = instance.chat(MESSAGES, max_tokens=4)
        assert recovered["completion_tokens"] == 4
        assert instance.scheduler._thread.is_alive()
    finally:
        instance.close()


def test_live_consumer_still_receives_the_complete_reply():
    instance = app()
    try:
        reply = instance.chat(MESSAGES, max_tokens=40, on_delta=lambda delta: None)
        assert reply["completion_tokens"] == 40
        assert len(instance.engine.audit[0].emitted) == 40
        assert instance.requests_completed == instance.scheduler.completed == 1
        assert getattr(instance.scheduler, "cancelled", 0) == 0
    finally:
        instance.close()


def test_queued_socket_disconnects_are_removed_before_prefill():
    instance = app()
    instance.engine.release.clear()
    server = serve(instance)
    connections = []
    try:
        connections.append(client(server))
        assert instance.engine.entered.wait(2)
        connections.extend([client(server), client(server)])
        until(lambda: instance.scheduler._queue.qsize() == 2)
        for connection in connections[1:]:
            connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        until(lambda: instance.scheduler._queue.qsize() == 0)
        assert len(instance.engine.audit) == 1
        instance.engine.release.set()
        connections[0].close()
        until(lambda: instance.scheduler.active == 0 and instance.engine.active_count == 0)
        assert len(instance.engine.audit) == 1
        assert instance.chat(MESSAGES, max_tokens=4)["completion_tokens"] == 4
    finally:
        instance.engine.release.set()
        for connection in connections:
            connection.close()
        server.shutdown()
        server.server_close()
        instance.close()


def test_streaming_socket_disconnect_during_decode_releases_the_active_row():
    instance = app()
    server = serve(instance)
    connection = None
    try:
        connection = client(server)
        response = b""
        while b'"content"' not in response:
            response += connection.recv(4096)
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        until(lambda: instance.scheduler.active == 0 and instance.engine.active_count == 0)
        assert len(instance.engine.audit[0].emitted) < 40
        assert instance.requests_completed == 0
        assert instance.chat(MESSAGES, max_tokens=4)["completion_tokens"] == 4
    finally:
        if connection is not None:
            connection.close()
        server.shutdown()
        server.server_close()
        instance.close()


@pytest.mark.parametrize("stream", [True, False])
def test_socket_disconnect_in_prefill_stops_before_next_chunk(stream):
    instance = app()
    instance.engine.release.clear()
    server = serve(instance)
    connection = None
    try:
        connection = client(server, stream)
        assert instance.engine.entered.wait(2)
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        time.sleep(.15)
        instance.engine.release.set()
        until(lambda: instance.scheduler._starting is None and instance.scheduler.active == 0)
        assert not instance.engine.audit[0].emitted
        assert not instance.engine.prefill_calls
        assert instance.engine.prefill_guard is None
        assert instance.chat(MESSAGES, max_tokens=4)["completion_tokens"] == 4
    finally:
        instance.engine.release.set()
        if connection is not None:
            connection.close()
        server.shutdown()
        server.server_close()
        instance.close()
