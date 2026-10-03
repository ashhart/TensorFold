"""New API routes use the same HTTP framing as chat, with CPU-only test engines."""

import http.client
import json
import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server.http import make_handler as mac_handler
from tests.test_token_routes import MacApp, cuda_app

CHAT = {"model": "local-model", "max_tokens": 8,
        "messages": [{"role": "user", "content": "Hello \u00e9"}]}
ROUTES = [
    ("/v1/messages", CHAT, "content"),
    ("/v1/messages/count_tokens", CHAT, "input_tokens"),
    ("/tokenize", {"prompt": "Hello \u00e9"}, "tokens"),
    ("/detokenize", {"tokens": [104, 105]}, "prompt"),
]


@pytest.fixture(params=["cuda", "mac"])
def auxiliary_port(request, tmp_path):
    app = cuda_app(tmp_path) if request.param == "cuda" else MacApp()
    app.reasoning_effort = None
    handler = (cuda_handler if request.param == "cuda" else mac_handler)(app)
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("path, body, key", ROUTES)
def test_chunked_auxiliary_routes_match_fixed_body_and_keep_the_connection(auxiliary_port, path, body, key):
    connection = http.client.HTTPConnection("127.0.0.1", auxiliary_port, timeout=5)
    raw = json.dumps(body, ensure_ascii=False).encode()
    try:
        connection.request("POST", path, raw, {"Content-Type": "application/json"})
        response = connection.getresponse()
        fixed = json.loads(response.read())
        assert response.status == 200, fixed
        original = connection.sock
        # Each UTF-8 byte may arrive in its own chunk; a trailer must also be drained.
        chunks = b"".join(b"1;name=ignored\r\n" + bytes([byte]) + b"\r\n" for byte in raw)
        chunks += b"0\r\nX-Test: ignored\r\n\r\n"
        connection.request("POST", path, chunks, {"Transfer-Encoding": "chunked"})
        response = connection.getresponse()
        chunked = json.loads(response.read())
        assert response.status == 200, chunked
        assert chunked[key] == fixed[key]
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        assert response.status == 200 and json.loads(response.read())["data"]
        assert connection.sock is original and original is not None
    finally:
        connection.close()


def test_chunked_anthropic_stream_completes(auxiliary_port):
    connection = http.client.HTTPConnection("127.0.0.1", auxiliary_port, timeout=5)
    raw = json.dumps({**CHAT, "stream": True}).encode()
    try:
        connection.request("POST", "/v1/messages", iter(raw[i:i + 1] for i in range(len(raw))),
                           {"Content-Type": "application/json"}, encode_chunked=True)
        response = connection.getresponse()
        stream = response.read().decode()
        assert response.status == 200, stream
        assert "event: message_start\n" in stream and "event: message_stop\n" in stream
        assert "event: error\n" not in stream
    finally:
        connection.close()


@pytest.mark.parametrize("path, body, key", ROUTES)
@pytest.mark.parametrize("headers, wire", [
    ("Transfer-Encoding: chunked\r\nContent-Length: 2", b"0\r\n\r\n"),
    ("Content-Length: 2\r\nContent-Length: 3", b"{}"),
    ("Content-Length: 20", b"{}"),
    ("Transfer-Encoding: chunked", b"+2\r\n{}\r\n0\r\n\r\n"),
])
def test_auxiliary_routes_close_on_bad_framing(auxiliary_port, path, body, key, headers, wire):
    with socket.create_connection(("127.0.0.1", auxiliary_port), timeout=5) as client:
        client.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\n{headers}\r\n\r\n".encode() + wire)
        client.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(client)
        response.begin()
        error = json.loads(response.read())
        assert response.status == 400, error
        assert response.getheader("Connection") == "close"
        assert error["error"]["type"] == "invalid_request_error"
        assert client.recv(1) == b""
