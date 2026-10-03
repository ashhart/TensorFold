"""HTTP framing is decoded before either backend sees JSON; no models or GPUs."""

import http.client
import json
import socket
import threading
from email.message import Message
from http.server import ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest

from tests.test_server_refused_bodies import port  # noqa: F401 - both real HTTP handlers


@pytest.mark.parametrize("path, fields", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello \u00e9"}]}),
    ("/v1/completions", {"prompt": "Hello \u00e9"}),
    ("/v1/responses", {"input": "Hello \u00e9"}),
])
@pytest.mark.parametrize("stream", [False, True])
def test_chunked_json_routes(port, path, fields, stream):
    body = json.dumps({**fields, "stream": stream, "max_tokens": 8}, ensure_ascii=False).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        # Split inside JSON tokens and UTF-8 characters, as a transforming proxy can.
        connection.request("POST", path, iter(body[i:i + 1] for i in range(len(body))),
                           {"Content-Type": "application/json"}, encode_chunked=True)
        response = connection.getresponse()
        data = response.read().decode()
        assert response.status == 200, data
        if stream:
            assert "response.completed" in data if path.endswith("responses") else "data: [DONE]" in data
        else:
            assert "Hello" in data
    finally:
        connection.close()


@pytest.mark.parametrize("path, expected", [
    ("/v1/chat/completions", 200), ("/unknown", 404),
])
def test_extensions_trailers_and_the_next_request(port, path, expected):
    body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}).encode()
    wire = f'{len(body):X};name="ignored"\r\n'.encode() + body + b"\r\n0\r\nX-Test: ignored\r\n\r\n"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", path, wire, {"Transfer-Encoding": "chunked"})
        response = connection.getresponse()
        response.read()
        original_socket = connection.sock
        assert response.status == expected
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["data"]
        assert connection.sock is original_socket and original_socket is not None
    finally:
        connection.close()


@pytest.mark.parametrize("headers, body", [
    ("Transfer-Encoding: chunked\r\nContent-Length: 2", b"0\r\n\r\n"),
    ("Transfer-Encoding: gzip", b""),
    ("Transfer-Encoding: gzip, chunked", b"0\r\n\r\n"),
    ("Transfer-Encoding: chunked, chunked", b"0\r\n\r\n"),
    ("Content-Length: bad", b""),
    ("Content-Length: -1", b""),
    ("Content-Length: 2\r\nContent-Length: 3", b"{}"),
    ("Content-Length: 20", b"{}"),
    ("Transfer-Encoding: chunked", b"+2\r\n{}\r\n0\r\n\r\n"),
    ("Transfer-Encoding: chunked", b"0x2\r\n{}\r\n0\r\n\r\n"),
    ("Transfer-Encoding: chunked", b"2\n{}\r\n0\r\n\r\n"),
    ("Transfer-Encoding: chunked", b"2\r\n{"),
    ("Transfer-Encoding: chunked", b"2\r\n{}XX0\r\n\r\n"),
    ("Transfer-Encoding: chunked", b"0\r\n"),
    ("Transfer-Encoding: chunked", b"0\r\nInvalid trailer\r\n\r\n"),
    ("Transfer-Encoding: chunked", b"2000001\r\n"),
])
def test_bad_framing_is_refused_and_connection_closed(port, headers, body):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
        client.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{headers}\r\n\r\n".encode() + body)
        client.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(client)
        response.begin()
        data = json.loads(response.read())
        assert response.status == 400, data
        assert response.getheader("Connection") == "close"
        assert data["error"]["type"] == "invalid_request_error"
        assert client.recv(1) == b""


def handler(headers, body):
    message = Message()
    for key, value in headers:
        message[key] = value
    return SimpleNamespace(headers=message, rfile=BytesIO(body), close_connection=False)


@pytest.mark.parametrize("backend", ["cuda", "mac"])
def test_chunked_decisions(backend):
    from tensorfold.cuda.http import make_handler as cuda_handler
    from tensorfold.server.http import make_handler as mac_handler

    app = SimpleNamespace(decisions=lambda body: {"echo": body})
    server = ThreadingHTTPServer(("127.0.0.1", 0), (cuda_handler if backend == "cuda" else mac_handler)(app))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    body = {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "max"}
    try:
        connection.request("POST", "/v1/decisions", iter([json.dumps(body).encode()]), encode_chunked=True)
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"echo": body}
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join()


def test_absent_and_identical_content_lengths():
    from tensorfold.server.request_body import read_body

    assert read_body(handler([], b"next request")) == b""
    request = handler([("Content-Length", "2, 02"), ("Content-Length", "2")], b"{}next request")
    assert read_body(request) == b"{}"
    assert request.rfile.read() == b"next request"


def test_body_reader_preserves_fields_and_exact_limit():
    from tensorfold.server.request_body import read_body

    payload = json.dumps({"messages": [], "reasoning_effort": "max", "tools": [],
                          "max_tokens": 262144, "stream": True}).encode()
    fixed = handler([("Content-Length", str(len(payload)))], payload)
    chunked = handler([("Transfer-Encoding", "ChUnKeD")],
                      f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
    assert read_body(fixed, limit=len(payload)) == read_body(chunked, limit=len(payload)) == payload
    assert not fixed.close_connection and not chunked.close_connection


@pytest.mark.parametrize("headers, wire", [
    ([("Content-Length", "9")], b"123456789"),
    ([("Transfer-Encoding", "chunked")], b"5\r\n12345\r\n4\r\n6789\r\n0\r\n\r\n"),
])
def test_body_reader_limits_total_decoded_bytes(headers, wire):
    from tensorfold.server.errors import RequestError
    from tensorfold.server.request_body import read_body

    request = handler(headers, wire)
    with pytest.raises(RequestError, match="limit"):
        read_body(request, limit=8)
    assert request.close_connection


@pytest.mark.parametrize("wire", [
    b"1;extension=" + b"x" * 65536 + b"\r\na\r\n0\r\n\r\n",
    b"0\r\n" + b"X-Test: " + b"x" * 65536 + b"\r\n\r\n",
    b"0\r\n" + b"X-Test: x\r\n" * 10000 + b"\r\n",
])
def test_chunk_metadata_is_bounded(wire):
    from tensorfold.server.errors import RequestError
    from tensorfold.server.request_body import read_body

    request = handler([("Transfer-Encoding", "chunked")], wire)
    with pytest.raises(RequestError):
        read_body(request)
    assert request.close_connection
