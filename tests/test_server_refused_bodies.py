"""A refused request's body never reaches the next request on its connection, on either server."""

import http.client
import json
import socket

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for
from tests.test_server_openai_compat import FakeApp, serve_fake


@pytest.fixture(params=["cuda", "mac"])
def port(request, tmp_path):
    if request.param == "cuda":
        with http_server(app_for(tmp_path)) as cuda_port:
            yield cuda_port
    else:
        server = serve_fake(FakeApp())
        yield server.server_port
        server.shutdown()
        server.server_close()


def test_a_post_to_an_unknown_route_leaves_the_connection_usable(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", "/v1/not-a-route", json.dumps({"max_tokens": 16}),
                           {"Content-Type": "application/json"})
        refused = connection.getresponse()
        refused.read()
        connection.request("POST", "/v1/chat/completions", json.dumps({"messages": HI, "max_tokens": 8}),
                           {"Content-Type": "application/json"})
        answered = connection.getresponse()
        answered.read()
    finally:
        connection.close()
    assert (refused.status, answered.status) == (404, 200)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_an_oversized_body_is_refused_and_the_connection_closed_as_the_reply_says(port, path):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
        client.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                       "Content-Length: 40000000\r\n\r\n".encode())
        reply = b""
        while chunk := client.recv(65536):      # the server closes once it has answered
            reply += chunk
    head = reply.split(b"\r\n\r\n", 1)[0].decode().lower().splitlines()
    assert head[0].split()[1] == "400"
    assert "connection: close" in head
