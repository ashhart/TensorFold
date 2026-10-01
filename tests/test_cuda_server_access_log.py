"""The CUDA server prints one line a request, as the Mac server does, except a successful metrics or health poll."""

import http.client

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for, post, serving

WAIT = 10


def get(port, path) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def test_requests_print_a_line_and_successful_polls_do_not(tmp_path, capfd):
    app = app_for(tmp_path, PacedEngine())
    app.context_window = 262144
    with serving(app) as port:
        assert post(port, {"messages": MESSAGES, "max_tokens": 4})[0] == 200
        assert get(port, "/health") == 200
        assert get(port, "/v1/health/") == 200
        assert get(port, "/v1/models") == 200
        assert get(port, "/nowhere") == 404
    lines = [line for line in capfd.readouterr().out.splitlines() if line.startswith("[tensorfold] 127.0.0.1 ")]
    assert any('"POST /v1/chat/completions HTTP/1.1" 200' in line for line in lines), lines
    assert any('"GET /v1/models HTTP/1.1" 200' in line for line in lines), lines
    assert any('"GET /nowhere HTTP/1.1" 404' in line for line in lines), lines
    assert not any("/health" in line for line in lines), lines


def test_a_poll_that_does_not_answer_200_still_prints(tmp_path, capfd):
    app = app_for(tmp_path, PacedEngine())
    with serving(app) as port:
        assert get(port, "/v1/health?full=1") == 404       # the health route takes no query string
    out = capfd.readouterr().out
    assert '"GET /v1/health?full=1 HTTP/1.1" 404' in out, out
