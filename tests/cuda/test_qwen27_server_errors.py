"""The CUDA server on the real Qwen3.8-27B engine and checkpoint template: refused and failed requests get an answer,
and the next reply is the same as before them."""

import http.client
import json
import os
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

MODEL = Path(os.environ.get("TF_QWEN27_MODEL", "/models/Qwen3.8-27B-MLX-4bit"))
DRAFTER = os.environ.get("TF_QWEN27_DRAFTER", "")          # the DFlash2 directory; unset serves --no-drafts
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").is_file(), reason="real checkpoint not mounted")

HI = [{"role": "user", "content": "Name three prime numbers."}]
VALID = {"messages": HI, "max_tokens": 24, "temperature": 0.7, "seed": 7,
         "chat_template_kwargs": {"enable_thinking": False}}
REFUSED = [   # (body, the start of the error message)
    ({"messages": [{"role": "tool", "content": "x"}]},
     "the chat template rejected the request: No user query found in messages."),
    ({"messages": HI, "chat_template_kwargs": {"reasoning_effort": "high"}},
     "the chat template rejected the request: Unexpected reasoning effort high."),
    ({"messages": HI, "chat_template_kwargs": "x"}, "chat_template_kwargs must be a JSON object or null"),
    ({"messages": HI, "temperature": "hot"}, "temperature must be a finite number or null"),
    ({"messages": HI, "temperature": 0.7, "seed": "abc"}, "seed must be an integer or null"),
]


@pytest.fixture(scope="module")
def app():
    from tensorfold.cuda.server import App
    from tensorfold.families.qwen3_5 import cuda_engine

    engine = cuda_engine(MODEL, drafter=DRAFTER, no_drafts=not DRAFTER, context=2048, context_explicit=True)
    return App(engine, MODEL, "qwen27-cuda")


@contextmanager
def serving(app):
    from tensorfold.cuda.server import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def post(port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read().decode()
    finally:
        connection.close()


def events(text):
    return [line[6:] if line == "data: [DONE]" else json.loads(line[6:])
            for line in text.splitlines() if line.startswith("data: ")]


def token_sha(port, stream):
    status, _, text = post(port, {**VALID, "stream": stream})
    assert status == 200
    if stream:
        return events(text)[-2]["tensorfold"]["token_sha"]
    return json.loads(text)["tensorfold"]["token_sha"]


def test_refused_and_failed_requests_leave_the_next_reply_unchanged(app):
    with serving(app) as port:
        reference = token_sha(port, False)
        assert token_sha(port, True) == reference

        for body, message in REFUSED:
            for stream in (False, True):
                status, kind, text = post(port, {**body, "stream": stream})
                assert (status, kind) == (400, "application/json")
                error = json.loads(text)["error"]
                assert error["type"] == "invalid_request_error" and error["message"].startswith(message)
        assert token_sha(port, False) == reference

        # a failure part-way through a reply, raised through the engine's generate from its token callback: at the
        # first decode round, after the prefill's token (a drafted round can finish a short reply, so no later one)
        real = app.engine.generate

        def failing(prompt, max_tokens, sampling, on_tokens, **kwargs):
            calls = [0]

            def callback(new):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("injected failure")
                return on_tokens(new)

            return real(prompt, max_tokens, sampling, callback, **kwargs)

        app.engine.generate = failing
        try:
            status, kind, text = post(port, VALID)
            assert (status, kind) == (500, "application/json")
            assert json.loads(text) == {"error": {"message": "injected failure"}}
            status, _, text = post(port, {**VALID, "stream": True})
            assert status == 200
            assert events(text)[-2:] == [{"error": {"message": "injected failure", "type": "server_error"}}, "[DONE]"]
        finally:
            del app.engine.generate
        assert token_sha(port, False) == reference
        assert token_sha(port, True) == reference
