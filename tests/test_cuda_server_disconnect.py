"""The CUDA server stops a request whose client has gone, and per-token failures never reach the engine."""

import errno
import http.client
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.turns import Turns
from tensorfold.families.glm5_next.cuda.app import GlmApp
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation

WAIT = 10                    # seconds: every wait on a thread, a socket or the engine is bounded by this
MESSAGES = [{"role": "user", "content": "Hi"}]
TOOLS = [{"type": "function", "function": {"name": "measure", "parameters": {"type": "object", "properties": {}}}}]
X = ord("x")


class TextTokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(c) for c in ids)


class PacedEngine:
    """One token a round from ``text`` (then "x"); ``hold_at`` pauses every request at that round until
    ``release``. ``finish_all`` decodes to the end whatever ``on_tokens`` returns, as two-rank GLM does."""

    eos = (0,)
    concurrent = False

    def __init__(self, text: str = "", *, hold_at: int | None = None, finish_all: bool = False):
        self.text, self.hold_at, self.finish_all = text, hold_at, finish_all
        self.held, self.release = threading.Event(), threading.Event()
        self.calls: list[dict] = []
        self.request = threading.local()            # GlmApp sets the request's policy here

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        call = {"prompt": list(prompt), "returned": [], "raised": None, "done": False}
        self.calls.append(call)
        try:
            for i in range(max_tokens):
                if i == self.hold_at:
                    self.held.set()
                    assert self.release.wait(WAIT)
                token = ord(self.text[i]) if i < len(self.text) else X
                try:
                    stop = on_tokens([token])
                except BaseException as exc:        # what the engine would see
                    call["raised"] = exc
                    raise
                call["returned"].append(bool(stop))
                if stop and not self.finish_all:
                    break
                time.sleep(0.002)
        finally:
            call["done"] = True
        return {"rounds": len(call["returned"])}


def stopped_at(call) -> int | None:
    """The round (1-based) at which ``on_tokens`` first asked the engine to stop."""

    return call["returned"].index(True) + 1 if True in call["returned"] else None


def app_for(tmp_path, engine, cls=server.App):
    template = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = cls.__new__(cls)
    app.engine = engine
    app.served = "fake-cuda"
    app.tok = TextTokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 4096
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def until(predicate, what: str) -> None:
    deadline = time.monotonic() + WAIT
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


class _Server(server.Server):
    def handle_error(self, request, client_address):     # quiet; the tests read the engine's record
        pass


@contextmanager
def serving(app, handler=None):
    httpd = _Server(("127.0.0.1", 0), handler or server.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(WAIT)


def send(port, body) -> socket.socket:
    """A raw client: the request is sent, nothing is read yet (a stream's headers are read)."""

    data = json.dumps(body).encode()
    conn = socket.create_connection(("127.0.0.1", port), timeout=WAIT)
    conn.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                 + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    if body.get("stream"):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = conn.recv(4096)
            assert chunk, "the server closed before the stream's headers"
            head += chunk
        assert b" 200 " in head.split(b"\r\n", 1)[0]
    return conn


def leave(conn: socket.socket) -> None:
    conn.shutdown(socket.SHUT_RDWR)
    conn.close()
    time.sleep(0.05)                      # the close reaches the server's side of the socket


def post(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


# -- a running request ------------------------------------------------------------------------

@pytest.mark.parametrize("stream,tools", [(False, False), (True, False), (True, True)],
                         ids=["non-streamed", "streamed", "streamed-inside-a-tool-call"])
def test_running_request_stops_when_its_client_leaves(tmp_path, stream, tools):
    # inside a tool call a stream writes nothing, so only the socket shows that the client has gone
    engine = PacedEngine("<tool_call>" if tools else "", hold_at=15)
    app = app_for(tmp_path, engine)
    body = {"messages": MESSAGES, "max_tokens": 400, "stream": stream, **({"tools": TOOLS} if tools else {})}
    with serving(app) as port:
        conn = send(port, body)
        assert engine.held.wait(WAIT)
        leave(conn)
        engine.release.set()
        until(lambda: engine.calls and engine.calls[0]["done"], "the request to end")
    call = engine.calls[0]
    assert call["raised"] is None
    assert stopped_at(call) is not None and stopped_at(call) <= 15 + 2, len(call["returned"])
    assert len(call["returned"]) == stopped_at(call)


@pytest.mark.parametrize("stream", [False, True])
def test_a_stopped_request_writes_no_reply_that_looks_whole(tmp_path, stream):
    # a client that shuts its sending side counts as gone (the Mac server's check); if it still reads, it must
    # not get the stopped prefix as a whole reply with finish_reason "length"
    engine = PacedEngine(hold_at=5)
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        conn = send(port, {"messages": MESSAGES, "max_tokens": 400, "stream": stream})
        assert engine.held.wait(WAIT)
        conn.shutdown(socket.SHUT_WR)
        time.sleep(0.05)
        engine.release.set()
        data = b""
        while chunk := conn.recv(65536):
            data += chunk
        conn.close()
    assert stopped_at(engine.calls[0]) is not None
    if stream:
        assert b'"finish_reason": "length"' not in data and b"[DONE]" not in data, data[-200:]
    else:
        assert data == b"", data[:200]


@pytest.mark.parametrize("stream", [False, True])
def test_a_connected_client_gets_the_whole_reply(tmp_path, stream):
    engine = PacedEngine()
    app = app_for(tmp_path, engine)
    with serving(app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
        try:
            for _ in range(1 if stream else 2):             # a kept-alive connection serves a second request
                conn.request("POST", "/v1/chat/completions",
                             json.dumps({"messages": MESSAGES, "max_tokens": 40, "stream": stream}),
                             {"Content-Type": "application/json"})
                response = conn.getresponse()
                text = response.read().decode()
                assert response.status == 200
                if stream:
                    assert text.count('"content": "x"') == 40 and text.endswith("data: [DONE]\n\n")
                else:
                    assert json.loads(text)["usage"]["completion_tokens"] == 40
        finally:
            conn.close()
    assert [call["returned"] for call in engine.calls] == [[False] * 40] * (1 if stream else 2)


# -- a request waiting for the lock -------------------------------------------------------------

class CountingTurns(Turns):
    """The server's turns, counting the requests that have been through them."""

    def __init__(self):
        super().__init__()
        self.left = 0

    def give(self) -> None:
        with self.cv:
            self.left += 1
        super().give()


@pytest.mark.parametrize("stream", [False, True])
def test_waiting_request_whose_client_left_never_reaches_the_engine(tmp_path, stream):
    engine = PacedEngine(hold_at=1)
    app = app_for(tmp_path, engine)
    app.turns = CountingTurns()
    with serving(app) as port:
        first = send(port, {"messages": [{"role": "user", "content": "first"}], "max_tokens": 3})
        assert engine.held.wait(WAIT)
        second = send(port, {"messages": [{"role": "user", "content": "second"}], "max_tokens": 3,
                             "stream": stream})
        until(lambda: app.turns.waiting == 1, "the second request to wait for its turn")
        leave(second)
        engine.release.set()
        response = http.client.HTTPResponse(first)
        response.begin()
        assert response.status == 200 and json.loads(response.read())["usage"]["completion_tokens"] == 3
        first.close()
        until(lambda: app.turns.left >= 1 and not app.turns.waiting and not app.turns.busy,
              "the second request to leave")
        assert [app.tok.decode(call["prompt"]) for call in engine.calls] == ["user:first;assistant:"]
        status, _ = post(port, {"messages": [{"role": "user", "content": "third"}], "max_tokens": 3})
    assert status == 200
    assert [app.tok.decode(call["prompt"]) for call in engine.calls] == ["user:first;assistant:",
                                                                        "user:third;assistant:"]


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("cls", [server.App, GlmApp])
def test_run_makes_no_engine_call_once_the_client_has_gone(tmp_path, concurrent, cls):
    engine = PacedEngine()
    engine.concurrent = concurrent
    app = app_for(tmp_path, engine, cls)
    with pytest.raises(RequestCancelled):
        app.run({"messages": MESSAGES, "max_tokens": 4}, True, lambda delta: True, cancelled=lambda: True)
    assert engine.calls == [] and not app._turns().busy
    assert app.run({"messages": MESSAGES, "max_tokens": 4}, True, lambda delta: True,
                   cancelled=lambda: False)["completion_tokens"] == 4


# -- per-token failures ---------------------------------------------------------------------------

class TimingOutWriter:
    """The handler's socket writer; every write after the first ``ok`` fails as a timed-out socket does."""

    def __init__(self, inner, ok: int):
        self.inner, self.ok, self.writes = inner, ok, 0

    def write(self, data):
        self.writes += 1
        if self.writes > self.ok:
            raise OSError(errno.ETIMEDOUT, "Operation timed out")
        return self.inner.write(data)

    def flush(self):
        self.inner.flush()

    @property
    def closed(self):
        return self.inner.closed

    def close(self):
        self.inner.close()


def test_a_timed_out_stream_write_stops_the_engine_without_raising_into_it(tmp_path):
    engine = PacedEngine()
    app = app_for(tmp_path, engine)
    handled, raised = threading.Event(), []

    class Handler(server.make_handler(app)):
        def setup(self):
            super().setup()
            self.wfile = TimingOutWriter(self.wfile, ok=5)     # headers, role and three tokens

        def do_POST(self):
            try:
                super().do_POST()
            except Exception as exc:                            # noqa: BLE001  what the handler lets out
                raised.append(exc)
                raise
            finally:
                handled.set()

    with serving(app, Handler) as port:
        conn = send(port, {"messages": MESSAGES, "max_tokens": 400, "stream": True})
        assert handled.wait(WAIT)
        conn.close()
    call = engine.calls[0]
    assert call["raised"] is None and raised == []             # a client that has gone, not a server error
    assert call["returned"] == [False, False, False, True]


@pytest.mark.parametrize("finish_all", [False, True], ids=["engine-stops", "engine-finishes-both-ranks"])
def test_a_per_token_failure_stops_the_engine_and_is_raised_after_generate(tmp_path, monkeypatch, finish_all):
    engine = PacedEngine(finish_all=finish_all)
    app = app_for(tmp_path, engine)
    adds = []
    add = server.StreamDecoder.add

    def failing_add(self, new):
        adds.append(new)
        if len(adds) == 3:
            raise RuntimeError("decoding failed")
        return add(self, new)

    monkeypatch.setattr(server.StreamDecoder, "add", failing_add)
    sent = []
    with pytest.raises(RuntimeError, match="decoding failed"):
        app.run({"messages": MESSAGES, "max_tokens": 20}, True, lambda delta: sent.append(delta) or True)
    call = engine.calls[0]
    assert call["raised"] is None                      # generate returned normally
    assert call["returned"] == [False, False] + [True] * (18 if finish_all else 1)
    assert len(adds) == 3 and len(sent) == 2           # nothing decoded or sent after the failure
    monkeypatch.setattr(server.StreamDecoder, "add", add)
    assert app.run({"messages": MESSAGES, "max_tokens": 4}, True, lambda delta: True)["completion_tokens"] == 4


class PacedDecoder:
    """The scheduler's decoder interface: one token a round for each live stream."""

    def __init__(self):
        self.streams, self.seen = [], []

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        self.streams.append(stream)
        self.seen.append(stream)
        stream.take([X])

    def round(self):
        time.sleep(0.002)
        done = []
        for stream in self.streams:
            if not stream.done:
                stream.take([X])
                stream.counted(1)
                if stream.done:
                    done.append(stream)
        return done

    def finish(self, done):
        self.streams = [s for s in self.streams if s not in done]

    def drop(self):
        dropped, self.streams = self.streams, []
        return dropped


class SchedulerEngine:
    eos = (0,)
    concurrent = True

    def __init__(self):
        self.decoder = PacedDecoder()
        self.scheduler = Scheduler(self.decoder, max_streams=2)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens)


def test_a_per_token_failure_ends_a_shared_round_stream(tmp_path, monkeypatch):
    engine = SchedulerEngine()
    app = app_for(tmp_path, engine)
    add = server.StreamDecoder.add
    calls = []

    def failing_add(self, new):
        calls.append(new)
        if len(calls) == 3:
            raise RuntimeError("decoding failed")
        return add(self, new)

    monkeypatch.setattr(server.StreamDecoder, "add", failing_add)
    with pytest.raises(RuntimeError, match="decoding failed"):
        app.run({"messages": MESSAGES, "max_tokens": 400}, True, lambda delta: True)
    until(lambda: engine.decoder.live() == 0, "the stream to leave the decoder")
    stream = engine.decoder.seen[0]
    assert len(stream.out) < 10, len(stream.out)          # it ended a round or two later, not at its count


def test_the_socket_check_reads_descriptors_past_1023():
    resource = pytest.importorskip("resource")          # POSIX: Windows' select() has no 1023 limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1100:
        pytest.skip("the descriptor limit is below 1100")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 1100), hard))
    try:
        a, b = socket.socketpair()
        high = socket.socket(fileno=os.dup2(a.fileno(), 1050))
        a.close()
        try:
            gone = socket_cancellation(high)
            assert not gone.cancelled                   # open and quiet: select() would raise here
            b.close()
            assert gone.cancelled
        finally:
            high.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
