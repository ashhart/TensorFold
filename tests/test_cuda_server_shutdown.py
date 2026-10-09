"""A stopped server gives the engine back its device memory: the tensors go when the server goes, receipts included."""

from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import http


class StoppedServer:
    """A ``Server`` that accepts, then stops at once: the shutdown path is what is under test, not the socket."""

    def __init__(self, address, handler):
        self.address, self.handler, self.closed = address, handler, False

    def serve_forever(self) -> None:
        raise KeyboardInterrupt          # what SIGTERM's handler raises in production

    def server_close(self) -> None:
        self.closed = True


def serve_once(monkeypatch, engine) -> StoppedServer:
    """Run ``serve`` to its shutdown path with no socket, and return the server it closed."""

    made: list[StoppedServer] = []

    def build(address, handler):
        made.append(StoppedServer(address, handler))
        return made[-1]

    monkeypatch.setattr(http, "Server", build)
    monkeypatch.setattr(http, "make_handler", lambda app: object())
    http.serve(SimpleNamespace(engine=engine), "127.0.0.1", 0)
    return made[-1]


def test_a_stopped_server_releases_the_engine(monkeypatch):
    """The engine holds the weights and the caches: the process dying must not be what frees them."""

    closed = []
    server = serve_once(monkeypatch, SimpleNamespace(close=lambda: closed.append(True)))
    assert server.closed is True and closed == [True]


def test_a_family_whose_engine_has_no_close_still_stops(monkeypatch):
    """Not every family has something to hand back; the shutdown path must not care."""

    server = serve_once(monkeypatch, SimpleNamespace(generate=lambda *a, **kw: None))
    assert server.closed is True
    server = serve_once(monkeypatch, SimpleNamespace())          # no engine at all
    assert server.closed is True
