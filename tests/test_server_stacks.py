"""``kill -USR1`` prints every thread's stack while serving, after a library took the signal as Triton's LLVM does."""

import faulthandler
import http.client
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.cuda import server
from tensorfold.server import stacks

HEADER = "most recent call first"


def _dumps(capfd, tries: int) -> bool:
    """Whether USR1 prints a stack within ``tries`` signals (a server re-arms just after writing its response)."""

    for _ in range(tries):
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        if HEADER in capfd.readouterr().err:
            return True
    return False


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals only")
def test_a_finished_request_points_usr1_at_the_dump_again_after_a_takeover(capfd, monkeypatch):
    monkeypatch.setattr(stacks, "_started", False)          # restored after: other tests' requests re-arm nothing
    httpd = server.Server(("127.0.0.1", 0), server.make_handler(SimpleNamespace(served="fake")))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        stacks.start()
        assert _dumps(capfd, 1)                               # armed at start
        signal.signal(signal.SIGUSR1, lambda signum, frame: None)          # the takeover: another handler on USR1
        assert not _dumps(capfd, 3)
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
        connection.close()
        assert _dumps(capfd, 100)                             # the finished request armed it again
    finally:
        httpd.shutdown()
        httpd.server_close()
        faulthandler.unregister(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)


def test_arming_does_nothing_in_a_process_that_never_started_the_dump(monkeypatch):
    monkeypatch.setattr(stacks, "_started", False)
    calls = []
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: calls.append(a))
    stacks.arm()                                             # an embedding host keeps its own USR1 handler
    assert calls == []
