import logging
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from tensorfold.control.logs import Tail
from tensorfold.control.runner import logger_for, supervise
from tensorfold.control.safety import ControlError


def test_tail_partial_append_rotation_and_truncation(tmp_path):
    path = tmp_path / "server.log"
    tail = Tail(path)
    assert tail.read() == []
    path.write_bytes(b"one\ntw")
    assert tail.read() == ["one", "tw"]
    with path.open("ab") as f:
        f.write(b"o\nthree\n")
    assert tail.read() == ["one", "two", "three"]
    path.rename(tmp_path / "server.log.1")
    path.write_text("new file\n")
    assert tail.read()[-1] == "new file"
    path.write_text("a\n")
    assert tail.read()[-1] == "a"
    assert "— log rotated / truncated —" in tail.read()


def test_tail_bounded_and_sanitized(tmp_path):
    path = tmp_path / "log"
    path.write_text("large line\n" * 10000 + "Authorization: Bearer never-show\n\x1b[2Jsafe\n")
    lines = Tail(path, limit=10, byte_limit=1024).read()
    assert len(lines) <= 10 and "never-show" not in "\n".join(lines)
    assert lines[-1] == "safe"


def test_tail_rejects_symlink(tmp_path):
    target = tmp_path / "real"
    target.write_text("no")
    path = tmp_path / "log"
    path.symlink_to(target)
    with pytest.raises(ControlError):
        Tail(path).read()


def test_runner_actual_child_exit_and_redaction(tmp_path):
    path = tmp_path / "log"
    log = logger_for(path, 65536, 2)
    try:
        code = supervise([sys.executable, "-u", "-c", "print('HF_TOKEN=never-show'); raise SystemExit(7)"],
                         dict(os.environ), log, threading.Event())
    finally:
        for handler in log.handlers:
            handler.close()
    assert code == 7
    data = path.read_text()
    assert "never-show" not in data and "[REDACTED]" in data
    assert "server exited status=7" in data


def test_real_child_termination_is_bounded(tmp_path):
    log = logger_for(tmp_path / "log", 65536, 2)
    stop = threading.Event()
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    t0 = time.monotonic()
    try:
        code = supervise([sys.executable, "-u", "-c", "import time; time.sleep(60)"],
                         dict(os.environ), log, stop, grace=0.5)
    finally:
        timer.cancel()
        for handler in log.handlers:
            handler.close()
    assert code == 0 and time.monotonic() - t0 < 4


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal escalation test")
def test_stubborn_child_is_killed(tmp_path):
    log = logger_for(tmp_path / "log", 65536, 2)
    stop = threading.Event()
    class Ready(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "ready":
                stop.set()
    log.addHandler(Ready())
    watchdog = threading.Timer(8, stop.set)
    watchdog.start()
    t0 = time.monotonic()
    try:
        code = supervise(
            [sys.executable, "-u", "-c",
             "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             "print('ready'); time.sleep(60)"],
            dict(os.environ), log, stop, grace=0.3)
    finally:
        watchdog.cancel()
        for handler in log.handlers:
            handler.close()
    assert code == 0 and time.monotonic() - t0 < 10
    assert "killing child" in (tmp_path / "log").read_text()


def test_rotating_logs_are_size_bounded(tmp_path):
    path = tmp_path / "log"
    log = logger_for(path, 65536, 2)
    try:
        for _ in range(200):
            log.info("x" * 2048)
    finally:
        for handler in log.handlers:
            handler.close()
    logs = list(tmp_path.iterdir())
    assert len(logs) == 3
    assert sum(p.stat().st_size for p in logs) <= 3 * 65536


def test_follow_keeps_repeated_identical_lines(tmp_path):
    path = tmp_path / "repeated.log"
    path.write_text("same\n")
    tail = Tail(path, limit=1)
    assert tail.read_new() == ["same"]
    with path.open("a") as stream:
        stream.write("same\n")
    assert tail.read_new() == ["same"]
    assert tail.read_new() == []
    with path.open("a") as stream:
        stream.write("partial")
    assert tail.read_new() == []
    with path.open("a") as stream:
        stream.write(" line\n")
    assert tail.read_new() == ["partial line"]
