#!/usr/bin/env python3
"""serve --listen-fd in a real process tree: the parent is served, others are not, and its exit stops the server."""
import argparse
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def bound(path, listening=True):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if path is not None:
        sock.bind(path)
        if listening:
            sock.listen(16)
    sock.set_inheritable(True)
    return sock


def environment(**extra):
    env = os.environ.copy()
    env.pop("TENSORFOLD_API_KEY", None)
    env["TENSORFOLD_NO_LIVE"] = "1"
    env.update(extra)
    return env


def command(binary, fixtures, fd):
    return [str(binary), "serve", str(fixtures), "--listen-fd", str(fd), "--keep-warm", "0", "--execution-witness"]


def ready(log_path, process, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(log_path).read_text(errors="replace").find("PORT ") >= 0:
            return
        if process is not None and process.poll() is not None:
            break
        time.sleep(0.05)
    raise AssertionError("the server did not start: " + Path(log_path).read_text(errors="replace")[-800:])


def get(path, timeout=5):
    """GET /health over the socket at `path`; the reply's bytes, empty when the server closes without one."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
        c.settimeout(timeout)
        c.connect(path)
        c.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        out = b""
        try:
            while chunk := c.recv(4096):
                out += chunk
        except ConnectionResetError:
            pass
        return out


# The server may close a refused connection before the request is sent: BrokenPipe there is a refusal too
SIBLING = "import socket,sys\nc=socket.socket(socket.AF_UNIX)\nc.settimeout(5)\nc.connect(sys.argv[1])\n" \
          "try:\n    c.sendall(b'GET /health HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n')\n    got=c.recv(64)\n" \
          "except (BrokenPipeError, ConnectionResetError):\n    got=b''\nprint(len(got))\n"


def served_and_refused(binary, fixtures, work):
    path = os.path.join(work, "a.sock")
    sock = bound(path)
    log = os.path.join(work, "a.log")
    with open(log, "wb") as out:
        server = subprocess.Popen(command(binary, fixtures, sock.fileno()), pass_fds=(sock.fileno(),), stdout=out,
                                  stderr=subprocess.STDOUT, env=environment())
    sock.close()
    try:
        ready(log, server)
        reply = get(path)
        assert reply.startswith(b"HTTP/1.1 200 "), reply[:80]
        other = subprocess.run([sys.executable, "-I", "-c", SIBLING, path], capture_output=True, text=True, timeout=15)
        assert other.stdout.strip() == "0", f"another process got {other.stdout.strip() or other.stderr} bytes"
        assert "refused a connection from a process other than" in Path(log).read_text(errors="replace")
    finally:
        server.terminate()
        server.wait(timeout=10)
    assert os.path.exists(path), "the server removed the parent's socket path"


def refused_at_start(binary, fixtures, work, label, sock, env, expected):
    log = os.path.join(work, label + ".log")
    fd = sock.fileno()
    with open(log, "wb") as out:
        code = subprocess.run(command(binary, fixtures, fd), pass_fds=(fd,), stdout=out, stderr=subprocess.STDOUT,
                              env=env, timeout=30).returncode
    text = Path(log).read_text(errors="replace")
    assert code != 0 and expected in text, f"exit {code}: {text[-400:]}"


# the parent: binds, starts the server, takes one reply on a kept connection, forks a holder of it, and exits
PARENT = """
import os, socket, subprocess, sys, time
binary, fixtures, path, log, life = sys.argv[1:6]
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(path)
sock.listen(16)
sock.set_inheritable(True)
life_fd = int(life)
out = open(log, "wb")
server = subprocess.Popen([binary, "serve", fixtures, "--listen-fd", str(sock.fileno()), "--keep-warm", "0"],
                          pass_fds=(sock.fileno(), life_fd), stdout=out, stderr=subprocess.STDOUT,
                          start_new_session=True)
print(server.pid, flush=True)
os.close(life_fd)  # only the server keeps it, so its end of file means the server exited
deadline = time.monotonic() + 15
while b"PORT " not in open(log, "rb").read() and time.monotonic() < deadline:
    time.sleep(0.05)
kept = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
kept.settimeout(5)
kept.connect(path)
kept.sendall(b"GET /health HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n")
first = kept.recv(4096)
print("served" if first.startswith(b"HTTP/1.1 200 ") else "unserved", flush=True)
if os.fork() == 0:
    kept.settimeout(15)
    try:
        while kept.recv(4096):
            pass
        print("closed", flush=True)
    except (ConnectionResetError, socket.timeout) as e:
        print("reset" if isinstance(e, ConnectionResetError) else "open", flush=True)
    os._exit(0)
os._exit(0)
"""


def parent_exit_stops(binary, fixtures, work):
    path = os.path.join(work, "p.sock")
    log = os.path.join(work, "p.log")
    life_read, life_write = os.pipe()
    os.set_inheritable(life_write, True)
    parent = subprocess.Popen([sys.executable, "-I", "-c", PARENT, str(binary), str(fixtures), path, log,
                               str(life_write)], pass_fds=(life_write,), stdout=subprocess.PIPE, text=True,
                              env=environment())
    os.close(life_write)
    server = int(parent.stdout.readline())
    first = parent.stdout.readline().strip()
    try:
        assert parent.wait(timeout=30) == 0 and first == "served", f"the parent saw {first!r}"
        # the server holds the only other copy of the pipe: end of file means it exited, with no signal sent to it
        ready_now, _, _ = select.select([life_read], [], [], 15)
        assert ready_now and os.read(life_read, 1) == b"", "the server still runs after its parent exited"
    except AssertionError:
        os.kill(server, 9)  # this check started it: it does not outlive a failure
        raise
    finally:
        os.close(life_read)
    held = parent.stdout.readline().strip()
    assert held in ("closed", "reset"), f"the held connection stayed {held or 'silent'}"
    text = Path(log).read_text(errors="replace")
    assert "the process that passed the socket exited" in text, text[-600:]


# the parent: binds, starts the server holding its load on TF_FAKE_LOAD_FD, and exits when the check says so
LOADING = """
import socket, subprocess, sys
binary, fixtures, path, log, life, load = sys.argv[1:7]
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(path)
sock.listen(16)
sock.set_inheritable(True)
env = dict(__import__("os").environ, TF_FAKE_LOAD_FD=load)
server = subprocess.Popen([binary, "serve", fixtures, "--listen-fd", str(sock.fileno()), "--keep-warm", "0"],
                          pass_fds=(sock.fileno(), int(life), int(load)), stdout=open(log, "wb"),
                          stderr=subprocess.STDOUT, start_new_session=True, env=env)
print(server.pid, flush=True)
sys.stdin.readline()
__import__("os")._exit(0)
"""


def parent_exit_during_load(binary, fixtures, work):
    """The parent exits after the server took the socket, before it serves: the server stops, its adopter is refused."""
    if sys.platform.startswith("linux"):  # this check adopts the orphan, so the adopter itself connects
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    path = os.path.join(work, "l.sock")
    log = os.path.join(work, "l.log")
    life_read, life_write = os.pipe()
    load_here, load_there = socket.socketpair()
    os.set_inheritable(life_write, True)
    load_there.set_inheritable(True)
    parent = subprocess.Popen([sys.executable, "-I", "-c", LOADING, str(binary), str(fixtures), path, log,
                               str(life_write), str(load_there.fileno())], pass_fds=(life_write, load_there.fileno()),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=environment())
    os.close(life_write)
    load_there.close()
    server = int(parent.stdout.readline())
    try:
        load_here.settimeout(15)
        assert load_here.recv(1) == b"t", "the server did not take the socket"
        parent.stdin.write("exit\n")
        parent.stdin.flush()
        assert parent.wait(timeout=15) == 0
        load_here.sendall(b"g")  # the load ends after its parent exited
        deadline = time.monotonic() + 15
        while b"PORT " not in Path(log).read_bytes() and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            reply = get(path, timeout=3)
        except OSError:
            reply = b""
        assert not reply.startswith(b"HTTP/1.1 200 "), "the process that adopted the server was served"
        ready_now, _, _ = select.select([life_read], [], [], 15)
        gone = ready_now and os.read(life_read, 1) == b""
        assert gone, "the server still runs after its parent exited during the load"
    except AssertionError:
        os.kill(server, 9)  # this check started it: it does not outlive a failure
        raise
    finally:
        os.close(life_read)
        load_here.close()
    text = Path(log).read_text(errors="replace")
    assert "the process that passed the socket exited" in text, text[-600:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parent / "fixtures")
    args = parser.parse_args()
    if os.name == "nt":
        print("SUMMARY listen-fd: 0 passed, 0 failed, 5 skipped (POSIX only)")
        return 0
    binary, fixtures = args.binary.resolve(), args.fixtures.resolve()
    failed = 0
    with tempfile.TemporaryDirectory(prefix="tf-lfd-", dir="/tmp") as work:
        key_socket = os.path.join(work, "k.sock")
        cases = [
            ("the parent is served, another process is refused, the path stays",
             lambda: served_and_refused(binary, fixtures, work)),
            ("serve refuses TENSORFOLD_API_KEY beside --listen-fd",
             lambda: refused_at_start(binary, fixtures, work, "key", bound(key_socket),
                                      environment(TENSORFOLD_API_KEY="k"), "not allowed with TENSORFOLD_API_KEY")),
            ("serve refuses a socket that was never bound",
             lambda: refused_at_start(binary, fixtures, work, "bare", bound(None), environment(),
                                      "not bound to an address")),
            ("the parent's exit stops the server and ends its kept connection",
             lambda: parent_exit_stops(binary, fixtures, work)),
            ("the parent's exit during the load stops the server, and its adopter is refused",
             lambda: parent_exit_during_load(binary, fixtures, work)),
        ]
        for label, check in cases:
            try:
                check()
                print("PASS", label)
            except (AssertionError, OSError, subprocess.SubprocessError) as error:
                failed += 1
                print("FAIL", label + ":", error)
    print(f"SUMMARY listen-fd: {len(cases) - failed} passed, {failed} failed, 0 skipped")
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
