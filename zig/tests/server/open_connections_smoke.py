#!/usr/bin/env python3
"""Model-free connection gauge smoke: pass fake_serve and its fixture directory."""
import contextlib
import http.client
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time


def listener(proc):
    deadline = time.monotonic() + 10
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(max(0, deadline - time.monotonic())):
                break
            line = proc.stdout.readline()
            if line.startswith(b"PORT "):
                return int(line.split()[1])
            if not line:
                break
    raise RuntimeError("fake server did not announce its listener")


def scrape(port):
    with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=1)) as conn:
        conn.request("GET", "/metrics", headers={"Connection": "close"})
        response = conn.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise AssertionError(f"metrics status: {response.status}")
        values = [int(line.split()[1]) for line in body.splitlines()
                  if line.startswith("tensorfold:open_connections ")]
        if len(values) != 1:
            raise AssertionError(f"expected one connection gauge, got {values}")
        print(f"tensorfold:open_connections {values[0]}", flush=True)
        return values[0]


def expect(port, expected):
    deadline = time.monotonic() + 2
    while True:
        observed = scrape(port)
        if observed == expected:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"expected {expected} open connections, observed {observed}")
        time.sleep(0.02)


def main(binary, fixtures):
    proc = subprocess.Popen(
        [str(binary), "serve", str(fixtures), "--name", "fixture", "--port", "0",
         "--host", "127.0.0.1", "--keep-warm", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        env={**os.environ, "TENSORFOLD_NO_LIVE": "1"},
    )
    try:
        port = listener(proc)
        observed = scrape(port)
        if observed != 1:
            raise AssertionError(f"expected 1 open connection, observed {observed}")
        with contextlib.ExitStack() as stack:
            for _ in range(3):
                sock = stack.enter_context(socket.create_connection(("127.0.0.1", port), timeout=1))
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n")
                with http.client.HTTPResponse(sock) as response:
                    response.begin()
                    response.read()
                    if response.status != 200 or response.will_close:
                        raise AssertionError("health did not retain the keep-alive connection")
            expect(port, 4)
        expect(port, 1)
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: open_connections_smoke.py FAKE_SERVE FIXTURES")
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
