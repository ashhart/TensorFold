#!/usr/bin/env python3
import argparse
import os
import re
import selectors
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


KEY = b"audit-local-key"


def launch(binary, fixtures):
    env = os.environ.copy()
    env.pop("TENSORFOLD_API_KEY", None)
    env["TENSORFOLD_NO_LIVE"] = "1"
    command = [str(binary), "serve", str(fixtures), "--host", "127.0.0.1",
               "--port", "0", "--api-key", KEY.decode(), "--keep-warm", "0"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 15
    output = bytearray()
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if not selector.select(0.1):
                continue
            output.extend(os.read(process.stdout.fileno(), 4096))
            match = re.search(rb"PORT (\d+)\n", output)
            if match:
                reader = threading.Thread(target=process.stdout.read, daemon=True)
                reader.start()
                return process, int(match[1]), reader
        raise RuntimeError("local fixture server did not start: " + output.decode(errors="replace"))
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise
    finally:
        selector.close()


def exchange(port, data):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.settimeout(5)
        connection.sendall(data)
        connection.shutdown(socket.SHUT_WR)
        output = bytearray()
        while True:
            part = connection.recv(8192)
            if not part:
                break
            output.extend(part)
            if len(output) > 1 << 20:
                raise AssertionError("fixture response exceeded 1 MiB")
    return bytes(output)


def statuses(wire):
    result = []
    while wire:
        head, marker, wire = wire.partition(b"\r\n\r\n")
        if not marker:
            raise AssertionError("incomplete response header")
        lines = head.split(b"\r\n")
        result.append(int(lines[0].split(b" ", 2)[1]))
        lengths = [int(line.split(b":", 1)[1]) for line in lines[1:]
                   if line.lower().startswith(b"content-length:")]
        if len(lengths) != 1 or len(wire) < lengths[0]:
            raise AssertionError("invalid response length")
        wire = wire[lengths[0]:]
    return result


def cases():
    decision = b'{"state":null,"questions":{"q":{"instructions":"Choose one","criteria":{"A":"Alpha","B":"Beta"}}}}'
    for path in (b"/systemone", b"/alternative/systemone", b"/systemone/?x=1", b"/v1/systemone"):
        yield "key required for " + path.decode(), [401], (
            b"POST " + path + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: "
            + str(len(decision)).encode() + b"\r\n\r\n" + decision)
        yield "valid key admits " + path.decode(), [200], (
            b"POST " + path + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer " + KEY
            + b"\r\nContent-Length: " + str(len(decision)).encode() + b"\r\n\r\n" + decision)
    nested = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    # the DELETE carries the key so it reaches routing, where its body must be consumed, not dispatched
    for method, path, expected in ((b"GET", b"/health", 200), (b"DELETE", b"/unknown", 404)):
        key = b"Authorization: Bearer " + KEY + b"\r\n" if method == b"DELETE" else b""
        head = method + b" " + path + b" HTTP/1.1\r\nHost: 127.0.0.1\r\n" + key
        yield method.decode() + " body cannot become another request", [expected], (
            head + b"Content-Length: " + str(len(nested)).encode() + b"\r\n\r\n" + nested)
        yield method.decode() + " chunked body stays framed", [expected], (
            head + b"Transfer-Encoding: chunked\r\n\r\n" + format(len(nested), "x").encode()
            + b"\r\n" + nested + b"\r\n0\r\n\r\n")
    malformed = (
        b"Content-Length : 0\r\n",
        b"Content-Length\t: 0\r\n",
        b"Bad Header: x\r\nContent-Length: 0\r\n",
        b"Broken-Line\r\nContent-Length: 0\r\n",
        b": ignored\r\n",
        b"X: first\r\n Content-Length: 0\r\n",
        b"X: embedded\x00byte\r\n",
        b"X: embedded\rbyte\r\n",
    )
    for index, headers in enumerate(malformed):
        yield f"malformed header {index} closes before routing", [400], (
            b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n" + headers + b"\r\n")
    yield "truncated request head cannot dispatch", [400], b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
    yield "GET conflicting framing is rejected", [400], (
        b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    yield "GET body preserves a genuine following request", [200, 200], (
        b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 4\r\n\r\ndata"
        b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer " + KEY
        + b"\r\nConnection: close\r\n\r\n")
    yield "ordinary pipelining remains supported", [200, 200], (
        b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n" + nested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parent / "fixtures")
    args = parser.parse_args()
    process, port, reader = launch(args.binary.resolve(), args.fixtures.resolve())
    failed = 0
    total = 0
    try:
        for label, expected, data in cases():
            total += 1
            try:
                actual = statuses(exchange(port, data))
                if actual != expected:
                    raise AssertionError(f"expected {expected}, got {actual}")
                print("PASS", label)
            except (AssertionError, OSError, ValueError) as error:
                failed += 1
                print("FAIL", label + ":", str(error))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=5)
    print(f"SUMMARY server-security: {total - failed} passed, {failed} failed, 0 skipped")
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
