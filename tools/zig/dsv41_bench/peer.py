#!/usr/bin/env python3
"""Two-host status channel: both ranks post state so a failure on one host stops the other at once."""

import os
import socket
import subprocess
import sys
import threading
import time

GRACE_S = float(os.environ.get("PEER_GRACE_S", 180))  # the other host may start its stage this much later
LOST_S = float(os.environ.get("PEER_LOST_S", 45))      # the server unreachable this long (after it answered once, or past the grace): the peer is gone


def serve(port: int) -> int:
    state: dict[str, tuple[str, float]] = {}     # rank -> (state, the server's time it came)
    lock = threading.Lock()
    srv = socket.create_server(("0.0.0.0", port), reuse_port=False)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def handle(c: socket.socket) -> None:
        with c:
            line = c.makefile().readline().strip()
            if line.startswith("SET "):
                _, rank, msg = line.split(" ", 2)
                with lock:
                    state[rank] = (msg, time.time())
                c.sendall(b"ok\n")
            else:
                now = time.time()
                with lock:
                    c.sendall(("".join(f"{r} {now - t:.1f} {m}\n" for r, (m, t) in state.items()) + "end\n").encode())

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def talk(host: str, port: int, line: str) -> str:
    with socket.create_connection((host, port), timeout=5) as c:
        c.sendall((line + "\n").encode())
        return c.makefile().read()


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "serve":
        return serve(int(sys.argv[2]))
    host, port = sys.argv[2], int(sys.argv[3])
    if cmd == "set":
        rank, msg = sys.argv[4], " ".join(sys.argv[5:])
        for _ in range(6):          # the server may still be starting (rank 0) or briefly busy
            try:
                talk(host, port, f"SET {rank} {msg}")
                return 0
            except OSError:
                time.sleep(2)
        return 1
    if cmd == "await":
        key, limit, other = sys.argv[4], float(sys.argv[5]), sys.argv[6]
        t0 = time.time()
        heard = False
        while time.time() - t0 < limit:
            try:
                st = {}
                for l in talk(host, port, "GET").splitlines():
                    if l and l != "end":
                        r, age, msg = l.split(" ", 2)
                        st[r] = (float(age), msg)
                if key in st:
                    print(st[key][1], flush=True)
                    return 0
                if other in st:
                    age, msg = st[other]
                    if msg.startswith("FAIL"):
                        print(f"rank {other} failed: {msg}", flush=True)
                        return 1
                    if heard and age > LOST_S:
                        print(f"rank {other} stopped answering ({age:.0f} s)", flush=True)
                        return 1
                    heard = heard or age < LOST_S
            except OSError:
                pass
            time.sleep(1)
        print(f"{key} not posted within {limit:.0f} s", flush=True)
        return 1
    if cmd == "beat":
        rank, pid = sys.argv[4], int(sys.argv[5])
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                return 0                # the stage's shell is gone: its rank's heartbeat stops
            try:
                talk(host, port, f"SET {rank} ok alive")
            except OSError:
                pass
            time.sleep(3)
    if cmd == "watch":
        rank, name = sys.argv[4], sys.argv[5]
        other = "1" if rank == "0" else "0"
        t0 = time.time()
        seen = last = None
        heard = False       # the other rank's heartbeat seen once
        while True:
            try:
                st = {}
                for l in talk(host, port, "GET").splitlines():
                    if l and l != "end":
                        r, age, msg = l.split(" ", 2)
                        st[r] = (float(age), msg)
                seen = last = time.time()
                if other in st:
                    age, msg = st[other]
                    if msg.startswith("FAIL"):
                        why = f"rank {other} failed: {msg}"
                        break
                    if msg == "ok alive":
                        if heard and age > LOST_S:
                            why = f"rank {other} stopped answering ({age:.0f} s)"
                            break
                        heard = heard or age < LOST_S
            except OSError:
                now = time.time()
                if (seen is not None and now - last > LOST_S) or (seen is None and now - t0 > GRACE_S):
                    why = f"rank {other} unreachable (its status server at {host}:{port})"
                    break
            time.sleep(3)
        print(f"peer: {why}; removing {name}", flush=True)
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        return 1
    print(f"peer.py: unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
