"""The idle doorbell of two tensor-parallel ranks: while the server idles, rank 1 waits for rank 0's next message on a
CPU socket instead of inside a GPU collective (an RoCE gather's kernel polls a host flag, NCCL's spins: ~95% of rank
1's GPU on an idle server, rank 0 meanwhile idling on the CPU in its HTTP server).

Both ranks ``arm`` it at the same point of rank 0's message stream (where rank 0 goes idle); the next message's
``gate`` then rings it on rank 0 (one byte) and, on rank 1, blocks reading that byte before the collective. Busy
messages pass the gate untouched. TF_IDLE_DOORBELL=0: off (rank 1 idles in the collective, as before)."""

from __future__ import annotations

import os
import socket

SETUP_S = 60.0                                 # connecting at startup; waiting for a ring has no timeout
_PORT_KEY = "tf_idle_doorbell_port"


def enabled() -> bool:
    return os.environ.get("TF_IDLE_DOORBELL", "1") != "0"


class Doorbell:
    def __init__(self, rank: int, sock: socket.socket) -> None:
        self.rank, self.sock = rank, sock
        self.armed = False
        self.waiting = False                   # rank 1 blocked on the ring now (tests, diagnostics)
        self.rings = 0

    def arm(self) -> None:
        self.armed = True

    def gate(self) -> bool:
        """Before a message: when armed, rank 0 rings and rank 1 blocks until it does; disarmed after. False: rank 0
        closed the socket while rank 1 waited (it has stopped), so rank 1 stops following."""

        if not self.armed:
            return True
        self.armed = False
        self.rings += 1
        if self.rank == 0:
            self.sock.sendall(b"\x01")
            return True
        self.waiting = True
        try:
            got = self.sock.recv(1)            # blocking, no timeout: a signal still interrupts it (PEP 475)
        except OSError:
            got = b""
        finally:
            self.waiting = False
        return got == b"\x01"

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _tune(sock: socket.socket) -> socket.socket:
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    return sock


def connect(rank: int, master: str, store, agree) -> Doorbell | None:
    """Both ranks at startup: rank 0 listens on ``master`` (a free port, published in the rendezvous ``store``) and
    rank 1 connects. ``agree`` is the engine's all-gather of ints: both ranks vote on TF_IDLE_DOORBELL and on the
    connection, and any failure leaves it off on both (rank 1 then idles in the collective, as before)."""

    if not all(row[0] for row in agree([int(enabled())])):
        if rank == 0 and enabled():
            print("[tensorfold] idle doorbell: off (TF_IDLE_DOORBELL=0 on rank 1)", flush=True)
        return None
    sock, why = None, ""
    try:
        if rank == 0:
            try:
                server = socket.create_server((master, 0))
            except OSError:
                store.set(_PORT_KEY, "0")      # rank 1 must not wait out the store's timeout for the port
                raise
            with server:
                server.settimeout(SETUP_S)
                store.set(_PORT_KEY, str(server.getsockname()[1]))
                sock, _ = server.accept()
        else:
            port = int(store.get(_PORT_KEY))
            if not port:
                raise OSError("rank 0 could not listen")
            sock = socket.create_connection((master, port), timeout=SETUP_S)
        _tune(sock)
    except Exception as exc:                   # noqa: BLE001 - reported below; both ranks then go without it
        why = f"{type(exc).__name__}: {exc}"
        if sock is not None:
            sock.close()
        sock = None
    if not all(row[0] for row in agree([int(sock is not None)])):
        if sock is not None:
            sock.close()
        if rank == 0:
            print("[tensorfold] idle doorbell unavailable" + (f" ({why})" if why else " on rank 1")
                  + "; rank 1 idles in the collective", flush=True)
        return None
    if rank == 0:
        print("[tensorfold] idle doorbell: rank 1 waits for requests on the CPU", flush=True)
    return Doorbell(rank, sock)
