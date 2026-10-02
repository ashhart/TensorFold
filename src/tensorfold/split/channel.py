"""The rows path between the Mac and a stage: over the control TCP connection, or an RDMA mailbox of `v41rpcd`.

Each end keeps at most one request in flight: ``submit`` sends it, ``result`` waits for its reply (with the GIL
released, so MLX keeps encoding the next piece meanwhile).
"""

from __future__ import annotations

import os
import select
import socket
import sys

from tensorfold.split import wire

# the directory holding v41rpc_mailbox.py and libv41rpc, the client and service ends of a `v41rpcd` RDMA mailbox
MAILBOX_PY = os.environ.get("TF_SPLIT_MAILBOX_PY", "")


def _mailbox_module(where: str | None):
    path = where or MAILBOX_PY
    if not path:
        raise ValueError("the mailbox transport needs v41rpc_mailbox.py: set TF_SPLIT_MAILBOX_PY (or --mailbox-py)")
    if path not in sys.path:
        sys.path.insert(0, path)
    import v41rpc_mailbox

    return v41rpc_mailbox


class TcpClient:
    kind = "tcp"

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.busy = False

    def submit(self, parts) -> None:
        if self.busy:
            raise RuntimeError("a split request is already in flight")
        wire.send_frame(self.sock, parts)
        self.busy = True

    def result(self, timeout: float = 600.0) -> memoryview:
        self.sock.settimeout(timeout)
        try:
            return wire.recv_frame(self.sock)
        finally:
            self.sock.settimeout(None)
            self.busy = False

    def close(self) -> None:
        pass


class MailboxClient:
    kind = "mailbox"

    def __init__(self, name: str, mailbox_py: str | None = None) -> None:
        self.box = _mailbox_module(mailbox_py).ClientBox(name)
        self.seq: int | None = None

    def submit(self, parts) -> None:
        if self.seq is not None:
            raise RuntimeError("a split request is already in flight")
        self.seq = self.box.stage(parts)

    def result(self, timeout: float = 600.0) -> memoryview:
        seq, self.seq = self.seq, None
        return self.box.wait(seq, timeout=timeout)

    def close(self) -> None:
        self.box.close()


class TcpService:
    """Requests arriving on the control connection itself."""

    kind = "tcp"

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def next_request(self, timeout: float):
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if not ready:
            return None
        return 0, wire.recv_frame(self.sock)

    def reply(self, seq: int, parts) -> None:
        wire.send_frame(self.sock, parts)

    def close(self) -> None:
        pass


class MailboxService:
    """Requests from a `v41rpcd spark` daemon's mailbox; the control connection only says the Mac is alive."""

    kind = "mailbox"

    def __init__(self, name: str, sock_path: str, mailbox_py: str | None = None) -> None:
        self.box = _mailbox_module(mailbox_py).ServiceBox(name, sock_path)

    def next_request(self, timeout: float):
        return self.box.next_request(timeout=timeout)

    def reply(self, seq: int, parts) -> None:
        self.box.reply(seq, parts)

    def close(self) -> None:
        close = getattr(self.box, "close", None)
        if close:
            close()
