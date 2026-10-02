"""The split's two protocols: JSON control messages (hello, weights, ready) and the binary rows a stage computes."""

from __future__ import annotations

import json
import socket
import struct

VERSION = 1
PORT = 18640

# a request: magic, version, op, flags, layer, epoch, slot, rows, start, extra, n_commit; then by op
#   PREFILL  commit path int32[n_commit], residual bf16[rows, H], pending bf16[rows, H]; extra 0 none, 1 last row, 2 all
#   FORWARD  commit path, parents int32[rows], residual, pending (a decode window; committed by the next request)
#   FLUSH    commit path only
#   FETCH    the state of layer ``extra`` for the Mac: a recurrent layer's conv tail bf16 and state fp32, or an
#            attention layer's keys and values bf16[rows, kv_heads, head_dim] for positions [start, start + rows)
# ``layer``: the layer the rows enter at (a prompt may enter earlier than decode windows; 0: the stage's first)
REQUEST = struct.Struct("<4sHHHHIIIIII")
MAGIC = b"TFS1"
#   PUSH     weights for the stage's cache: piece ``start`` of unit ``extra`` (its index in the stage's ``need``);
#            flags LAST on a unit's last piece
PREFILL, FORWARD, FLUSH, FETCH, PUSH = 1, 2, 3, 4, 5
TAPS = 1                      # flags: also return this stage's DFlash2 tap layers, each bf16[rows, H]
ALL = 2                       # FETCH flags: every layer the Mac fetches at once (``extra`` 0: recurrent, 1: attention)
LAST = 4                      # PUSH flags: the unit's last piece
ONE = 8                       # PREFILL/FORWARD flags: one block of rows (a family whose residual is its streams), no pending
PIECE = 48 << 20              # a pushed piece's bytes (a mailbox's request half is 64 MiB)
# a reply: status (0 ok), rows, tap layers; then normed bf16[rows, H] and taps x bf16[rows, H]; else an error string
REPLY = struct.Struct("<III")


def request(op: int, *, epoch: int, rows: int, start: int, extra: int, n_commit: int, flags: int = 0,
            slot: int = 0, layer: int = 0) -> bytes:
    return REQUEST.pack(MAGIC, VERSION, op, flags, layer, epoch, slot, rows, start, extra, n_commit)


def parse_request(buf) -> dict:
    magic, version, op, flags, layer, epoch, slot, rows, start, extra, n_commit = REQUEST.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not a split request of version {VERSION} (magic {magic!r}, version {version})")
    return {"op": op, "flags": flags, "layer": layer, "epoch": epoch, "slot": slot, "rows": rows, "start": start,
            "extra": extra, "n_commit": n_commit}


def error_reply(text: str) -> bytes:
    raw = text.encode()
    return REPLY.pack(1, len(raw), 0) + raw


# control messages: u64 length then UTF-8 JSON; raw byte streams (weights) follow the message announcing them
_LEN = struct.Struct("<Q")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError(f"the peer closed the connection {n - got} bytes short")
        got += k
    return bytes(buf)


def send_msg(sock: socket.socket, obj: dict) -> None:
    raw = json.dumps(obj).encode()
    sock.sendall(_LEN.pack(len(raw)) + raw)


def recv_msg(sock: socket.socket) -> dict:
    n = _LEN.unpack(recv_exact(sock, _LEN.size))[0]
    if n > 1 << 30:
        raise ValueError(f"a {n}-byte control message")
    return json.loads(recv_exact(sock, n))


def send_frame(sock: socket.socket, parts) -> None:
    """One binary frame of the rows path over TCP: its length, then its parts."""

    views = [memoryview(p).cast("B") for p in parts]
    sock.sendall(_LEN.pack(sum(v.nbytes for v in views)))
    for v in views:
        sock.sendall(v)


def recv_frame(sock: socket.socket) -> memoryview:
    n = _LEN.unpack(recv_exact(sock, _LEN.size))[0]
    return memoryview(recv_exact(sock, n))
