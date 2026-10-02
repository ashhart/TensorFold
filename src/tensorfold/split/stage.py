"""``tensorfold stage``: the machine that runs a split model's later layers for a Mac's ``serve --split``.

A session: the Mac connects, says which checkpoint units and layers it wants (``hello``); the stage answers with the
units it lacks (``need``), receives them into its cache, loads the layers (or keeps the ones it already holds) and
says ``ready``. Rows then flow over the same connection or an RDMA mailbox until the Mac disconnects. A new
connection always replaces the current session, so a Mac that crashed and came back is never refused.
"""

from __future__ import annotations

import argparse
import importlib
import select
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from tensorfold.split import wire
from tensorfold.split.channel import MailboxService, TcpService
from tensorfold.split.store import Store

# model_type -> the class running a layer range of that family on this machine
BACKENDS = {
    "qwen3_5": "tensorfold.families.qwen3_5.cuda.stage:Stage",
}


def model_type(config: dict) -> str:
    return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "")


def backend_class(kind: str) -> Any:
    target = BACKENDS.get(kind)
    if target is None:
        raise ValueError(f"no split stage for model_type {kind!r} (stages: {', '.join(sorted(BACKENDS))})")
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


class Server:
    def __init__(self, listen: str, cache: str, mailbox: str = "", mailbox_socket: str = "",
                 mailbox_py: str | None = None, log=print) -> None:
        host, _, port = listen.rpartition(":")
        self.address = (host or "0.0.0.0", int(port or wire.PORT))
        self.store = Store(cache)
        self.mailbox, self.mailbox_socket, self.mailbox_py = mailbox, mailbox_socket, mailbox_py
        self.service_box: MailboxService | None = None
        self.loaded: tuple[Path, Any] | None = None       # the stage directory and backend now in memory
        self.log = log
        self.listener: socket.socket | None = None

    # -- sessions ---------------------------------------------------------------------------------------------
    def serve_forever(self) -> None:
        self.listener = socket.create_server(self.address, reuse_port=False)
        self.log(f"[stage] listening on {self.address[0]}:{self.address[1]}, cache {self.store.root}")
        conn = None
        while True:
            if conn is None:
                conn, peer = self.listener.accept()
                self.log(f"[stage] session from {peer[0]}")
            try:
                conn = self.session(conn)
            except Exception as exc:   # noqa: BLE001 - the session ends, the stage stays up for the next one
                traceback.print_exc()
                self.log(f"[stage] session ended: {type(exc).__name__}: {exc}")
                try:
                    conn.close()
                except OSError:
                    pass
                conn = None

    def session(self, conn: socket.socket) -> socket.socket | None:
        """Run one Mac's session; returns a newer connection that replaced it, or None once the Mac left."""

        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = wire.recv_msg(conn)
        if hello.get("version") != wire.VERSION:
            wire.send_msg(conn, {"error": f"split protocol {hello.get('version')}, this stage speaks {wire.VERSION}"})
            conn.close()
            return None
        config, units = hello["config"], hello["units"]
        first, last, layers = int(hello["first"]), int(hello["last"]), int(hello["layers"])
        epoch = int(hello["epoch"])
        cls = backend_class(model_type(config))       # refuse an unknown family before any weights move

        need = self.store.missing(units)
        wire.send_msg(conn, {"need": need, "have": len(units) - len(need)})
        by_key = {u["key"]: u for u in units}
        started, moved = time.perf_counter(), 0
        for _ in need:
            msg = wire.recv_msg(conn)
            unit = by_key[msg["unit"]]
            self.store.receive(unit, lambda n: conn.recv(n))
            moved += int(unit["nbytes"])
        if need:
            spent = time.perf_counter() - started
            self.log(f"[stage] received {len(need)} of {len(units)} units, {moved / 1024**3:.2f} GiB in {spent:.1f} s "
                     f"({moved * 8 / spent / 1e9:.1f} Gbit/s)")
        where = self.store.stage(config, units)

        if self.loaded is None or self.loaded[0] != where:
            if self.loaded is not None:
                self.loaded[1].close()
                self.loaded = None
            t = time.perf_counter()
            backend = cls(where, first, last, layers, hello.get("options", {}))
            self.loaded = (where, backend)
            self.log(f"[stage] layers {first}..{last - 1} of {layers} loaded in {time.perf_counter() - t:.1f} s")
        backend = self.loaded[1]
        backend.reset()

        transport = hello.get("transport", "tcp")
        if transport == "mailbox":
            if not self.mailbox:
                raise ValueError("the Mac asked for the mailbox, and this stage was started without --mailbox")
            if self.service_box is None:
                self.service_box = MailboxService(self.mailbox, self.mailbox_socket, self.mailbox_py)
            service = self.service_box
        else:
            service = TcpService(conn)
        wire.send_msg(conn, {"ready": True, "hidden": backend.hidden, "taps": list(backend.tap_layers),
                             "transport": transport, "mailbox": self.mailbox})
        self.log(f"[stage] serving epoch {epoch} over {transport}")
        return self.rows(conn, service, backend, epoch)

    def rows(self, conn: socket.socket, service, backend: Any, epoch: int) -> socket.socket | None:
        calls, busy = 0, 0.0
        by_op: dict[tuple[int, int], list[float]] = {}
        while True:
            ready, _, _ = select.select([self.listener] + ([conn] if service.kind == "mailbox" else []), [], [], 0)
            if self.listener in ready:
                newer, peer = self.listener.accept()
                self.log(f"[stage] session from {peer[0]} replaces epoch {epoch}")
                conn.close()
                return newer
            if conn in ready:                         # mailbox sessions: the control socket only closes
                if not conn.recv(1, socket.MSG_PEEK):
                    self.log(f"[stage] epoch {epoch} disconnected")
                    conn.close()
                    return None
            try:
                got = service.next_request(timeout=0.2)
            except ConnectionError:
                self.log(f"[stage] epoch {epoch} disconnected")
                conn.close()
                return None
            if got is None:
                continue
            seq, payload = got
            t = time.perf_counter()
            try:
                head = wire.parse_request(payload)
                if head["epoch"] != epoch:
                    raise RuntimeError(f"a request of epoch {head['epoch']}, the session is epoch {epoch}")
                reply = backend.handle(head, payload[wire.REQUEST.size:])
            except Exception as exc:  # noqa: BLE001 - reported to the Mac, which stops that request
                traceback.print_exc()
                reply = wire.error_reply(f"{type(exc).__name__}: {exc}")
                head = {"op": 0, "rows": 0}
            service.reply(seq, [reply])
            spent = time.perf_counter() - t
            calls += 1
            busy += spent
            key = (head["op"], min(head["rows"], 16) if head["op"] == wire.FORWARD else 0)
            by_op.setdefault(key, []).append(spent)
            if calls % 500 == 0:
                names = {wire.PREFILL: "prefill", wire.FORWARD: "fwd", wire.FLUSH: "flush"}
                parts = ", ".join(f"{names.get(k[0], '?')}{k[1] or ''} {len(v)}x {sum(v) / len(v) * 1e3:.1f} ms"
                                  for k, v in sorted(by_op.items()))
                self.log(f"[stage] {calls} calls: {parts}")
                by_op.clear()


def add_parser(commands) -> None:
    stage = commands.add_parser("stage", help="run a split model's later layers for a Mac's `serve --split`",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stage.add_argument("--listen", default=f"0.0.0.0:{wire.PORT}", help="address:port the Mac connects to")
    stage.add_argument("--cache", default="~/.cache/tensorfold/stage", help="where pushed weights are kept, by hash")
    stage.add_argument("--mailbox", default="", help="a `v41rpcd spark` mailbox name for the rows (RDMA)")
    stage.add_argument("--mailbox-socket", default="", help="that daemon's service socket (default /dev/shm/v41rpc-NAME.sock)")
    stage.add_argument("--mailbox-py", default=None, help="directory holding v41rpc_mailbox.py and libv41rpc")
    stage.set_defaults(func=cmd_stage)


def cmd_stage(args: argparse.Namespace) -> int:
    sock = args.mailbox_socket or (f"/dev/shm/v41rpc-{args.mailbox}.sock" if args.mailbox else "")
    Server(args.listen, args.cache, args.mailbox, sock, args.mailbox_py,
           log=lambda s: print(s, flush=True)).serve_forever()
    return 0
