"""``tensorfold stage``: the machine that runs a split model's later layers for a Mac's ``serve --split``.

A session: the Mac connects, says which checkpoint units and layers it wants (``hello``); the stage answers with the
units it lacks (``need``), receives them into its cache, loads the layers (or keeps the ones it already holds) and
says ``ready``. Rows then flow over the same connection or an RDMA mailbox until the Mac disconnects. A new
connection always replaces the current session, so a Mac that crashed and came back is never refused.
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    "qwen4_exp": "tensorfold.families.qwen4_exp.cuda.stage:Stage",
    "qwen3_8_flash_next": "tensorfold.families.qwen4_exp.cuda.stage:Stage",
    "glm5_next": "tensorfold.families.glm5_next.cuda.stage:Stage",
}


def model_type(config: dict) -> str:
    return str(config.get("model_type") or config.get("text_config", {}).get("model_type") or "").removesuffix("_text")


def backend_class(kind: str) -> Any:
    target = BACKENDS.get(kind)
    if target is None:
        raise ValueError(f"no split stage for model_type {kind!r} (stages: {', '.join(sorted(BACKENDS))})")
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


class Server:
    def __init__(self, listen: str, cache: str, mailbox: str = "", mailbox_socket: str = "",
                 mailbox_py: str | None = None, log=print, rdma=None) -> None:
        host, _, port = listen.rpartition(":")
        self.address = (host or "0.0.0.0", int(port or wire.PORT))
        self.store = Store(cache)
        self.mailbox, self.mailbox_socket, self.mailbox_py = mailbox, mailbox_socket, mailbox_py
        self.service_box: MailboxService | None = None
        self.rdma = rdma                                  # a split.rdma.SparkDaemon this stage runs, or None
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
        transport = hello.get("transport", "tcp")
        extra: dict = {}
        service = None
        if transport == "rdma":                      # the daemons come up first, so weights can ride them too
            service, extra = self.start_rdma()
        wire.send_msg(conn, {"need": need, "have": len(units) - len(need), **extra})
        by_key = {u["key"]: u for u in units}
        started, moved = time.perf_counter(), 0
        if need and transport == "rdma":
            moved = self.receive_both(conn, service, by_key, need, epoch)
        else:
            for _ in need:
                msg = wire.recv_msg(conn)
                unit = by_key[msg["unit"]]
                self.store.receive(unit, lambda n: conn.recv(n))
                moved += int(unit["nbytes"])
        if need:
            spent = time.perf_counter() - started
            self.log(f"[stage] received {len(need)} of {len(units)} units over {'RDMA' if transport == 'rdma' else 'TCP'}"
                     f", {moved / 1024**3:.2f} GiB in {spent:.1f} s ({moved * 8 / spent / 1e9:.1f} Gbit/s)")
        where = self.store.stage(config, units)

        options = hello.get("options", {})
        key = (where, json.dumps(options, sort_keys=True))
        if self.loaded is None or self.loaded[0] != key:
            if self.loaded is not None:
                self.loaded[1].close()
                self.loaded = None
            t = time.perf_counter()
            backend = cls(where, first, last, layers, options)
            self.loaded = (key, backend)
            self.log(f"[stage] layers {first}..{last - 1} of {layers} loaded in {time.perf_counter() - t:.1f} s")
        backend = self.loaded[1]
        backend.reset()

        if transport == "rdma":
            pass                                     # started before the weights moved
        elif transport == "mailbox":
            if not self.mailbox:
                raise ValueError("the Mac asked for the mailbox, and this stage was started without --mailbox")
            if self.service_box is None:
                self.service_box = MailboxService(self.mailbox, self.mailbox_socket, self.mailbox_py)
            service = self.service_box
        else:
            service = TcpService(conn)
        wire.send_msg(conn, {"ready": True, "hidden": backend.hidden, "taps": list(backend.tap_layers),
                             "out_width": int(getattr(backend, "out_width", backend.hidden)),
                             "decode_first": int(getattr(backend, "decode_first", first)),
                             "decode_taps": list(getattr(backend, "decode_taps", backend.tap_layers)),
                             "transport": transport, "mailbox": self.mailbox, **extra})
        self.log(f"[stage] serving epoch {epoch} over {transport}")
        return self.rows(conn, service, backend, epoch)

    def start_rdma(self) -> tuple[MailboxService, dict]:
        """A fresh daemon and mailbox for this session (nothing of an earlier one can refuse it), and what the Mac
        needs to reach it."""

        if self.rdma is None:
            raise ValueError("the Mac asked for RDMA, and this stage was started without --rdma")
        if self.service_box is not None:             # the last session's mailbox goes with its daemon
            self.service_box.close()
            self.service_box = None
        t = time.perf_counter()
        self.rdma.start()
        self.service_box = MailboxService(self.rdma.name, self.rdma.sock, self.mailbox_py)
        a = self.rdma.address
        self.log(f"[stage] v41rpcd spark {self.rdma.name} on {a.device} gid {a.gid_index} ({a.ip}, {a.mac}) "
                 f"up in {time.perf_counter() - t:.2f} s")
        return self.service_box, {"rdma": {"name": self.rdma.name, "port": self.rdma.port, "ip": a.ip,
                                           "prefix": a.prefix, "mac": a.mac, "req_mib": self.rdma.req_mib,
                                           "rep_mib": self.rdma.rep_mib}}

    def receive_both(self, conn: socket.socket, service, by_key: dict, need: list[str], epoch: int) -> int:
        """Units over both links at once: the ones the Mac names for TCP on the control connection (a thread), the
        rest through the mailbox. Their sum is the push's rate: 10 GbE and RoCE are separate wires."""

        import threading

        plan = wire.recv_msg(conn)["paths"]
        over_tcp = [by_key[k] for k in plan["tcp"]]
        moved = [0]
        failed: list[BaseException] = []

        def tcp() -> None:
            try:
                for _ in over_tcp:
                    unit = by_key[wire.recv_msg(conn)["unit"]]
                    self.store.receive(unit, lambda n: conn.recv(n))
                    moved[0] += int(unit["nbytes"])
            except BaseException as exc:       # noqa: BLE001 - raised on the session's thread
                failed.append(exc)

        thread = threading.Thread(target=tcp, name="stage-push-tcp", daemon=True)
        thread.start()

        def gone() -> BaseException | None:
            """A Mac gone mid-push: the TCP half failed, or (it done) the control connection closed."""

            if failed:
                return failed[0]
            if thread.is_alive():
                return None
            try:
                if conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                    return ConnectionError("the Mac closed the session during the push")
            except (BlockingIOError, InterruptedError):
                pass
            return None

        try:
            got = (self.receive_pushed(None, service, [by_key[k] for k in plan["rdma"]], epoch, gone)
                   if plan["rdma"] else 0)
        finally:
            thread.join()
        if failed:
            raise failed[0]
        msg = wire.recv_msg(conn)
        if not msg.get("pushed"):
            raise RuntimeError(f"expected the end of the push, got {msg}")
        return got + moved[0]

    def receive_pushed(self, conn: socket.socket | None, service, units: list[dict], epoch: int,
                       gone: Any = None) -> int:
        """Units the Mac pushes through the mailbox, written by a thread while the next piece crosses."""

        import queue
        import threading

        todo: queue.Queue = queue.Queue(maxsize=4)          # pieces copied out of the mailbox, awaiting the disk
        failed: list[BaseException] = []

        def write() -> None:
            writers: dict[int, Any] = {}
            try:
                while True:
                    item = todo.get()
                    if item is None:
                        for w in writers.values():       # units the push left unfinished
                            w.abort()
                        return
                    index, piece, last = item
                    w = writers.get(index)
                    if w is None:
                        w = writers[index] = self.store.writer(units[index])
                    w.write(piece)
                    if last:
                        w.finish()
                        del writers[index]
            except BaseException as exc:       # noqa: BLE001 - reported on the session's thread
                failed.append(exc)
                for w in writers.values():
                    w.abort()

        writer = threading.Thread(target=write, name="stage-push-writer", daemon=True)
        writer.start()
        done, moved = 0, 0
        try:
            while done < len(units):
                if failed:
                    raise failed[0]
                got = service.next_request(timeout=0.5)
                if got is None:
                    lost = gone() if gone else None
                    if lost is not None:
                        raise lost
                    continue
                seq, payload = got
                head = wire.parse_request(payload)
                if head["op"] != wire.PUSH or head["epoch"] != epoch:
                    service.reply(seq, [wire.error_reply(f"op {head['op']} while weights are being pushed")])
                    continue
                piece = bytes(payload[wire.REQUEST.size:])          # out of the mailbox before the reply frees it
                todo.put((head["extra"], piece, bool(head["flags"] & wire.LAST)))
                service.reply(seq, [wire.REPLY.pack(0, 0, 0)])
                moved += len(piece)
                done += bool(head["flags"] & wire.LAST)
        finally:
            todo.put(None)
            writer.join()
        if failed:
            raise failed[0]
        if conn is not None:
            msg = wire.recv_msg(conn)
            if not msg.get("pushed"):
                raise RuntimeError(f"expected the end of the push, got {msg}")
        return moved

    def rows(self, conn: socket.socket, service, backend: Any, epoch: int) -> socket.socket | None:
        calls, busy = 0, 0.0
        by_op: dict[tuple[int, int], list[float]] = {}
        while True:
            ready, _, _ = select.select([self.listener] + ([conn] if service.kind != "tcp" else []), [], [], 0)
            if self.listener in ready:
                newer, peer = self.listener.accept()
                self.log(f"[stage] session from {peer[0]} replaces epoch {epoch}")
                conn.close()
                return newer
            if conn in ready:                         # mailbox sessions: the control socket only closes
                # (the stage keeps its daemon until the next session restarts it)
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
            service.reply(seq, list(reply) if isinstance(reply, (list, tuple)) else [reply])
            spent = time.perf_counter() - t
            calls += 1
            busy += spent
            key = (head["op"], min(head["rows"], 16) if head["op"] == wire.FORWARD else 0)
            by_op.setdefault(key, []).append(spent)
            if calls % 500 == 0:
                names = {wire.PREFILL: "prefill", wire.FORWARD: "fwd", wire.FLUSH: "flush", wire.FETCH: "fetch"}
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
    stage.add_argument("--rdma", default="", metavar="DEVICE",
                       help="run the RDMA mailbox daemon on this device (e.g. rocep1s0f1) for Macs that ask for "
                            "--split-transport rdma; its GID is found by IPv4 address")
    stage.add_argument("--rdma-ip", default="", help="with --rdma: the device's RoCE address to use (default: its first)")
    stage.add_argument("--rdma-dir", default="", help="with --rdma: directory with v41rpcd, v41rpc_mailbox.py "
                       "and libv41rpc (default: --mailbox-py, or TF_SPLIT_RDMA_DIR)")
    stage.add_argument("--rdma-port", type=int, default=18620, help="with --rdma: the daemon's control port")
    stage.set_defaults(func=cmd_stage)


def cmd_stage(args: argparse.Namespace) -> int:
    import os
    import signal

    log = lambda s: print(s, flush=True)       # noqa: E731
    rdma = None
    mailbox_py = args.mailbox_py
    if args.rdma:
        from tensorfold.split.rdma import SparkDaemon, roce_address

        where = args.rdma_dir or args.mailbox_py or os.environ.get("TF_SPLIT_RDMA_DIR", "")
        if not where or not os.path.isfile(os.path.join(where, "v41rpcd")):
            raise SystemExit("tensorfold stage --rdma: give --rdma-dir, the directory holding v41rpcd")
        address = roce_address(args.rdma, args.rdma_ip)
        rdma = SparkDaemon(os.path.join(where, "v41rpcd"), args.mailbox or "tfsplit", address, args.rdma_port)
        mailbox_py = mailbox_py or where
        log(f"[stage] RDMA on {address.device}: {address.ip}/{address.prefix} ({address.mac}), GID {address.gid_index}")
    sock = args.mailbox_socket or (f"/dev/shm/v41rpc-{args.mailbox}.sock" if args.mailbox else "")
    def stop(signum, frame):                    # SIGTERM too unwinds, so the daemon is stopped below
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        Server(args.listen, args.cache, args.mailbox, sock, mailbox_py, log=log, rdma=rdma).serve_forever()
    finally:
        if rdma is not None:
            rdma.stop()
    return 0
