"""The Mac's end of a split: open a session with a stage, push the weights it lacks, then send rows through ``Link``."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tensorfold.split import checkpoint, wire
from tensorfold.split.channel import MailboxClient, TcpClient


@dataclass
class Session:
    sock: socket.socket
    channel: Any
    epoch: int
    hidden: int
    tap_layers: list[int]
    first: int
    layers: int
    pushed: int = 0
    seconds: float = 0.0
    decode_first: int = 0            # the layer decode windows enter the stage at (prompts enter at ``first``)
    decode_taps: list[int] = field(default_factory=list)
    daemon: Any = None               # the `v41rpcd mac` this session started (``--split-transport rdma``)
    out_width: int = 0               # a reply row's values (0: ``hidden``, the request's)


def _layers(config: dict) -> int:
    text = config.get("text_config", config)
    return int(text["num_hidden_layers"])


def open_session(address: str, model_dir: Path, first: int, *, transport: str = "tcp", mailbox: str = "",
                 mailbox_py: str | None = None, options: dict | None = None, log=print, rdma: dict | None = None
                 ) -> Session:
    """Connect to ``tensorfold stage`` at ``address``, make it hold layers [first, N) of ``model_dir``, and return the open rows path.

    ``transport="rdma"``: the stage starts a fresh mailbox daemon and this machine one toward it; ``rdma`` holds
    ``dir`` (v41rpcd and its Python module), ``socket`` (the daemon's control socket) and this machine's RoCE
    ``ip`` (default: the stage's point-to-point peer) and ``mac``.
    """

    host, _, port = address.rpartition(":") if ":" in address else (address, "", "")
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    layers = _layers(config)
    t = time.perf_counter()
    units = checkpoint.stage_units(model_dir, first, layers, layers)
    total = sum(u.nbytes for u in units)
    log(f"[split] stage layers {first}..{layers - 1} of {layers}: {len(units)} units, {total / 1024**3:.2f} GiB "
        f"(hashed in {time.perf_counter() - t:.1f} s)")
    if transport == "rdma":
        from tensorfold.split.rdma import shutdown_mac

        rdma = dict(rdma or {})
        if not shutdown_mac(rdma["socket"]):    # before the stage restarts the daemon it talked to
            raise RuntimeError(f"a v41rpcd on {rdma['socket']} does not shut down")
    sock = socket.create_connection((host, int(port or wire.PORT)), timeout=30)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    epoch = int.from_bytes(os.urandom(4), "little") | 1
    wire.send_msg(sock, {"version": wire.VERSION, "config": config, "units": [u.describe() for u in units],
                         "first": first, "last": layers, "layers": layers, "epoch": epoch, "transport": transport,
                         "options": options or {}})
    answer = wire.recv_msg(sock)
    if "error" in answer:
        raise RuntimeError(f"stage {address}: {answer['error']}")
    need = set(answer["need"])
    pushed, t = 0, time.perf_counter()
    daemon = channel = None
    if transport == "rdma":                       # the stage's daemon is up: start ours, the weights ride it too
        from tensorfold.split.rdma import MacDaemon, peer_ip

        far = answer["rdma"]
        if not rdma.get("mac"):
            raise RuntimeError("--split-transport rdma needs this machine's RoCE MAC: --split-rdma-mac or MAC_ROCE_MAC")
        local_ip = rdma.get("ip") or peer_ip(far["ip"], int(far["prefix"]))
        t0 = time.perf_counter()
        daemon = MacDaemon(os.path.join(rdma["dir"], "v41rpcd"), rdma["socket"], far["name"], host, int(far["port"]),
                           local_ip=local_ip, local_mac=rdma["mac"], peer_mac=far["mac"],
                           req_mib=int(far["req_mib"]), rep_mib=int(far["rep_mib"]))
        daemon.start()
        log(f"[split] RDMA mailbox {far['name']}: {local_ip} ({rdma['mac']}) <-> {far['ip']} ({far['mac']}), "
            f"connected in {time.perf_counter() - t0:.2f} s")
        channel = MailboxClient(far["name"], rdma["dir"])
    if need:
        todo = [u for u in units if u.key in need]
        size = sum(u.nbytes for u in todo)
        log(f"[split] pushing {len(todo)} units ({size / 1024**3:.2f} GiB) to the stage over "
            f"{'RDMA and TCP at once' if channel is not None else 'TCP'}; it has {answer['have']}")
        progress = _Progress(size, t, log)
        if channel is not None:
            # both links at once, split by bytes in proportion to what each carries (RoCE a little faster)
            over_rdma, over_tcp, a, b = [], [], 0, 0
            for u in sorted(todo, key=lambda u: -u.nbytes):
                if a <= b * 1.1:
                    over_rdma.append(u)
                    a += u.nbytes
                else:
                    over_tcp.append(u)
                    b += u.nbytes
            wire.send_msg(sock, {"paths": {"rdma": [u.key for u in over_rdma], "tcp": [u.key for u in over_tcp]}})
            index = {u.key: i for i, u in enumerate(over_rdma)}
            failed: list[BaseException] = []

            def tcp() -> None:
                try:
                    for u in over_tcp:
                        wire.send_msg(sock, {"unit": u.key})
                        checkpoint.read_unit(u, sock.sendall)
                        progress.add(u.nbytes)
                except BaseException as exc:       # noqa: BLE001 - raised below
                    failed.append(exc)

            import threading

            thread = threading.Thread(target=tcp, name="split-push-tcp", daemon=True)
            thread.start()
            try:
                for u, n in _push_mailbox(channel, over_rdma, index, epoch):
                    progress.add(n)
            finally:
                thread.join()
            if failed:
                raise failed[0]
            wire.send_msg(sock, {"pushed": True})
        else:
            for u in todo:
                wire.send_msg(sock, {"unit": u.key})
                checkpoint.read_unit(u, sock.sendall)
                progress.add(u.nbytes)
        pushed = progress.done
        spent = time.perf_counter() - t
        log(f"[split] pushed {pushed / 1024**3:.2f} GiB in {spent:.1f} s ({pushed * 8 / spent / 1e9:.1f} Gbit/s)")
    else:
        log(f"[split] the stage holds all {len(units)} units already")
    ready = wire.recv_msg(sock)                     # after the stage has loaded its layers
    if not ready.get("ready"):
        raise RuntimeError(f"stage {address}: {ready.get('error', ready)}")
    if transport == "mailbox":
        channel = MailboxClient(mailbox or ready.get("mailbox", ""), mailbox_py)
    elif channel is None:
        channel = TcpClient(sock)
    spent = time.perf_counter() - t
    log(f"[split] stage ready over {transport} (epoch {epoch}, taps {ready['taps']}, "
        f"{'pushed ' + format(pushed / 1024**3, '.2f') + ' GiB, ' if pushed else ''}{spent:.1f} s)")
    return Session(sock, channel, epoch, int(ready["hidden"]), list(ready["taps"]), first, layers, pushed, spent,
                   int(ready.get("decode_first", first)), list(ready.get("decode_taps", ready["taps"])), daemon,
                   int(ready.get("out_width", ready["hidden"])))


class _Progress:
    def __init__(self, size: int, t: float, log) -> None:
        import threading

        self.size, self.t, self.log, self.done, self.last = size, t, log, 0, t
        self.lock = threading.Lock()

    def add(self, n: int) -> None:
        with self.lock:
            self._add(n)

    def _add(self, n: int) -> None:
        self.done += n
        now = time.perf_counter()
        if now - self.last > 5:
            self.log(f"[split]   {self.done / 1024**3:.1f} / {self.size / 1024**3:.1f} GiB, "
                     f"{self.done * 8 / (now - self.t) / 1e9:.1f} Gbit/s")
            self.last = now


def _push_mailbox(channel, todo: list, index: dict[str, int], epoch: int):
    """Each unit's bytes as PUSH pieces through the mailbox; a thread reads the next piece while one crosses."""

    import queue
    import threading

    pieces: queue.Queue = queue.Queue(maxsize=2)

    def read() -> None:
        try:
            for u in todo:
                buf = bytearray()
                count = 0

                def sink(chunk: bytes) -> None:
                    nonlocal buf, count
                    buf += chunk
                    while len(buf) >= wire.PIECE:
                        pieces.put((u, count, bytes(buf[:wire.PIECE]), False))
                        del buf[:wire.PIECE]
                        count += 1

                checkpoint.read_unit(u, sink)
                pieces.put((u, count, bytes(buf), True))           # the last piece, maybe short
            pieces.put(None)
        except BaseException as exc:       # noqa: BLE001 - raised on the pushing thread
            pieces.put(exc)

    threading.Thread(target=read, name="split-push-reader", daemon=True).start()
    while True:
        item = pieces.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        u, count, data, last = item
        channel.submit([wire.request(wire.PUSH, epoch=epoch, rows=0, start=count, extra=index[u.key], n_commit=0,
                                     flags=wire.LAST if last else 0), data])
        reply = channel.result()
        status, n, _ = wire.REPLY.unpack_from(reply, 0)
        if status:
            raise RuntimeError(f"split stage: {bytes(reply[12:12 + n]).decode(errors='replace')}")
        yield u, len(data)


class Link:
    """Rows to the stage and its final normed rows back; at most one request in flight."""

    def __init__(self, session: Session) -> None:
        self.s = session
        self.path: list[int] = []      # the last decode window's accepted path, sent with the next request
        self.flags = 0                 # wire.TAPS once a drafter reads the stage's tap layers
        self.rows_out = 0
        self.calls = 0
        self.wait_s = 0.0
        self.rounds: list[tuple[float, float, int]] = []
        self.last_reply = 0.0
        self._sent = 0.0

    @property
    def hidden(self) -> int:
        return self.s.hidden

    def submit(self, op: int, start: int, x: Any = None, pending: Any = None, parents: list[int] | None = None,
               want: int = 0, layer: int = 0, one: bool = False) -> None:
        """Send a request; ``x``/``pending`` are evaluated MLX rows [.., R, H] (pending None: zeros)."""

        import mlx.core as mx

        rows = 0 if x is None else int(x.shape[-2])
        extra = len(parents) if parents is not None else want
        flags = (self.flags if x is not None else 0) | (wire.ONE if one else 0)
        parts: list[Any] = [wire.request(op, epoch=self.s.epoch, rows=rows, start=start, extra=extra,
                                         n_commit=len(self.path), flags=flags, layer=layer),
                            np.asarray(self.path, dtype=np.int32)]
        self.path = []
        if parents is not None:
            parts.append(np.asarray(parents, dtype=np.int32))
        if x is not None and one:
            x = x.reshape(rows, -1).astype(mx.bfloat16)
            mx.eval(x)
            parts.append(np.array(x.view(mx.uint16), copy=False))
        elif x is not None:
            if pending is None:
                pending = mx.zeros_like(x)
            x = x.reshape(rows, -1).astype(mx.bfloat16)
            pending = pending.reshape(rows, -1).astype(mx.bfloat16)
            mx.eval(x, pending)
            parts += [np.array(x.view(mx.uint16), copy=False), np.array(pending.view(mx.uint16), copy=False)]
        self._sent = time.perf_counter()
        self._op, self._rows = op, rows
        if op == wire.FORWARD:
            self.rounds.append((self._sent - self.last_reply if self.last_reply else 0.0, 0.0, rows))
        self.s.channel.submit(parts)

    def collect(self) -> tuple[Any, list[Any]]:
        """The reply to the request in flight: normed rows [1, n, H] (or None) and the stage's taps, each [1, R, H]."""

        import mlx.core as mx

        reply = self.s.channel.result()
        now = time.perf_counter()
        self.wait_s += now - self._sent
        self.calls += 1
        self.last_reply = now
        if self._op == wire.FORWARD and self.rounds:
            mac, _, rows = self.rounds[-1]
            self.rounds[-1] = (mac, now - self._sent, rows)
            if len(self.rounds) >= 300:
                p50 = lambda k: sorted(r[k] for r in self.rounds)[len(self.rounds) // 2]   # noqa: E731
                print(f"[split] {len(self.rounds)} rounds, p50: Mac side {p50(0) * 1e3:.1f} ms, "
                      f"stage wait {p50(1) * 1e3:.1f} ms, rows {p50(2)}", flush=True)
                self.rounds.clear()
        status, n, count = wire.REPLY.unpack_from(reply, 0)
        if status:
            raise RuntimeError(f"split stage: {bytes(reply[12:12 + n]).decode(errors='replace')}")
        body = np.frombuffer(bytes(reply[wire.REPLY.size:]), dtype=np.uint16)   # copied out of the mailbox
        h = self.s.out_width or self.hidden
        normed = mx.array(body[:n * h]).view(mx.bfloat16).reshape(1, n, h) if n else None
        taps = []
        if count:
            rows = self._rows
            block = mx.array(body[n * h:]).view(mx.bfloat16).reshape(count, 1, rows, h)
            taps = [block[j] for j in range(count)]
        return normed, taps

    def fetch(self, layer: int, start: int = 0, rows: int = 0, flags: int = 0) -> tuple[int, bytes]:
        """State the stage computed for the Mac: (rows it holds, its bytes); see ``wire.FETCH``."""

        self.s.channel.submit([wire.request(wire.FETCH, epoch=self.s.epoch, rows=rows, start=start, extra=layer,
                                            n_commit=0, flags=flags)])
        reply = self.s.channel.result()
        status, n, _ = wire.REPLY.unpack_from(reply, 0)
        if status:
            raise RuntimeError(f"split stage: {bytes(reply[12:12 + n]).decode(errors='replace')}")
        return n, bytes(reply[wire.REPLY.size:])

    def call(self, *args, **kwargs) -> tuple[Any, list[Any]]:
        self.submit(*args, **kwargs)
        return self.collect()

    def close(self) -> None:
        try:
            self.s.channel.close()
        finally:
            if self.s.daemon is not None:
                self.s.daemon.stop()
            self.s.sock.close()

