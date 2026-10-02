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
    extra: dict = field(default_factory=dict)


def _layers(config: dict) -> int:
    text = config.get("text_config", config)
    return int(text["num_hidden_layers"])


def open_session(address: str, model_dir: Path, first: int, *, transport: str = "tcp", mailbox: str = "",
                 mailbox_py: str | None = None, options: dict | None = None, log=print) -> Session:
    """Connect to ``tensorfold stage`` at ``address``, make it hold layers [first, N) of ``model_dir``, and return the open rows path."""

    host, _, port = address.rpartition(":") if ":" in address else (address, "", "")
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    layers = _layers(config)
    t = time.perf_counter()
    units = checkpoint.stage_units(model_dir, first, layers, layers)
    total = sum(u.nbytes for u in units)
    log(f"[split] stage layers {first}..{layers - 1} of {layers}: {len(units)} units, {total / 1024**3:.2f} GiB "
        f"(hashed in {time.perf_counter() - t:.1f} s)")
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
    if need:
        todo = [u for u in units if u.key in need]
        size = sum(u.nbytes for u in todo)
        log(f"[split] pushing {len(todo)} units ({size / 1024**3:.2f} GiB) to the stage; it has {answer['have']}")
        last_log = t
        for u in todo:
            wire.send_msg(sock, {"unit": u.key})
            checkpoint.read_unit(u, sock.sendall)
            pushed += u.nbytes
            now = time.perf_counter()
            if now - last_log > 5:
                log(f"[split]   {pushed / 1024**3:.1f} / {size / 1024**3:.1f} GiB, "
                    f"{pushed * 8 / (now - t) / 1e9:.1f} Gbit/s")
                last_log = now
    else:
        log(f"[split] the stage holds all {len(units)} units already")
    ready = wire.recv_msg(sock)                     # after the stage has loaded its layers
    if not ready.get("ready"):
        raise RuntimeError(f"stage {address}: {ready.get('error', ready)}")
    channel = MailboxClient(mailbox or ready.get("mailbox", ""), mailbox_py) if transport == "mailbox" else TcpClient(sock)
    spent = time.perf_counter() - t
    log(f"[split] stage ready over {transport} (epoch {epoch}, taps {ready['taps']}, "
        f"{'pushed ' + format(pushed / 1024**3, '.2f') + ' GiB, ' if pushed else ''}{spent:.1f} s)")
    return Session(sock, channel, epoch, int(ready["hidden"]), list(ready["taps"]), first, layers, pushed, spent)


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
               want: int = 0) -> None:
        """Send a request; ``x``/``pending`` are evaluated MLX rows [.., R, H] (pending None: zeros)."""

        import mlx.core as mx

        rows = 0 if x is None else int(x.shape[-2])
        extra = len(parents) if parents is not None else want
        flags = self.flags if x is not None else 0
        parts: list[Any] = [wire.request(op, epoch=self.s.epoch, rows=rows, start=start, extra=extra,
                                         n_commit=len(self.path), flags=flags),
                            np.asarray(self.path, dtype=np.int32)]
        self.path = []
        if parents is not None:
            parts.append(np.asarray(parents, dtype=np.int32))
        if x is not None:
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
        h = self.hidden
        normed = mx.array(body[:n * h]).view(mx.bfloat16).reshape(1, n, h) if n else None
        taps = []
        if count:
            rows = self._rows
            block = mx.array(body[n * h:]).view(mx.bfloat16).reshape(count, 1, rows, h)
            taps = [block[j] for j in range(count)]
        return normed, taps

    def call(self, *args, **kwargs) -> tuple[Any, list[Any]]:
        self.submit(*args, **kwargs)
        return self.collect()

    def close(self) -> None:
        try:
            self.s.channel.close()
        finally:
            self.s.sock.close()

