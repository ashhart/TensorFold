"""Figures kept by the server for a few hours, so Monitor draws its charts as soon as a page opens.

A thread reads the engine every few seconds (the reading a page just made, when it is fresh) and keeps one row per
reading in SQLite: decode and prefill speed, connections, KV fill, the engine's memory and the machine's free memory.
Rows older than the retention go. A reading the engine did not answer is kept as a row of nothing, so a chart draws a
gap there instead of a line across it.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

EVERY_S = 5.0
KEEP_S = 8 * 3600.0
FIELDS = ("decode_tps", "prefill_tps", "connections", "kv_fill", "engine_bytes", "available_bytes")


def row_of(info: dict) -> dict:
    """The figures of one reading; None where the engine is down or did not report them."""

    if info.get("status") != "running":
        return dict.fromkeys(FIELDS)
    mem, mm = info.get("memory") or {}, info.get("machine_memory") or {}
    engine = next((mem[k] for k in ("active", "footprint") if isinstance(mem.get(k), (int, float))), None)
    return {"decode_tps": info.get("decode_tps"), "prefill_tps": info.get("prefill_tps"),
            "connections": info.get("connections"), "kv_fill": info.get("kv_fill"),
            "engine_bytes": engine if engine is not None else mm.get("engine"),
            "available_bytes": mm.get("available")}


class History:
    def __init__(self, path: Path, read: Callable[[], dict], every: float = EVERY_S, keep: float = KEEP_S,
                 then: Callable[[dict], None] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.read, self.every, self.keep, self.then = path, read, every, keep, then
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        with self._db() as db:
            db.execute(f"create table if not exists history (t real primary key, {', '.join(f + ' real' for f in FIELDS)})")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def add(self, info: dict, t: float | None = None) -> None:
        r = row_of(info)
        t = time.time() if t is None else t
        with self._db() as db:
            db.execute(f"insert or replace into history (t, {', '.join(FIELDS)}) values (?{', ?' * len(FIELDS)})",
                       (t, *(r[f] for f in FIELDS)))
            db.execute("delete from history where t < ?", (t - self.keep,))

    def since(self, seconds: float) -> dict:
        """Columns for a chart: the times, then each figure, oldest first; `every` lets the page see gaps."""

        start = time.time() - seconds
        with self._db() as db:
            rows = db.execute(f"select t, {', '.join(FIELDS)} from history where t >= ? order by t", (start,)).fetchall()
        cols = list(zip(*rows, strict=True)) if rows else [()] * (len(FIELDS) + 1)
        return {"every": self.every, "from": start, "t": list(cols[0]),
                **{f: list(cols[i + 1]) for i, f in enumerate(FIELDS)}}

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            with contextlib.suppress(Exception):       # a missed reading is a gap in the chart, not a stopped thread
                info = self.read()
                self.add(info)
                if self.then:
                    self.then(info)                    # the same reading feeds the usage history

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="history", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
