"""What the engine served, kept for months: tokens and requests by model and hour, peak speeds, when it last served.

The engine counts tokens and requests since it started. Each reading of the history thread is compared with the
previous one, kept in the database so a TensorFold Web restart loses nothing: the difference goes to the current hour's row
for the model served. Totals lower than before mean the engine restarted; its new totals are then all new. A first
reading with nothing to compare with only sets the baseline.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

KEEP_DAYS = 400
HOUR = 3600
KINDS = ("prompt", "completion", "cached", "requests")


class Usage:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("create table if not exists usage (hour integer, model text, prompt integer default 0, "
                       "completion integer default 0, cached integer default 0, requests integer default 0, "
                       "peak_decode real, peak_prefill real, primary key (hour, model))")
            db.execute("create table if not exists state (k text primary key, v text)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _get(self, db: sqlite3.Connection, k: str) -> dict | None:
        row = db.execute("select v from state where k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, db: sqlite3.Connection, k: str, v: dict) -> None:
        db.execute("insert or replace into state (k, v) values (?, ?)", (k, json.dumps(v)))

    def record(self, info: dict, t: float | None = None) -> None:
        """One reading of the engine: its totals' growth since the previous reading, and its speeds."""

        t = time.time() if t is None else t
        if info.get("status") != "running" or not info.get("model"):
            return
        model, totals = str(info["model"]), info.get("totals")
        hour = int(t // HOUR * HOUR)
        decode, prefill = info.get("decode_tps"), info.get("prefill_tps")
        with self._lock, self._db() as db:
            delta = dict.fromkeys(KINDS, 0)
            if isinstance(totals, dict):
                now = {k: int(totals.get(k) or 0) for k in KINDS}
                # one baseline per server: going back to a server counts what it served meanwhile, and a baseline
                # kept before servers were told apart (no address) is not trusted for any of them
                url = info.get("engine_url")
                prev = self._get(db, f"totals@{url}") if url else self._get(db, "totals")
                if prev is None and url:              # kept before one per server: this server's when its address says so
                    legacy = self._get(db, "totals")
                    prev = legacy if legacy and legacy.get("url") == url else None
                # the engine's start time when TensorFold Web knows it: the same model restarted between two readings
                # can pass its old totals, and only this tells the two engines apart
                started = info.get("started_at")
                same_engine = not (started and prev and prev.get("started")
                                   and abs(float(started) - float(prev["started"])) > 2.0)
                if prev and same_engine and prev.get("model") == model and \
                        all(now[k] >= prev["totals"].get(k, 0) for k in KINDS):
                    delta = {k: now[k] - prev["totals"].get(k, 0) for k in KINDS}
                elif prev:                             # another model, or the engine restarted: all of it is new
                    delta = now
                self._set(db, f"totals@{url}" if url else "totals",
                          {"model": model, "totals": now, "t": t, "started": started, "url": url})
            db.execute("insert into usage (hour, model) values (?, ?) on conflict do nothing", (hour, model))
            db.execute("update usage set prompt = prompt + ?, completion = completion + ?, cached = cached + ?, "
                       "requests = requests + ?, peak_decode = max(coalesce(peak_decode, 0), ?), "
                       "peak_prefill = max(coalesce(peak_prefill, 0), ?) where hour = ? and model = ?",
                       (delta["prompt"], delta["completion"], delta["cached"], delta["requests"],
                        float(decode or 0), float(prefill or 0), hour, model))
            if delta["completion"] > 0 or delta["requests"] > 0 or (info.get("connections") or 0) > 0:
                self._set(db, "last_served", {"t": t, "model": model})
            db.execute("delete from usage where hour < ?", (hour - KEEP_DAYS * 86400,))

    def last_served(self) -> dict | None:
        with self._db() as db:
            return self._get(db, "last_served")

    def summary(self, days: int = 30, hours: int = 48, now: float | None = None) -> dict:
        """Days (this machine's local dates) and recent hours, tokens by model, and the peak speeds."""

        now = time.time() if now is None else now
        start = int((now - days * 86400) // HOUR * HOUR)
        with self._db() as db:
            rows = db.execute("select hour, model, prompt, completion, cached, requests, peak_decode, peak_prefill "
                              "from usage where hour >= ? order by hour", (start,)).fetchall()
            last = self._get(db, "last_served")
        by_day: dict[str, dict] = {}
        by_hour: dict[int, dict] = {}
        models: dict[str, dict] = {}
        for hour, model, prompt, completion, cached, requests, pd, pp in rows:
            day = time.strftime("%Y-%m-%d", time.localtime(hour))      # this machine's own dates
            d = by_day.setdefault(day, {"day": day, "models": {}, "peak_decode": 0.0, "peak_prefill": 0.0,
                                        "requests": 0})
            dm = d["models"].setdefault(model, {"prompt": 0, "completion": 0})
            dm["prompt"] += prompt
            dm["completion"] += completion
            d["requests"] += requests
            d["peak_decode"] = max(d["peak_decode"], pd or 0.0)
            d["peak_prefill"] = max(d["peak_prefill"], pp or 0.0)
            m = models.setdefault(model, {"model": model, "prompt": 0, "completion": 0, "cached": 0, "requests": 0})
            for k, v in (("prompt", prompt), ("completion", completion), ("cached", cached), ("requests", requests)):
                m[k] += v
            if hour >= now - hours * HOUR:
                h = by_hour.setdefault(hour, {"hour": hour, "models": {}})
                hm = h["models"].setdefault(model, {"prompt": 0, "completion": 0})
                hm["prompt"] += prompt
                hm["completion"] += completion
        return {"days": list(by_day.values()), "hours": list(by_hour.values()),
                "models": sorted(models.values(), key=lambda m: -m["completion"]), "last_served": last,
                "since": rows[0][0] if rows else None}
