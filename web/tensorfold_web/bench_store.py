"""Bench store."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class Store:
    """Runs and their results in one SQLite file; a run's receipt is built from both."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._db() as db:
            db.executescript("""
                create table if not exists runs (id text primary key, created real not null, finished real,
                    mode text not null, status text not null, model text, doc text not null);
                create table if not exists results (run_id text not null references runs(id) on delete cascade,
                    seq integer not null, suite text not null, prompt text not null, draft integer not null,
                    doc text not null, primary key (run_id, seq));
            """)
            # a TensorFold Web stopped mid-run leaves it running in the file: it is not anymore
            for rid, doc in db.execute("select id, doc from runs where status='running'").fetchall():
                run = {**json.loads(doc), "status": "interrupted", "progress": None}
                db.execute("update runs set status='interrupted', doc=? where id=?", (json.dumps(run), rid))

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("pragma foreign_keys = on")
        return db

    def save_run(self, run: dict) -> None:
        with self._lock, self._db() as db:
            db.execute("insert into runs (id, created, finished, mode, status, model, doc) values (?,?,?,?,?,?,?) "
                       "on conflict(id) do update set finished=excluded.finished, status=excluded.status, "
                       "doc=excluded.doc",
                       (run["id"], run["created"], run.get("finished"), run["mode"], run["status"], run.get("model"),
                        json.dumps(run)))

    def add_result(self, run_id: str, seq: int, r: dict) -> None:
        with self._lock, self._db() as db:
            db.execute("insert into results values (?,?,?,?,?,?)",
                       (run_id, seq, r["suite"], r["prompt"], 1 if r["draft"] else 0, json.dumps(r)))

    def run(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("select doc from runs where id=?", (run_id,)).fetchone()
            if not row:
                return None
            results = [json.loads(d) for (d,) in db.execute("select doc from results where run_id=? order by seq",
                                                             (run_id,))]
        return {**json.loads(row[0]), "results": results}

    def runs(self, limit: int = 50) -> list[dict]:
        with self._db() as db:
            rows = db.execute("select doc from runs order by created desc limit ?", (limit,)).fetchall()
        out = []
        for (d,) in rows:
            run = json.loads(d)
            out.append({k: run.get(k) for k in ("id", "created", "finished", "mode", "status", "model", "engine",
                                                "summary", "error")})
            if run.get("kl"):
                out[-1]["kl"] = {"mean_kl": run["kl"].get("mean_kl")}     # the list's figure for a KL run
        return out

    def delete(self, run_id: str) -> bool:
        with self._lock, self._db() as db:
            return db.execute("delete from runs where id=?", (run_id,)).rowcount > 0
