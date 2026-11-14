from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .schema import Fact, normalized


class LedgerError(RuntimeError):
    pass


def private_dir(directory: Path) -> None:
    # Directories this tool creates are 0700; an existing one is left as it is, since it may hold other state.
    missing = []
    while not directory.exists():
        missing.append(directory)
        directory = directory.parent
    for item in reversed(missing):
        try:
            item.mkdir(mode=0o700)
        except FileExistsError:
            pass


def private_journal(path: Path) -> None:
    """The database, WAL and SHM hold fact text: refuse ones others can open, and create the database 0600."""
    for item in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            mode = stat.S_IMODE(item.stat().st_mode)
        except FileNotFoundError:
            continue
        if mode & 0o077:
            raise LedgerError(f"{item} is open to other users (mode {mode:o}) and holds fact text; "
                              "run chmod 600 on it first")
    private_dir(path.parent)
    try:
        os.close(os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
    except FileExistsError:
        pass


@contextlib.contextmanager
def exclusive_file(path: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:
        raise LedgerError("exclusive locking requires a POSIX host") from exc
    private_dir(path.parent)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError as exc:
        raise LedgerError("another Living Weights v2 writer is active") from exc
    finally:
        os.close(descriptor)


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        private_journal(path)  # SQLite gives the WAL and SHM the database file's mode
        self.db = sqlite3.connect(path, timeout=2)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                fact_digest TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                answer TEXT NOT NULL,
                fact_json TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                state TEXT NOT NULL,
                receipt_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE INDEX IF NOT EXISTS entries_model_state ON entries(model, state);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def require_safe(self, model: str, fact: Fact) -> None:
        pending = self.db.execute("SELECT id, state FROM entries WHERE model=? AND state IN ('inflight','needs_recovery')", (model,)).fetchall()
        if pending:
            raise LedgerError(f"model has unresolved write records {[int(row['id']) for row in pending]}; recovery required")
        entries = self.db.execute("SELECT answer, fact_digest FROM entries WHERE model=? AND subject_key=? AND state='committed'", (model, fact.key)).fetchall()
        for row in entries:
            if normalized(row["answer"]) != normalized(fact.answer):
                raise LedgerError("contradictory fact about the same subject/relation; explicit delete and independent validation required")
            if row["fact_digest"] == fact.digest:
                raise LedgerError("identical fact already committed; teaching again is not idempotent")
            raise LedgerError("subject/relation already learned with different evidence; review required")

    def retained(self, model: str) -> tuple[Fact, ...]:
        entries = self.db.execute("SELECT fact_json FROM entries WHERE model=? AND state='committed' ORDER BY id", (model,)).fetchall()
        result = []
        for row in entries:
            value = json.loads(row["fact_json"])
            value["questions"] = tuple(value["questions"])
            value["near_misses"] = tuple(value["near_misses"])
            result.append(Fact(**value))
        return tuple(result)

    def begin(self, model: str, manifest_digest: str, fact: Fact) -> int:
        self.require_safe(model, fact)
        with self.db:
            cursor = self.db.execute("INSERT INTO entries(model, fact_digest, subject_key, answer, fact_json, manifest_digest, state) VALUES(?,?,?,?,?,?,?)", (
                model, fact.digest, fact.key, fact.answer, json.dumps(asdict(fact), ensure_ascii=False), manifest_digest, "inflight",
            ))
        return int(cursor.lastrowid)

    def finish(self, entry_id: int, state: str, receipt: dict) -> None:
        if state not in {"committed", "skipped", "rejected", "rolled_back", "needs_recovery"}:
            raise LedgerError("invalid final state")
        with self.db:
            cursor = self.db.execute("UPDATE entries SET state=?, receipt_json=?, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=? AND state='inflight'", (
                state, json.dumps(receipt, sort_keys=True, ensure_ascii=False), entry_id,
            ))
            if cursor.rowcount != 1:
                raise LedgerError("no matching inflight entry")

    def history(self, model: str | None = None) -> list[dict]:
        if model:
            rows = self.db.execute("SELECT id, model, subject_key, state, receipt_json, created_at FROM entries WHERE model=? ORDER BY id", (model,))
        else:
            rows = self.db.execute("SELECT id, model, subject_key, state, receipt_json, created_at FROM entries ORDER BY id")
        return [{**dict(row), "receipt": json.loads(row["receipt_json"])} for row in rows]
